import torch
import torch.nn.functional as F
from torch import nn, optim

# K = num_topics
# V = vocab_size
# M = embedding_size
# D = num_documents

# rho = word embeddings matrix (M, V)
# alpha_k = topic embedding (M, 1)
# alpha = topic embeddings matrix (M, K)
# theta_d = topic proportion for doc d (K, 1)
# theta = topic proportions matrix (K, D)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ETM(nn.Module):
    def __init__(self, num_topics=50, vocab_size=300, embedding_size=300, flag_finetune_embeddings=False,
                 embeddings=None, t_hidden_size=800, theta_act='relu', enc_drop=0.0):
        super(ETM, self).__init__()

        ## define hyperparameters
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.t_hidden_size = t_hidden_size

        ## define the word embedding matrix \rho
        if embeddings is not None:
            embeddings = torch.from_numpy(embeddings).clone().float().to(device)
            if flag_finetune_embeddings:
                self.rho = nn.Embedding.from_pretrained(embeddings, freeze=False)
            else:
                self.rho = nn.Embedding.from_pretrained(embeddings, freeze=True)
        else:
            self.rho = nn.Linear(embedding_size, vocab_size, bias=False)

        ## define the matrix containing the topic embeddings
        self.alphas = nn.Linear(embedding_size, num_topics, bias=False)

        ## define variational distribution for \theta_{1:D} via amortizartion
        self.q_theta = nn.Sequential(
                nn.Linear(vocab_size, t_hidden_size),
                self.get_activation(theta_act),
                nn.Linear(t_hidden_size, t_hidden_size),
                self.get_activation(theta_act),
                nn.Dropout(enc_drop)
            )
        self.mu_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)

    def get_activation(self, act):
        if act == 'tanh':
            act = nn.Tanh()
        elif act == 'relu':
            act = nn.ReLU()
        else:
            print('Defaulting to tanh activations...')
            act = nn.Tanh()
        return act

    # beta = Softmax(rho.T x alpha)
    def get_beta(self):
        logit = self.alphas(self.rho.weight)
        beta = F.softmax(logit, dim=0).transpose(1, 0)
        return beta

    # (mu, sigma) = NeuralNetwork (with variational params)
    # delta ~ N(mu, sigma) (re-parameterization trick)
    # theta = Softmax(delta)
    def reparameterize(self, mu, logvar):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def encode(self, corpus):
        q_theta = self.q_theta(corpus)
        mu_theta = self.mu_q_theta(q_theta)
        logsigma_theta = self.logsigma_q_theta(q_theta)
        return mu_theta, logsigma_theta

    def get_theta(self, normalized_corpus):
        mu_theta, logsigma_theta = self.encode(normalized_corpus)
        delta = self.reparameterize(mu_theta, logsigma_theta)
        theta = F.softmax(delta, dim=-1)
        KL_p_q = -0.5 * torch.sum(1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp(), dim=-1).mean()
        return theta, KL_p_q

    # log p(w_dn | theta) = log(sum(theta.T x beta))
    def decode(self, theta, beta, eps=1e-6):
        p_w = torch.mm(theta, beta)
        log_p_w = torch.log(p_w + eps)
        return log_p_w

    # loss: ELBO(q) = E[log p(w|z)] - KL(p(z)||q(z))
    def forward(self, corpus, normalized_corpus):
        # get theta
        theta, KL_p_q = self.get_theta(normalized_corpus)

        # get beta
        beta = self.get_beta()

        log_p_w = self.decode(theta, beta)
        E_log_p_w = -(log_p_w * corpus).sum(1)
        E_log_p_w = E_log_p_w.mean()
        #ELBO =
        #loss =
        return E_log_p_w, KL_p_q
