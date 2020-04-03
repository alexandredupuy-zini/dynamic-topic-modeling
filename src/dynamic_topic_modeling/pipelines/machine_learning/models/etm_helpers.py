def get_batch(tokens, counts, ind, vocab_size, device, emsize=300):
    """fetch input data by batch."""
    batch_size = len(ind)
    data_batch = np.zeros((batch_size, vocab_size))

    for i, doc_id in enumerate(ind):
        doc = tokens[doc_id]
        count = counts[doc_id]
        L = count.shape[1]
        if len(doc) == 1:
            doc = [doc.squeeze()]
            count = [count.squeeze()]
        else:
            doc = doc.squeeze()
            count = count.squeeze()
        if doc_id != -1:
            for j, word in enumerate(doc):
                data_batch[i, word] = count[j]
    data_batch = torch.from_numpy(data_batch).float().to(device)
    return data_batch

def train_step(model, optimizer, num_docs_train, train_tokens, train_counts, vocab_size, device, epoch=10, batch_size=1000):
    model.train()

    acc_loss = 0
    acc_kl_theta_loss = 0
    cnt = 0

    indices = torch.randperm(num_docs_train)
    indices = torch.split(indices, batch_size)
    for idx, ind in enumerate(indices):

        optimizer.zero_grad()
        model.zero_grad()

        data_batch = get_batch(train_tokens, train_counts, ind, vocab_size, device)

        sums = data_batch.sum(1).unsqueeze(1)
        normalize_batch = True
        if normalize_batch:
            normalized_data_batch = data_batch / sums
        else:
            normalized_data_batch = data_batch

        E_log_p_w, KL_p_q = model(data_batch, normalized_data_batch)
        total_loss = E_log_p_w + KL_p_q
        total_loss.backward()

        clip = 0
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        acc_loss += torch.sum(E_log_p_w).item()
        acc_kl_theta_loss += torch.sum(KL_p_q).item()
        cnt += 1

        #log_interval = 2
        #if idx % log_interval == 0 and idx > 0:
            #cur_loss = round(acc_loss / cnt, 2)
            #cur_kl_theta = round(acc_kl_theta_loss / cnt, 2)
            #cur_real_loss = round(cur_loss + cur_kl_theta, 2)

            #print('Epoch: {} .. batch: {}/{} .. LR: {} .. KL_theta: {} .. Rec_loss: {} .. NELBO: {}'.format(
            #    epoch, idx, len(indices), optimizer.param_groups[0]['lr'], cur_kl_theta, cur_loss, cur_real_loss))

    cur_loss = round(acc_loss / cnt, 2)
    cur_kl_theta = round(acc_kl_theta_loss / cnt, 2)
    cur_real_loss = round(cur_loss + cur_kl_theta, 2)
    print('*'*100)
    print('Epoch----->{} .. LR: {} .. KL_theta: {} .. Rec_loss: {} .. NELBO: {}'.format(
            epoch, optimizer.param_groups[0]['lr'], cur_kl_theta, cur_loss, cur_real_loss))
    print('*'*100)

def train(model, optimizer, num_docs_train, train_tokens, train_counts, vocab_size, device, epochs=10, batch_size=1000, visualize_every=10):
    #best_epoch = 0
    #best_val_ppl = 1e9
    #all_val_ppls = []
    #print('\n')
    #print('Visualizing model quality before training...')
    #visualize(model)
    #print('\n')
    for epoch in range(1, epochs):
        train_step(model, optimizer, num_docs_train, train_tokens, train_counts, vocab_size, device, epoch, batch_size)
        #val_ppl = evaluate(model, 'val')
        #if epoch % visualize_every == 0:
            #visualize(model)
        #all_val_ppls.append(val_ppl)
    model = model.to(device)
    #val_ppl = evaluate(model, 'val')
