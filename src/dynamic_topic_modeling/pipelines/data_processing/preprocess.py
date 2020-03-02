from typing import Any, Dict

import pandas as pd

from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from sklearn.feature_extraction.text import CountVectorizer
from gensim.corpora import Dictionary,MmCorpus
import numpy as np
import re 
from kedro.context import load_context

from time import time 
import sys


def get_frequency_and_vocab(tokenized_texts,list_to_remove) : 
	texts=[' '.join(text).strip() for text in tokenized_texts]
	vectorizer=CountVectorizer()
	features=vectorizer.fit_transform(texts)
	vocab_count=vectorizer.vocabulary_
	count=0
	for word in list_to_remove :
		if word in vocab_count.keys() : 
			count+=np.sum(features[:,vocab_count[word]])
	return np.round((count/np.sum(features))*100,3)
	
	
def preprocess_dataset(dataset : pd.DataFrame , extreme_no_below: int, extreme_no_above: float, enable_bigram: bool, 
					   min_bigram_count: int, basic_word_analysis : bool) -> Dict[str, Any]:
	"""Node for preprocessing the UN General Debates dataset.
	Parameters are taken from conf/base/parameters.yml.
	The data and the parameters will be loaded and provided to this function
	automatically when the pipeline is executed and it is time to run this node.	

		Args:
			dataset: Source data. Must have a column named "text" to be processed. Dataset must be in catalog.yml
		Returns:
			Preprocessed dataset,
			corpus,
			dictionnary
		Parameters : 
			extreme_no_below : if >1 : for a word w, delete this word from vocabulary if w in less than extreme_no_below documents. if in [0,1], for a word w, delete this word from vocabulary if w in less than extreme_no_below% documents
			extreme_no_above : if >1 : for a word w, delete this word from vocabulary if w in more than extreme_no_below documents. if in [0,1], for a word w, delete this word from vocabulary if w in more than extreme_no_below% documents
			enable_bigram : Boolean, decide if you want bigrams or not in the dictionary
			min_bigram_count : Int, threshold for bigrams :  Bigram will be added to the dictionary if in more than min_bigram_count documents
			basic_word_analysis : Boolean, set to True if you want to print some basic word anaylis (basically the number of words removed from each preprocces steps.)

	"""
	t0=time()

	print('\n\nCurrent set of parameters :\n')
	print('\textreme_no_below : {}'.format(extreme_no_below))
	print('\textreme_no_above : {}'.format(extreme_no_above))
	print('\tenable_bigram : {}'.format(enable_bigram))
	print('\tmin_bigram_count : {}'.format(min_bigram_count))
	print('\tbasic_word_analysis : {}\n'.format(basic_word_analysis))
	print('\nStart preprocessing on dataset')

	if "text" not in dataset.columns : 
		raise ValueError('Dataset does not have a column named "text". You must rename the your text column to "text".')
	if "timestamp" not in dataset.columns : 
		raise ValueError('Dataset does not have a column named "timestamp". You must rename your time column to "timestamp".')

	dataset.sort_values('timestamp',inplace=True)
	dataset.reset_index(drop=True,inplace=True)
	
	dataset['text'] = dataset['text'].str.lower()
	
	stop_words = set(stopwords.words('english'))
	tokenizer = RegexpTokenizer(r'\w+')
	tokenized_texts=[]
	print('Tokenizing............')
	count=0
	for text in dataset['text'] :
		count+=1
		if count % 1000 == 0 :
			print('\tDone tokenizing {}/{} texts'.format(count,len(dataset['text'])))
		tokens=tokenizer.tokenize(text.strip())
		tokenized_texts.append(tokens)
	dataset['text']=tokenized_texts
	del tokenized_texts,tokens,count
	print('\nTokenizing done')

	
	
	if basic_word_analysis :
	
		print('Basic word analysis enabled. It will take more time to compute......\n')
		
		before_texts=dataset['text']
		starting_dict=Dictionary(before_texts)
		sw_in_vocab=[word for word in stop_words if word in starting_dict.values()]
		starting_vocab=len(starting_dict)
		del starting_dict
		
		print('\nBeginning dictionary contains : {} words\n'.format(starting_vocab))

		print('Removing stopwords ......')
		# Remove stopwords from text 
		freq1=get_frequency_and_vocab(dataset['text'],stop_words)
		
		dataset['text'] = [[token for token in doc if token not in sw_in_vocab] for doc in dataset['text']]
		
		print('\tRemoved {} stopwords from dictionary. It represents {}% of total words in starting vocabulary\n'.format(len(sw_in_vocab),freq1))

				
		print('Removing unique numbers (not words that contain numbers) ..........')
		# Remove numbers, but not words that contain numbers.

		text_gathered=' '.join([' '.join(text).strip() for text in dataset['text']])
		
		iterator=[[token for token in doc if token.isnumeric()] for doc in dataset['text']]
		num_removed=[]
		for unique in iterator:
				num_removed+=unique
		num_removed=np.unique(num_removed)
		freq2=get_frequency_and_vocab(before_texts,num_removed)					
		dataset['text'] = [[token for token in doc if not token.isnumeric()] for doc in dataset['text']]
		print('\tRemoved {} numeric words from dictionary. It represents {}% of total words in starting vocabulary\n'.format(len(num_removed),freq2))

				
		print('Removing words that contains only one character ..........')
			
		# Remove words that are only one character.
		regex_finder=re.findall(r'\b\w{1}\b',text_gathered)		
		one_char_removed=np.unique(regex_finder).tolist()

		freq3=get_frequency_and_vocab(before_texts,one_char_removed)					
		dataset['text'] = [[token for token in doc if len(token) > 1] for doc in dataset['text']]		
		
		print('\tRemoved {} one length characters from dictionary. It represents {}% of total words in starting vocabulary\n'.format(len(one_char_removed),freq3))
							
		print('-'*100) 			
		print('\nPreprocessed {} total words from beginning dictionary. It represents {}% of total words in starting vocabulary\n'.format(len(sw_in_vocab)+len(num_removed)+len(one_char_removed),freq1+freq2+freq3))
		print('-'*100)
		
		
	else : 
		print('\nWord analysis disabled')
		
		print('Removing stopwords ......')
		# Remove stopwords from text 		
		dataset['text'] = [[token for token in doc if token not in stop_words] for doc in dataset['text']]		
		print('Removing unique numbers (not words that contain numbers) ..........')
		# Remove numbers, but not words that contain numbers.		
		dataset['text'] = [[token for token in doc if not token.isnumeric()] for doc in dataset['text']]
		print('Removing words that contains only one character ..........')
		# Remove words that are only one character.				
		dataset['text'] = [[token for token in doc if len(token) > 1] for doc in dataset['text']]	
		
	print('\nNow starting to lemmatize')
	
	##Lemmatizing 
	
	lemmatizer = WordNetLemmatizer()
	new_texts = []
	count=0
	for text in dataset['text'] : 
		count+=1 
		if count % 1000 == 0 : 
			print('\tDone lemmatizing {}/{} texts'.format(count,len(dataset['text'])))
		new_texts.append([lemmatizer.lemmatize(token) for token in text])
	dataset['text']=new_texts	
	del new_texts,count
	
	print('Done lemmatizing & removing noise .........\n')
	
	#dataset['text'] = [[lemmatizer.lemmatize(token) for token in doc] for doc in dataset['text']]
	
	if enable_bigram:
	
		if basic_word_analysis : 	
			before_dict=Dictionary(dataset['text'])
			before_vocab=len(before_dict)
			del before_dict
			
		bigram_tokens=[]
		count=0
		print('\nBigrams enabled, starting to create bigram model\n')
		# Add bigrams and trigrams to docs (only ones that appear ... times or more).
		bigram = Phrases(dataset['text'], min_count=min_bigram_count,delimiter=b' ')
		bigram_phraser=Phraser(bigram)
		print('\nModel created, starting to process texts')
		for text in dataset['text']:
			count+=1
			if count % 1000 == 0 :
				print('\tDone processing {}/{} texts'.format(count,len(dataset['text'])))
			bigram_tokens.append(bigram_phraser[text])
		

		
		dataset['text']=bigram_tokens
		del bigram_tokens
		
		if basic_word_analysis : 
			print('Now creating Gensim dictionary\n')
			dictionary = Dictionary(dataset['text'])
			bigram_vocab=len(dictionary)
			print('\nFound {} bigrams in text\n'.format(bigram_vocab-before_vocab))
		



	elif enable_bigram==False : 
		print('Bigrams disabled')
		if basic_word_analysis : 
			print('\nNow creating Gensim dictionary\n')
			dictionary = Dictionary(dataset['text'])
	
	if not basic_word_analysis : 
		print('\nNow creating Gensim dictionary')
		dictionary = Dictionary(dataset['text'])
		
		
	bef = len(dictionary)
	print('Done creating dictionary\n')
	# Filter out words that occur less than ... documents, or more than ...% of the documents.
	dictionary.filter_extremes(no_below=extreme_no_below, no_above=extreme_no_above)
	print('\n')
	print('-'*100)
	
	if extreme_no_below>1 : 
		extreme_no_below_str=str(extreme_no_below)+' '+'documents'
	else : 
		extreme_no_below_str = str(extreme_no_below*100)+'%'+' '+'documents'
		
	if extreme_no_above>1 : 
		extreme_no_above_str=str(extreme_no_above)+' '+'documents'
	else : 
		extreme_no_above_str = str(extreme_no_above*100)+'%'+' '+'documents'

	print('\nKeeping words in no less than {} & in no more than {}:'.format(extreme_no_below_str,extreme_no_above_str))

	print('Number of unique tokens reduced from {} to {}, representing {} % of total vocabulary.'.format(bef, len(dictionary),np.round(((bef-len(dictionary))/bef)*100,3)))
	
	print('\nNow creating corpus of texts')
	corpus = [dictionary.doc2bow(doc) for doc in dataset['text']]
	
	print('Number of unique tokens: %d' % len(dictionary))
	print('Number of documents: %d \n' % len(corpus))
	print('-'*100)
		
	print('\nDone in {} minutes'.format(int((time()-t0)/60)))

	unique_time=np.unique(dataset['timestamp'])
	mapper_time=dict(zip(unique_time,range(len(unique_time))))
	dataset['timeslice']=dataset['timestamp'].apply(lambda x: mapper_time[x])

	return dict(
		dataset_preprocessed=dataset,
		corpus=corpus,
		dictionary=dictionary,
	)