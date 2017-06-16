"""
create the dataset pickled
"""
dataset_path = "data/"

import numpy as np
import cPickle as pkl
import gensim, logging, os

import os
import glob

from subprocess import Popen, PIPE


class Sentences(object):
	def __init__(self, dirname):
		self.dirname = dirname

	def __iter__(self):
		for fname in os.listdir(self.dirname):
			for line in open(os.path.join(self.dirname, fname)):
				yield line.split()

def get_word2vec(data_dir, model_name):
	en_data_raw = Sentences(data_dir)
	model = gensim.models.Word2Vec(en_data_raw, size=620, workers=4, min_count=5)
	model.save(model_name)

def word2vec_models():
	get_word2vec("data/en/", "model/en_model")
	get_word2vec("data/zh-cn", "model/zh-cn_model")

def grab_data(path, word2vec_path):
	data_raw = Sentences(path)
	model = gensim.models.Word2Vec.load(word2vec_path)
	seqs = []
	embed_dim = 620
	for line in data_raw:
		vecs = []
		for word in line:
			try:
				vecs.append(model[word])
			except KeyError:
				print word, "not in vocab"
				vecs.append([0 for i in xrange(embed_dim)])
		seqs.append(vecs)
	return seqs
	
def main_word2vec():
	data_en = grab_data("data/en/", "model/en_model")
	data_zhcn = grab_data("data/zh-cn", "model/zh-cn_model")
	train_en, eval_en, test_en = data_en[:-2000], data_en[-2000:-1000], data_en[-1000:]
	train_zhcn, eval_zhcn, test_zhcn = data_zh-cn[:-2000], data_zh-cn[-2000:-1000], data_zh-cn[-1000:]
	
	print len(train_en[0]), len(train_en), len(eval_en), len(test_en)
	f = open("data/test.pkl", "wb")
	pkl.dump((train_en, eval_en, test_en), f, -1)
	pkl.dump((train_zhcn, eval_zhcn, test_zhcn), f, -1)
	f.close()

def tokenize(sentences, lan):
	tokenizer_cmd = ['./tokenizer.perl', '-l', lan, '-q', '-']
	print 'Tokenizing...'
	text = "\n".join(sentences)
	tokenizer = Popen(tokenizer_cmd, stdin=PIPE, stdout=PIPE)
	tok_text, _ = tokenizer.communicate(text)
	toks = tok_text.split('\n')[:-1]
	print 'Done'

	return toks

def build_dict(sentences, lan):
	print "Building dictionary..."
	wordcount = dict()
	for ss in sentences:
		words = ss.strip().lower().split()
		for w in words:
			if w not in wordcount:
				wordcount[w] = 1
			else:
				wordcount[w] += 1
	counts = wordcount.values()
	keys = wordcount.keys()
	sorted_idx = np.argsort(counts)[::-1]

	worddict = dict()
	for idx, ss in enumerate(sorted_idx):
		worddict[keys[ss]] = idx+2 # leave 0 and 1 for UNK

	print np.sum(counts), 'total words', len(keys), 'unique words'
	return worddict	

def grab_data(path, lan, data_name, dict_name):
	sentences = []
	currdir = os.getcwd()
	os.chdir(path)
	for ff in glob.glob("*.txt"):
		with open(ff, 'r') as f:
			for line in f:
				sentences.append(line.strip())
	os.chdir(currdir)

	sentences = tokenize(sentences, lan)
	dictionary = build_dict(sentences, lan)
	
	seqs = [None] * len(sentences)
	for idx, ss in enumerate(sentences):
		words = ss.strip().lower().split()
		seqs[idx] = [dictionary[w] if w in dictionary else 1 for w in words]

	f = open(data_name, 'wb')
	pkl.dump((seqs[:-2000], seqs[-2000:-1000], seqs[-1000:]), f, -1)
	f.close()

	f = open(dict_name, 'wb')
	pkl.dump(dictionary, f, -1)
	f.close()
	return seqs, dictionary

def main():
	en_path = 'data/en/'
	zhch_path = 'data/zh-cn'
	pickle_path = 'data/'

	en_data, en_dict = grab_data(en_path, 'en', pickle_path + "en.pkl", pickle_path + "en_dict.pkl")
	zhcn_data, zhcn_dict = grab_data(zhch_path, 'zh-CN', pickle_path + 'zhcn.pkl', pickle_path + "zhcn_dict.pkl")



if __name__ == "__main__":
	main()
