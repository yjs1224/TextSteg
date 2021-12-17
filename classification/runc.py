import torch
from torch import nn
import torch.optim as optim
import argparse
import sys
from logger import Logger
import numpy as np

import data
import textcnn

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_args():
	parser = argparse.ArgumentParser()
	parser.register("type", "bool", lambda x : x.lower() == 'true')
	parser.add_argument("--neg_filename", type=str, default='../data/tweet-ac-5-onlyends/cover.txt')
	parser.add_argument("--pos_filename", type=str, default="../data/tweet-ac-5-onlyends/stego.txt")
	parser.add_argument("--epoch", type=int, default=200)
	parser.add_argument("--stop", type=int, default=50)
	parser.add_argument("--max_length", type=int, default=60)
	parser.add_argument("--logdir", type=str, default="./cnnlog")
	parser.add_argument("--sentence_num", type=int, default=100000)
	parser.add_argument("--rand_seed", type=int, default=0)
	args = parser.parse_args(sys.argv[1:])
	return args


args = get_args()
log_dir = args.logdir
os.makedirs(log_dir, exist_ok=True)
# logger
log_file = log_dir + "/cnn_{}.txt".format("_".join(args.neg_filename.split("/")[1:])+"___"+"_".join(args.pos_filename.split("/")[1:]))
logger = Logger(log_file)
import random
random.seed(args.rand_seed)
# print(random.random())

def main(data_helper):
	# ======================
	# 超参数
	# ======================
	BATCH_SIZE = 128
	EMBED_SIZE = 128
	FILTER_NUM = 128
	FILTER_SIZE = [3, 4, 5]
	CLASS_NUM = 2
	DROPOUT_RATE = 0.2
	EPOCH = args.epoch
	LEARNING_RATE = 0.01
	SAVE_EVERY = 20
	STOP = args.stop
	SENTENCE_NUM = args.sentence_num
	checkpoint_path = json_file = "".join(log_file.split(".txt")) + "_maxacc.pth"

	all_var = locals()
	print()
	for var in all_var:
		if var != "var_name":
			logger.info("{0:15} ".format(var))
			logger.info(all_var[var])
	print()

	# ======================
	# 数据
	# ======================
	

	# ======================
	# 构建模型
	# ======================
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = textcnn.TextCNN(
		vocab_size=data_helper.vocab_size,
		embed_size=EMBED_SIZE,
		filter_num=FILTER_NUM,
		filter_size=FILTER_SIZE,
		class_num=CLASS_NUM,
		dropout_rate=DROPOUT_RATE
	)
	model.to(device)
# 	summary(model, (20,))
	criteration = nn.CrossEntropyLoss()
# 	criteration = nn.BCELoss()
	optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
	early_stop = 0
	best_acc = 0
	best_test_loss = 1000
	# ======================
	# 训练与测试
	# ======================
	test_generate = data_helper.test_generator(BATCH_SIZE)
	for epoch in range(EPOCH):
		generator_train = data_helper.train_generator(BATCH_SIZE)
		generator_val = data_helper.val_generator(BATCH_SIZE)

		train_loss = []
		train_acc = []
		while True:
			try:
				text, label = generator_train.__next__()
			except:
				break
			optimizer.zero_grad()
			y = model(torch.from_numpy(text).long().to(device))
			loss = criteration(y, torch.from_numpy(label).long().to(device))
			loss.backward()
			optimizer.step()
			train_loss.append(loss.item())
			y = y.cpu().detach().numpy()
			train_acc += [1 if np.argmax(y[i]) == label[i] else 0 for i in range(len(y))]

		test_loss = 0
		test_acc = []
		test_tp = []
		tfn = []
		tpfn = []
		length_sum = 0

		while True:
			with torch.no_grad():
				try:
					text, label = generator_val.__next__()
				except:
					break
				y = model(torch.from_numpy(text).long().to(device))
				loss = criteration(y, torch.from_numpy(label).long().to(device))
				loss = loss.cpu().numpy()
				test_loss += loss * len(text)
				length_sum += len(text)
				y = y.cpu().numpy()
				label_pred = np.argmax(y, axis=-1)
				test_acc += [1 if np.argmax(y[i]) == label[i] else 0 for i in range(len(y))]
				test_tp += [1 if np.argmax(y[i]) == label[i] and label[i] == 1 else 0 for i in range(len(y))]
				tfn += [1 if np.argmax(y[i]) == 1 else 0 for i in range(len(y))]
				tpfn += [1 if label[i] == 1 else 0 for i in range(len(y))]
				# print(np.sum(test_tp), np.sum(tfn))

		# logger.info("epoch {:d}   training loss {:.4f}    test loss {:.4f}    train acc {:.4f}    test acc {:.4f}"
		#       .format(epoch + 1, np.mean(train_loss), np.mean(test_loss), np.mean(train_acc), np.mean(test_acc)))
		#
		test_loss = test_loss/length_sum
		acc = np.mean(test_acc)
		tpsum = np.sum(test_tp)
		test_precision = tpsum / (np.sum(tfn) + 1e-5)
		test_recall = tpsum / np.sum(tpfn)
		test_Fscore = 2 * test_precision * test_recall / (test_recall + test_precision + 1e-10)
		logger.info(
			"epoch {:d}, training loss {:.4f}, train acc {:.4f}, test loss {:.4f}, test acc {:.4f}, pre {:.4f}, recall {:.4f}, F1 {:.4f}"
			.format(epoch + 1, np.mean(train_loss), np.mean(train_acc), test_loss, acc, test_precision, test_recall, test_Fscore))

		# if np.mean(test_acc) > best_acc:
		# 	best_acc = np.mean(test_acc)
		# 	precison = test_precision
		# 	recall = test_recall
		# 	F1 = test_Fscore
		# if test_loss < best_test_loss:
		if np.mean(test_acc) > best_acc:
			state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, "val loss":test_loss, "val acc":np.mean(test_acc)}
			torch.save(state, checkpoint_path)
			best_test_loss = test_loss
			best_acc = np.mean(test_acc)
			precison = test_precision
			recall = test_recall
			F1 = test_Fscore
			early_stop = 0
		else:
			# state = torch.load(checkpoint_path)
			# optimizer.load_state_dict(state["optimizer"])
			# recover_epoch = state["epoch"]
			# model.load_state_dict(state["model"])
			# recover_loss = state["loss"]
			# recover_acc = state["acc"]
			# print("restart from epoch : %s      loss: %s, acc : %s"%(str(recover_epoch),str(recover_loss), str(recover_acc)))
			early_stop += 1
		if early_stop >= STOP:
			# logger.info('best acc: {:.4f}'.format(best_acc))
			# return best_acc
			test_loss = 0
			test_acc = []
			test_tp = []
			tfn = []
			tpfn = []
			length_sum = 0

			state = torch.load(checkpoint_path)
			optimizer.load_state_dict(state["optimizer"])
			model.load_state_dict(state["model"])
			while True:
				with torch.no_grad():
					try:
						text, label = test_generate.__next__()
					except:
						break
					y = model(torch.from_numpy(text).long().to(device))
					loss = criteration(y, torch.from_numpy(label).long().to(device))
					loss = loss.cpu().numpy()
					test_loss += loss * len(text)
					length_sum += len(text)
					y = y.cpu().numpy()
					label_pred = np.argmax(y, axis=-1)
					test_acc += [1 if np.argmax(y[i]) == label[i] else 0 for i in range(len(y))]
					test_tp += [1 if np.argmax(y[i]) == label[i] and label[i] == 1 else 0 for i in range(len(y))]
					tfn += [1 if np.argmax(y[i]) == 1 else 0 for i in range(len(y))]
					tpfn += [1 if label[i] == 1 else 0 for i in range(len(y))]
			test_loss = test_loss / length_sum
			acc = np.mean(test_acc)
			tpsum = np.sum(test_tp)
			test_precision = tpsum / (np.sum(tfn) + 1e-5)
			test_recall = tpsum / np.sum(tpfn)
			test_Fscore = 2 * test_precision * test_recall / (test_recall + test_precision + 1e-10)
			logger.info('val: loss: {:.4f}, acc: {:.4f}, pre {:.4f}, recall {:.4f}, F1 {:.4f}'.format(best_test_loss, best_acc, precison, recall, F1))
			logger.info("test: loss {:.4f}, acc {:.4f}, pre {:.4f}, recall {:.4f}, F1 {:.4f}".format(test_loss, acc, test_precision, test_recall, test_Fscore))
			# return (best_acc, precison, recall, F1)
			return acc, test_precision, test_recall, test_Fscore

	test_loss = 0
	test_acc = []
	test_tp = []
	tfn = []
	tpfn = []
	length_sum = 0

	state = torch.load(checkpoint_path)
	optimizer.load_state_dict(state["optimizer"])
	model.load_state_dict(state["model"])
	while True:
		with torch.no_grad():
			try:
				text, label = test_generate.__next__()
			except:
				break
			y = model(torch.from_numpy(text).long().to(device))
			loss = criteration(y, torch.from_numpy(label).long().to(device))
			loss = loss.cpu().numpy()
			test_loss += loss * len(text)
			length_sum += len(text)
			y = y.cpu().numpy()
			label_pred = np.argmax(y, axis=-1)
			test_acc += [1 if np.argmax(y[i]) == label[i] else 0 for i in range(len(y))]
			test_tp += [1 if np.argmax(y[i]) == label[i] and label[i] == 1 else 0 for i in range(len(y))]
			tfn += [1 if np.argmax(y[i]) == 1 else 0 for i in range(len(y))]
			tpfn += [1 if label[i] == 1 else 0 for i in range(len(y))]
	test_loss = test_loss / length_sum
	acc = np.mean(test_acc)
	tpsum = np.sum(test_tp)
	test_precision = tpsum / (np.sum(tfn) + 1e-5)
	test_recall = tpsum / np.sum(tpfn)
	test_Fscore = 2 * test_precision * test_recall / (test_recall + test_precision + 1e-10)
	logger.info('val: loss: {:.4f}, acc: {:.4f}, pre {:.4f}, recall {:.4f}, F1 {:.4f}'.format(best_test_loss, best_acc,
																							  precison, recall, F1))
	logger.info(
		"test: loss {:.4f}, acc {:.4f}, pre {:.4f}, recall {:.4f}, F1 {:.4f}".format(test_loss, acc, test_precision,
																					 test_recall, test_Fscore))
	# return (best_acc, precison, recall, F1)
	return acc, test_precision, test_recall, test_Fscore
		


if __name__ == '__main__':
	acc = []
	preci = []
	recall = []
	F1 = []
	with open(args.neg_filename, 'r', encoding='utf-8') as f:
		raw_pos = f.read().lower().split("\n")
	raw_pos = list(filter(lambda x: x not in ['', None], raw_pos))
	new_raw_pos = []
	if args.max_length is not None:
		for text in raw_pos:
			new_raw_pos.append(
				text if len(text.split()) < args.max_length else " ".join(text.split()[: args.max_length]))
	# raw_pos = [text for text in raw_pos if len(text.split()) < args.max_length]
	import random

	random.shuffle(new_raw_pos)
	# raw_pos = [' '.join(list(jieba.cut(pp))) for pp in raw_pos]
	# with open(args.pos_filename+"_filter", 'r', encoding='utf-8') as f:
	with open(args.pos_filename, 'r', encoding='utf-8') as f:
		raw_neg = f.read().lower().split("\n")
	raw_neg = list(filter(lambda x: x not in ['', None], raw_neg))
	new_raw_neg = []
	if args.max_length is not None:
		for text in raw_neg:
			new_raw_neg.append(
				text if len(text.split()) < args.max_length - 1 else " ".join(text.split()[: args.max_length - 1]))
	# raw_neg = [text for text in raw_neg if len(text.split()) < args.max_length ]
	random.shuffle(new_raw_neg)
	# raw_neg = [' '.join(list(jieba.cut(pp))) for pp in raw_neg]
	length = min(args.sentence_num, len(new_raw_neg), len(new_raw_pos))
	data_helper = data.DataHelper([new_raw_pos, new_raw_neg], use_label=True, word_drop=0)
	for i in range(10):
		random.seed(i)
		index = main(data_helper)
		acc.append(index[0])
		preci.append(index[1])
		recall.append(index[2])
		F1.append(index[3])
	acc_mean = np.mean(acc)
	acc_std = np.std(acc)
	pre_mean = np.mean(preci)
	pre_std = np.std(preci)
	recall_mean = np.mean(recall)
	recall_std = np.std(recall)
	f1_mean = np.mean(F1)
	f1_std = np.std(F1)
	logger.info("=============================================================================================")
	logger.info("best acc : {:.4f}".format(min(acc)))
	logger.info("worst acc: {:.4f}".format(max(acc)))
	logger.info("Final: acc {:.2f}+{:.2f}, precision {:.2f}+{:.2f}, recall {:.2f}+{:.2f}, F1 {:.2f}+{:.2f}"
		.format(acc_mean *100, acc_std * 100, pre_mean * 100, pre_std * 100, recall_mean * 100, recall_std * 100, f1_mean * 100, f1_std * 100))