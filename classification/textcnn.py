import torch
from torch import nn


class TextCNN(nn.Module):
	def __init__(self, vocab_size, embed_size, filter_num, filter_size, class_num, dropout_rate):
		super(TextCNN, self).__init__()
		self.embedding = nn.Embedding(vocab_size, embed_size)
		self.cnn_list = nn.ModuleList()
		for size in filter_size:
			self.cnn_list.append(nn.Conv1d(embed_size, filter_num, size))
		self.relu = nn.ReLU()
		self.max_pool = nn.AdaptiveMaxPool1d(1)
		self.dropout = nn.Dropout(dropout_rate)
		self.output_layer = nn.Linear(filter_num * len(filter_size), class_num)
		self.softmax = nn.Softmax(dim=1)

	def forward(self, x):
		"""
		:param x:(N,L)
		:return: (N,class_num)
		"""
		x = x.long()
		_ = self.embedding(x)
		_ = _.permute(0, 2, 1)
		result = []
		for self.cnn in self.cnn_list:
			__ = self.cnn(_)
			__ = self.max_pool(__)
			__ = self.relu(__)
			result.append(__.squeeze(dim=2))

		_ = torch.cat(result, dim=1)
		_ = self.dropout(_)
		_ = self.output_layer(_)
		# _ = self.softmax(_)
		return _


if __name__ == "__main__":
	"""# m = nn.AdaptiveMaxPool1d(5)
	# input = torch.randn(1, 64, 8)
	# output = m(input)
	textCnn = TextCNN(10,30,5,[3,4,5],2)
	x = torch.randint(low=0, high=10, size=(64,20))
	y = textCnn(x)
	criteration = nn.CrossEntropyLoss()
	optimizer = optim.SGD(textCnn.parameters(), lr=0.001)"""
	pass