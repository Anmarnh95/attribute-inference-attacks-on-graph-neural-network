from scipy.spatial import distance_matrix
import numpy as np
import random
import pandas as pd
import os
from torch_geometric.utils import remove_self_loops, subgraph, from_scipy_sparse_matrix
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
import itertools

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def encode_onehot(labels):
	classes = set(labels)
	classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
	                enumerate(classes)}
	labels_onehot = np.array(list(map(classes_dict.get, labels)),
	                         dtype=np.int32)
	return labels_onehot


def build_relationship(x, thresh=0.25):
	#    df_euclid = pd.DataFrame(1 / (1 + distance_matrix(x.T.T, x.T.T)), columns=x.T.columns, index=x.T.columns)
	#    df_euclid = df_euclid.to_numpy()
	#    np.save('credit_normal.npy', df_euclid)
	df_euclid = np.load('credit_normal.npy')
	idx_map = []
	for ind in range(df_euclid.shape[0]):
		max_sim = np.sort(df_euclid[ind, :])[-2]
		neig_id = np.where(df_euclid[ind, :] > thresh * max_sim)[0]
		import random
		random.seed(912)
		random.shuffle(neig_id)
		for neig in neig_id[:200]:
			if neig != ind:
				idx_map.append([ind, neig])
	print('building edge relationship complete')
	idx_map = np.array(idx_map)

	return idx_map


def load_credit(path, sens_attr="Age", predict_attr="NoDefaultNextMonth", label_number=1000, train_to_split = (0,0)):
	dataset = "credit"
	idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
	header = list(idx_features_labels.columns)
	header.remove(predict_attr)
	header.remove('Single')

	#    # Normalize MaxBillAmountOverLast6Months
	#    idx_features_labels['MaxBillAmountOverLast6Months'] = (idx_features_labels['MaxBillAmountOverLast6Months']-idx_features_labels['MaxBillAmountOverLast6Months'].mean())/idx_features_labels['MaxBillAmountOverLast6Months'].std()
	#
	#    # Normalize MaxPaymentAmountOverLast6Months
	#    idx_features_labels['MaxPaymentAmountOverLast6Months'] = (idx_features_labels['MaxPaymentAmountOverLast6Months'] - idx_features_labels['MaxPaymentAmountOverLast6Months'].mean())/idx_features_labels['MaxPaymentAmountOverLast6Months'].std()
	#
	#    # Normalize MostRecentBillAmount
	#    idx_features_labels['MostRecentBillAmount'] = (idx_features_labels['MostRecentBillAmount']-idx_features_labels['MostRecentBillAmount'].mean())/idx_features_labels['MostRecentBillAmount'].std()
	#
	#    # Normalize MostRecentPaymentAmount
	#    idx_features_labels['MostRecentPaymentAmount'] = (idx_features_labels['MostRecentPaymentAmount']-idx_features_labels['MostRecentPaymentAmount'].mean())/idx_features_labels['MostRecentPaymentAmount'].std()
	#
	#    # Normalize TotalMonthsOverdue
	#    idx_features_labels['TotalMonthsOverdue'] = (idx_features_labels['TotalMonthsOverdue']-idx_features_labels['TotalMonthsOverdue'].mean())/idx_features_labels['TotalMonthsOverdue'].std()

	# build relationship
	if os.path.exists(f'{path}/{dataset}_edges.txt'):
		edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
	else:
		edges_unordered = build_relationship(idx_features_labels[header], thresh=0.7)
		np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

	features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
	labels = idx_features_labels[predict_attr].values
	idx = np.arange(features.shape[0])
	idx_map = {j: i for i, j in enumerate(idx)}
	edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
	                 dtype=int).reshape(edges_unordered.shape)
	adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
	                    shape=(labels.shape[0], labels.shape[0]),
	                    dtype=np.float32)

	# build symmetric adjacency matrix
	adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

	# features = normalize(features)
	adj = adj + sp.eye(adj.shape[0])

	edge_index, edge_weight = from_scipy_sparse_matrix(adj)

	features = torch.FloatTensor(np.array(features.todense()))
	labels = torch.LongTensor(labels)
	# adj = sparse_mx_to_torch_sparse_tensor(adj)

	import random
	random.seed(20)
	label_idx_0 = np.where(labels == 0)[0]
	label_idx_1 = np.where(labels == 1)[0]
	random.shuffle(label_idx_0)
	random.shuffle(label_idx_1)

	if train_to_split == (0,0):
		idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
							label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
		idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
							label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
		idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])
	else: 
		idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), train_to_split[0] // 2)],
							label_idx_1[:min(int(0.5 * len(label_idx_1)), train_to_split[0] // 2)])
		idx_test = np.append(label_idx_0[int(0.5 * len(label_idx_0)):min(int(0.75 * len(label_idx_0)),(int(0.5 * len(label_idx_0)) + (train_to_split[1] // 2)))],
							label_idx_1[int(0.5 * len(label_idx_1)):min(int(0.75 * len(label_idx_1)),(int(0.5 * len(label_idx_1)) + (train_to_split[1] // 2)))])
		idx_val = np.append(label_idx_0[int(0.75 * len(label_idx_0)):(int(0.75 * len(label_idx_0))+(train_to_split[1] // 4))], 
							label_idx_1[int(0.75 * len(label_idx_1)):(int(0.75 * len(label_idx_0))+(train_to_split[1] // 4))])

	# for ind in idx_val:
	# 	if ind in idx_train:
	# 		print(ind)
	# 	if ind in idx_test:
	# 		print(ind)
	#
	# for ind in idx_test:
	# 	if ind in idx_train:
	# 		print(ind)

	#    idx_train = label_idx[:min(int(0.5 * len(label_idx)),label_number)]
	#    idx_val = label_idx[int(0.5 * len(label_idx)):int(0.75 * len(label_idx))]
	#    idx_test = label_idx[int(0.75 * len(label_idx)):]


	all_idx = list(itertools.chain(idx_train, idx_val, idx_test))
	all_idx.sort()

	# get index
	indices = [i for i in range(len(labels))]
	listidx_dataidx_dict = dict(zip(all_idx, indices))

	# print("listidx_dataidx_dict", listidx_dataidx_dict)


	train_mask = np.full_like(labels, False, dtype=bool)
	test_mask = np.full_like(labels, False, dtype=bool)
	val_mask = np.full_like(labels, False, dtype=bool)
	for i in all_idx:
		if i in idx_train:
			# print(listidx_dataidx_dict[i])
			train_mask[listidx_dataidx_dict[i]] = True
		elif i in idx_test:
			test_mask[listidx_dataidx_dict[i]] = True
		elif i in idx_val:
			val_mask[listidx_dataidx_dict[i]] = True


	train_mask = torch.tensor(train_mask)
	val_mask = torch.tensor(val_mask)
	test_mask = torch.tensor(test_mask)


	sens = idx_features_labels[sens_attr].values.astype(int)
	sens = torch.FloatTensor(sens)
	idx_train = torch.LongTensor(idx_train)
	idx_val = torch.LongTensor(idx_val)
	idx_test = torch.LongTensor(idx_test)
	# ipdb.set_trace()


	data = Data(x=features, edge_index=edge_index, y=labels, train_mask=train_mask, test_mask=test_mask, val_mask=val_mask)
	# print("data", data)
	# print(data.x)
	return data


def normalize(mx):
	"""Row-normalize sparse matrix"""
	rowsum = np.array(mx.sum(1))
	r_inv = np.power(rowsum, -1).flatten()
	r_inv[np.isinf(r_inv)] = 0.
	r_mat_inv = sp.diags(r_inv)
	mx = r_mat_inv.dot(mx)
	return mx


def feature_norm(features):
	min_values = features.min(axis=0)[0]
	max_values = features.max(axis=0)[0]
	return 2 * (features - min_values).div(max_values - min_values) - 1


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
	"""Convert a scipy sparse matrix to a torch sparse tensor."""
	sparse_mx = sparse_mx.tocoo().astype(np.float32)
	indices = torch.from_numpy(
		np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
	values = torch.from_numpy(sparse_mx.data)
	shape = torch.Size(sparse_mx.shape)
	return torch.sparse.FloatTensor(indices, values, shape)


# Running the code
#data = load_credit("./Dataset/Credit")
#print(data)