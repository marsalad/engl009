import ast
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def vectorize(label_lists):
	all_labels = set()
	for label_list in label_lists:
		for label in label_list:
			all_labels.add(label)
	X = []
	all_labels = list(all_labels)
	for label_list in label_lists:
		X.append([label in label_list for label in all_labels])
	return np.array(X), all_labels

def load_data(source):
	label_lists = []
	medias = []
	with open('%s.txt' % source, 'r') as f:
		for line in f:
			data = line.strip().split('\t')
			label_lists.append(ast.literal_eval(data[2]))
			medias.append(data[1])
	return label_lists, medias

def dev(source, fold_count=5):
	label_lists, medias = load_data(source)
	X = vectorize(label_lists)[0]
	y = LabelEncoder().fit_transform(medias)
	y_trues = []
	y_preds = []
	folds = KFold(n_splits=fold_count)
	for i, (train_ind, test_ind) in enumerate(folds.split(X)):
		print('Running Fold %d/%d' % (i + 1, fold_count))
		X_train, y_train = X[train_ind], y[train_ind]
		X_test, y_test = X[test_ind], y[test_ind]
		clf = LogisticRegression()
		clf.fit(X_train, y_train)
		y_pred = clf.predict(X_test)
		y_trues.extend(y_test.ravel().tolist())
		y_preds.extend(y_pred.ravel().tolist())
	return accuracy_score(y_trues, y_preds)

def classify(source):
	label_lists, medias = load_data(source)
	X, all_labels = vectorize(label_lists)
	encoder = LabelEncoder()
	y = encoder.fit_transform(medias)
	clf = LogisticRegression()
	clf.fit(X, y)
	return np.array(all_labels), encoder.classes_, clf.coef_

def most_important_labels(weights, n=10):
	importants = []
	for weight in weights:
		indexes = np.argpartition(weight, -n)[-n:]
		important = indexes[np.argsort(weight[indexes])].ravel()
		importants.append(np.flip(important).tolist())
	return importants

def save_label_frequencies(source):
	label_lists, medias = load_data(source)
	medias = np.array(medias)
	X, all_labels = vectorize(label_lists)
	encoder = LabelEncoder()
	y = encoder.fit_transform(medias)
	freqs = []
	for media in encoder.classes_:
		freq = np.sum(X[np.where(medias == media)], axis=0).ravel().tolist()
		freqs.append(freq)
	freqs = np.array(freqs).T.tolist()
	with open('%s_label_frequencies.tsv' % source, 'w') as f:
		f.write('label\t%s\ttotal\n' % '\t'.join(encoder.classes_))
		for i, freq in enumerate(freqs):
			data = [str(x) for x in freq]
			total = sum(freq)
			f.write('%s\t%s\t%d\n' % (all_labels[i], '\t'.join(data), total))

def run(source):
	save_label_frequencies(source)
	print('Accuracy: %.4f' % dev(source))
	labels, medias, weights = classify(source)
	indexes = most_important_labels(weights)
	for i, media in enumerate(medias):
		names = labels[indexes[i]]
		weight = weights[i][indexes[i]]
		print('\n%s top features: ' % media.title())
		for j, name in enumerate(names):
			print('%s: %.4f' % (name, weight[j]))

if __name__ == '__main__':
	run('dataset')
	print()
	run('musemart')