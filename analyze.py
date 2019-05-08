import ast, sys
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def load_data(input_file):
	label_lists = []
	categories = []
	with open(input_file, 'r') as f:
		for line in f:
			data = line.strip().split('\t')
			label_lists.append(ast.literal_eval(data[2]))
			categories.append(data[1])
	return label_lists, categories

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

def save_label_frequencies(X, classes, all_labels, categories, output_file):
	categories = np.array(categories)
	freqs = []
	for category in classes:
		freq = np.sum(X[np.where(categories == category)], axis=0)
		freqs.append(freq.ravel().tolist())
	freqs = np.array(freqs).T.tolist()
	with open(output_file, 'w') as f:
		f.write('label\t%s\ttotal\n' % '\t'.join(classes))
		for i, freq in enumerate(freqs):
			data = [str(x) for x in freq]
			f.write('%s\t%s\n' % (all_labels[i], '\t'.join(data)))

def dev(X, y, fold_count=5):
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

def classify(X, y):
	clf = LogisticRegression()
	clf.fit(X, y)
	return clf.coef_

def most_important_labels(weights, n=10):
	importants = []
	for weight in weights:
		indexes = np.argpartition(weight, -n)[-n:]
		important = indexes[np.argsort(weight[indexes])].ravel()
		importants.append(np.flip(important).tolist())
	return importants

def run(input_file, output_file):
	label_lists, categories = load_data(input_file)
	X, all_labels = vectorize(label_lists)
	encoder = LabelEncoder()
	y = encoder.fit_transform(categories)
	classes = encoder.classes_
	
	save_label_frequencies(X, classes, all_labels, categories, output_file)

	print('Accuracy: %.4f' % dev(X, y))
	weights = classify(X, y)
	
	indexes = most_important_labels(weights)
	labels = np.array(all_labels)
	for i, category in enumerate(classes):
		names = labels[indexes[i]]
		weight = weights[i][indexes[i]]
		print('\n%s top features: ' % category.title())
		for j, name in enumerate(names):
			print('%s: %.4f' % (name, weight[j]))

if __name__ == '__main__':
	if len(sys.argv) != 3:
		print('correct usage: analyze.py <input_file> <output_file>')
	else:
		run(sys.argv[1], sys.argv[2])