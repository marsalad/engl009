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
	for label_list in label_lists:
		X.append([label in label_list for label in all_labels])
	return X

def load_data(source):
	label_lists = []
	medias = []
	with open('%s.txt' % source, 'r') as f:
		for line in f:
			data = line.strip().split('\t')
			label_lists.append(list(data[2]))
			medias.append(data[1])
	X = np.array(vectorize(label_lists))
	y = LabelEncoder().fit_transform(medias)
	return X, y

def classify(source, fold_count=5):
	X, y = load_data(source)
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
	
if __name__ == '__main__':
	accuracy = classify('musemart')
	print('Accuracy: %.4f' % accuracy)
	