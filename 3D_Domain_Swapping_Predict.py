import pandas as pd
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.svm import SVC,LinearSVC
from sklearn.preprocessing import RobustScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cross_validation import KFold,train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from sklearn.neural_network import MLPClassifier

def rank_set(list):
	rnk_lst = np.zeros(len(list))
	print list
	for (i,j) in zip(list,range(1,len(list)+1)):
		rnk_lst[int(i)-1] = j
	return rnk_lst

def borda_count(ranked_list,items):
	count_dict = {}
	m,n = ranked_list.shape
	for it in items:
		count_dict[it] = {}
		for i in xrange(m):
			ind = np.where(ranked_list[i,:]==it)[0][0] + 1
			if ind not in count_dict[it].keys():
				count_dict[it][ind] = 1
			else:
				count_dict[it][ind] += 1
	BordaCount = []
	for it in items:
		bcount = sum([(len(items) - x + 1)*count_dict[it][x] for x in count_dict[it].keys()])
		BordaCount.append([it,bcount])
	BordaCount.sort(key=lambda x: x[1],reverse=True)
	bindex = [bc[0]-1 for bc in BordaCount]
	return bindex


train_pos = pd.read_csv("TRAIN_POS.csv",header=None)
test_pos = pd.read_csv("TEST_POS.csv",header=None)
train_neg = pd.read_csv("TRAIN_NEG.csv",header=None)
test_neg = pd.read_csv("TEST_NEG.csv",header=None)

features = pd.read_table("FEATURES.txt",header=None)
num_of_features = features.shape[0]
ranked_lists = ['RANKED_FEATURE_COND','RANKED_FEATURE_IMP','RANKED_FEATURE_AUC']

rank_array = np.zeros((len(ranked_lists),num_of_features))

for i in xrange(len(ranked_lists)):
	index = []
	feature_rank = pd.read_csv(ranked_lists[i]+".csv")
	for j in list(feature_rank.iloc[:,0]):
		index.append(int(j[1:]))
	rank_array[i] = index

borda_index = borda_count(rank_array,range(1,(num_of_features+1)))

print features.iloc[borda_index,:]


plt.title("Combined Statistics", fontweight = 'bold', fontsize = 18)
plt.ylabel("Ranking", fontweight='bold')
plt.xticks(range(1,(num_of_features)+1), features.iloc[:,0], size = 'small', rotation = 'vertical')
plt.plot(range(1,(num_of_features)+1), rank_set(rank_array[0]), label = "COND", color = "magenta")
plt.plot(range(1,(num_of_features)+1), rank_set(rank_array[1]), label = "IMP", color = "blue")
plt.plot(range(1,(num_of_features)+1), rank_set(rank_array[2]), label = "AUC", color = "green")
plt.plot(range(1,(num_of_features)+1), rank_set([i+1 for i in borda_index]), label = "Borda", linewidth=4, color = "red")
plt.legend(loc = 'upper left')
plt.show()

train_data = pd.DataFrame(np.vstack((train_pos.iloc[:,0:num_of_features],train_neg.iloc[:,0:num_of_features])))
train_class = np.array(list(train_pos.iloc[:,num_of_features]) + list(train_neg.iloc[:,num_of_features]))


test_data = pd.DataFrame(np.vstack((test_pos.iloc[:,0:num_of_features],test_neg.iloc[:,0:num_of_features])))
test_class = np.array(list(test_pos.iloc[:,num_of_features]) + list(test_neg.iloc[:,num_of_features]))


train_data_normalized = deepcopy(train_data)
test_data_normalized = deepcopy(test_data)

#for i in xrange(num_of_features):
#	train_data_normalized.iloc[:,i] = (train_data.iloc[:,i] - np.mean(train_data.iloc[:,i])) / np.std(train_data.iloc[:,i])
#	test_data_normalized.iloc[:,i] = (test_data.iloc[:,i] - np.mean(test_data.iloc[:,i])) / np.std(test_data.iloc[:,i])

#for i in xrange(num_of_features):
#	train_data_normalized.iloc[:,i] = (train_data.iloc[:,i] - np.min(train_data.iloc[:,i])) / (np.max(train_data.iloc[:,i]) - np.min(train_data.iloc[:,i]))
#	test_data_normalized.iloc[:,i] = (test_data.iloc[:,i] - np.min(test_data.iloc[:,i])) / (np.max(test_data.iloc[:,i]) - np.min(test_data.iloc[:,i]))

rscale = RobustScaler()
train_data_normalized = pd.DataFrame(rscale.fit_transform(train_data))
test_data_normalized = pd.DataFrame(rscale.fit_transform(test_data))

accuracy_train_test = []
f1_score_train_test = []

plt.scatter(train_data_normalized.iloc[:,borda_index[0]],train_data_normalized.iloc[:,borda_index[1]],color=['r' for i in xrange(150)]+['y' for i in xrange(150)])
plt.xlabel(features.iloc[borda_index[0],0])
plt.ylabel(features.iloc[borda_index[1],0])
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(train_data_normalized.iloc[:,borda_index[0]],train_data_normalized.iloc[:,borda_index[1]],\
	train_data_normalized.iloc[:,borda_index[2]],color=['r' for i in xrange(150)]+['y' for i in xrange(150)],marker='o')
ax.set_xlabel(features.iloc[borda_index[0],0])
ax.set_ylabel(features.iloc[borda_index[1],0])
ax.set_zlabel(features.iloc[borda_index[2],0])
plt.show()


for ft in [10,25,50,num_of_features+1]:
	print "Features ",ft,": ",
	clf = SVC(C=100,gamma=0.001)
	#clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
	kfold = KFold(n=train_data_normalized.shape[0],n_folds=5)
	kfold_average = []
	for train,test in kfold:
		X_train, X_test, Y_train, Y_test = train_data_normalized.iloc[train,:ft], train_data_normalized.iloc[test,:ft], train_class[train], train_class[test]
		clf.fit(X_train,Y_train)
		test_predict = clf.predict(X_test)
		kfold_average.append(accuracy_score(Y_test,test_predict))
	print np.mean(kfold_average)


for it in xrange(1,(num_of_features+1)):
	#print "Iterations ",it,": ",
	#clf = ann.AdalineGD(eta=0.0001)
	clf = SVC(C=100,gamma=0.001)
	#clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
	clf.fit(train_data_normalized.iloc[:,borda_index[:it]],train_class)
	test_predict = clf.predict(test_data_normalized.iloc[:,borda_index[:it]])
	accuracy_train_test.append(accuracy_score(test_class,test_predict))
	f1_score_train_test.append(f1_score(test_class,test_predict))

plt.title("Classification Performance", fontweight = 'bold', fontsize = 18)
plt.ylabel("Performance", fontweight='bold')
plt.xticks(range(1,(num_of_features)+1), range(1,(num_of_features)+1), size = 'small', rotation = 'vertical')
plt.plot(range(1,(num_of_features)+1), accuracy_train_test, color = "green", label = "Accuracy", linewidth=3)
plt.plot(range(1,(num_of_features)+1), f1_score_train_test, color = "red", label = "F1-Score", linewidth=3)
plt.legend(loc = 'upper left')
plt.show()
