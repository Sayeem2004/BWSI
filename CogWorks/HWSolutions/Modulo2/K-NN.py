# Program for doing K-Nearest Neighbors on the cifar10 data set.
# The images we are using are split into 10 types listed below.
# airplane, class-id: 0
# automobile, class-id: 1
# bird, class-id: 2
# cat, class-id: 3
# deer, class-id: 4
# dog, class-id: 5
# frog, class-id: 6
# horse, class-id: 7
# ship, class-id: 8
# truck, class-id: 9



# Necessary Imports and obtaining the training and testing data.
import cifar10;
import numpy as np;
from bwsi_grader.cogworks.nearest_neighbors import grade_distances;
from bwsi_grader.cogworks.nearest_neighbors import grade_predict;
from bwsi_grader.cogworks.nearest_neighbors import grade_make_folds;
# "cifar10" must be a subfolder in the current directory for this part to work.
if not cifar10.get_path().is_file():
    cifar10.download();
else:
    print("cifar10 is already downloaded at:\n{}".format(cifar10.get_path()));
# Loading in the training data and converting them into floats.
x_train, y_train, x_test, y_test = (i.astype("float32") for i in cifar10.load());
x_train = x_train.transpose([0,2,3,1]);
x_test = x_test.transpose([0,2,3,1]);



# Limiting the data to make the program run faster.
x_train, y_train = x_train[:5000], y_train[:5000];
x_test, y_test = x_test[:500], y_test[:500];
print("\n");
print('Training data shape: ', x_train.shape);
print('Training labels shape: ', y_train.shape);
print('Test data shape: ', x_test.shape);
print('Test labels shape: ', y_test.shape);
# Flattening the data values
print("\n");
x_train = np.reshape(x_train, (x_train.shape[0], -1));
x_test = np.reshape(x_test, (x_test.shape[0], -1));
print("new train-shape:", x_train.shape);
print("new test-shape:", x_test.shape);



# Distance Function.
def compute_distances(x, y):
    M = np.shape(x)[0];
    N = np.shape(y)[0];
    ret = np.zeros((M,N));
    for i in range(M):
        for j in range(N):
             ret[i,j] = np.sqrt(np.sum(np.square(np.subtract(x[i],y[j]))));
    return ret;
grade_distances(compute_distances);
# Predict Function.
def predict(dists, labels, k=1):
    M = np.shape(dists)[0];
    ret = np.zeros(M);
    clabel = np.argsort(labels)
    for i in range(M):
        cdist = dists[i,clabel];
        slabel = np.argsort(cdist);
        values, counts = np.unique(labels[clabel[slabel[:k]]],return_counts=True);
        ret[i] = values[np.argmax(counts)];
    ret = ret.astype(np.int32);
    return ret;
grade_predict(predict);



# Making Folds to find best value of K.
def make_folds(x, num_folds):
    ret = [];
    M = np.shape(x)[0];
    W = M//num_folds;
    i = 0;
    for _ in range(num_folds):
        ret.append(x[i:i+W]);
        i += W;
    return ret;
grade_make_folds(make_folds)
def combine(x, fold_i, num_folds):
    ret = np.copy(x[0]);
    M = np.shape(ret)[0];
    for i in range(num_folds):
        if (i != fold_i):
            ret = np.concatenate((ret,x[i]));
    return ret[M:];
def compare(predictions, labels):
    total = np.size(labels);
    count = np.sum(np.equal(predictions,labels));
    return count/total;


# Code for actually testing out each K value.
num_folds = 5;
x_train_folds = make_folds(x_train, num_folds=num_folds);
y_train_folds = make_folds(y_train, num_folds=num_folds);
k_values = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100];
accuracies = {};
for k in k_values:
    accuracies[k] = [];
for fold_i in range(num_folds):
    validation_data = x_train_folds[fold_i];
    labeled_data = combine(x_train_folds, fold_i, num_folds);
    validation_labels = y_train_folds[fold_i];
    labels = combine(y_train_folds, fold_i, num_folds);
    computed_data = compute_distances(validation_data, labeled_data);
    for k in k_values:
        predictions = predict(computed_data, labels, k=k);
        acc = compare(predictions, validation_labels)
        acc = round(float(acc), 3);
        accuracies[k].append(acc);



# Printing out best and worst accuracies and their k-values.
print("Accuracies:");
for k in k_values:
    print(k,accuracies[k]);
all_acc = list()
for k in k_values:
    all_acc.extend(accuracies[k])
best = max(all_acc);
worst = min(all_acc)
best_k = k_values[all_acc.index(best)//5];
worst_k = k_values[all_acc.index(worst)//5];
print("\n");
print("Best Accuracy:",best,"with K value",best_k);
print("Worst Accuracy:",worst,"with K value",worst_k);



# Finally using test images.
dists = compute_distances(x_test, x_train);
guesses = predict(dists, y_train, best_k);
accuracy = compare(guesses, y_test);
accuracy = round(float(accuracy), 3);
print("\n");
print("Final Test Accuracy:",accuracy);
