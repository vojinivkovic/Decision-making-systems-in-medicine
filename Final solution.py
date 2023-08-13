import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier as KNN
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import imblearn


pd.set_option('display.max_columns', None)
pd.set_option("expand_frame_repr", False)

def swap_columns(data, col1, col2):
    data[col1], data[col2] = data[col2], data[col1]
    data = data.rename(columns={col1 : col2, col2 : col1})

    return data

def replace_values_in_bmi(data):
    data.bmi.fillna(data.bmi.mean(), inplace=True)

def one_hot_encoding(data, feature):
    unique_values = feature.unique()
    name = feature.name
    for unique in unique_values:
        new_values = feature == unique
        data[name + "_" + str(unique).lower()] = new_values.astype("int32")     #Ovde izbacuje upozorenje
    data.drop(columns=name, inplace=True)

def classifier_for_smoking_status(data):
    training_data_for_smoking = data[~data.smoking_status.isna()]
    classifier_data = data[data.smoking_status.isna()]
    classifier_data = classifier_data.drop(columns=["stroke", "smoking_status"])
    classifier_data = classifier_data[quantitative]         #ovo sam dodao
    training_data_x = training_data_for_smoking.drop(columns=["stroke", "smoking_status"])
    training_data_x = training_data_x[quantitative]         #ovo sam dodao
    training_data_y = training_data_for_smoking.smoking_status

    encoder = LabelEncoder()
    training_data_y = encoder.fit_transform(training_data_y)
    classes = encoder.classes_

    knn = KNN(n_neighbors=3)
    knn.fit(training_data_x, training_data_y)
    predictions = knn.predict(classifier_data)

    N = len(data)
    count = 0
    for i in range(N):
        if (pd.isna(data.iloc[i].smoking_status)):
            data.loc[i, "smoking_status"] = classes[predictions[count]]
            count += 1

def calculate_gini_index(column):
    unique_values = np.unique(column)
    all_probalities = 0
    for unique in unique_values:
        p = sum(column == unique) / len(column)
        all_probalities += p ** 2
    gini_index = 1 - all_probalities
    return gini_index

def digitize_values(column):
    min_value = min(column)
    max_value = max(column)
    n_bins = 10

    step = (max_value - min_value) / n_bins
    bins = np.array([min_value + i * step for i in range(n_bins + 1)])

    indexes = np.digitize(column, bins)
    new_column = np.array([bins[indexes[i] - 1] for i in range(len(indexes))])
    return new_column

def calculate_entropy(column):
    entropy = 0
    unique_values = np.unique(column)
    for unique in unique_values:
        p = sum(column == unique) / len(column)
        entropy += -p * np.log2(p)
    return entropy

def entropy_of_feature(feature, data):
    unique_values = np.unique(feature)
    entropy_feature = 0
    for unique in unique_values:
        p = sum(feature == unique) / len(feature)
        entropy_of_unique = calculate_entropy(data.stroke[feature == unique])
        entropy_feature += p * entropy_of_unique
    return entropy_feature

def remove_examples_with_spec_value(data, feature, value):
    data = data.loc[feature != value]
    return data  # Mora da vrati data na kraju inace nece da radi?

def normalization(data):
    data_norm = data - np.mean(data, axis=0)
    data_norm /= np.std(data, axis=0)
    return data_norm

def eigenvalues(data):
    Sx = np.cov(data.T)
    eigval, eigvec = np.linalg.eig(Sx)
    idx = np.argsort(eigval)[::-1]
    eigval = eigval[idx]
    eigvec = eigvec[:, idx]
    return eigval, eigvec

def PCA(data_norm, eigvec, new_dimension, outcome):
    A = eigvec[:, :new_dimension]

    Y = A.T @ data_norm.T
    Y = Y.T

    principal_data = pd.concat([Y, outcome], axis=1)
    principal_data.columns = ['PC1', 'PC2', 'Outcome']

    plt.figure(1)
    sns.scatterplot(data=principal_data, x='PC1', y='PC2', hue='Outcome')
    plt.show()
    return principal_data

def dimensionality_reduction_PCA(data, new_dimension):
    data_norm = normalization(data.loc[:, data.columns!='stroke'])
    eigval, eigvec = eigenvalues(data_norm)
    principal_data = PCA(data_norm, eigvec, new_dimension, data.stroke)
    return principal_data

def dimensionality_reduction_LDA(data, new_dimension):
    data_norm = normalization(data.loc[:, data.columns!='stroke'])
    X1 = data_norm.loc[data.stroke == 1, :]
    p1 = X1.shape[0]/data.shape[0]
    M1 = X1.mean().values.reshape(X1.shape[1], 1)
    S1 = X1.cov()

    X2 = data_norm.loc[data.stroke == 0, :]
    p2 = X2.shape[0] / data.shape[0]
    M2 = X2.mean().values.reshape(X2.shape[1], 1)
    S2 = X2.cov()

    M = p1*M1 + p2*M2
    Sw = p1*S1 + p2*S2
    Sb = p1*(M1-M)@(M1-M).T + p2*(M2-M)@(M2-M).T
    Sm = Sb + Sw

    S1 = Sw
    S2 = Sb
    T = np.linalg.inv(S1) @ S2      #Probati sve kombinacije S1 i S2, samo ne sme S1 = Sb
    eigval, eigvec = np.linalg.eig(T)


    A = eigvec[:, :new_dimension]

    Y = A.T @ data_norm.T
    Y = Y.T

    principal_df = pd.concat([Y, data.stroke], axis=1)
    principal_df.columns = ['LDA1', 'Outcome']

    return principal_df

def calculate_information_gain(data):
    label_entropy = calculate_entropy(data.stroke)

    temp_data = data.loc[:, data.columns != 'stroke']
    information_gains = {}
    columns = temp_data.columns

    for column in columns:
        if (column in quantitative):
            information_gain = label_entropy - entropy_of_feature(digitize_values(temp_data[column]), data)
            information_gains[column] = information_gain
        else:
            information_gain = label_entropy - entropy_of_feature(temp_data[column], data)
            information_gains[column] = information_gain

    #information_gains = dict(sorted(information_gains.items(), key=lambda x: x[1], reverse=True))
    information_gains = sorted(information_gains.items(), key=lambda x: x[1], reverse=True)
    #Promenio sam da ne vraca dict nego listu
    return information_gains

def plot_3d(columns, label, num):
    if num > len(columns) or num <= 0:
        print('Neodgovarajuci broj primera!')
        return

    plt.figure()
    data_temp = columns.head(num)
    ax = plt.axes(projection='3d')
    x = data_temp[data_temp.columns[0]]
    y = data_temp[data_temp.columns[1]]
    z = data_temp[data_temp.columns[2]]
    c = label.head(num)
    ax.scatter(x, y, z, c=c)
    plt.show()

def bayes(data, dimensions, label_string):
    X1 = data[dimensions].loc[data[label_string] == 1]
    N1 = X1.shape[0]
    X2 = data[dimensions].loc[data[label_string] == 0]
    N2 = X2.shape[0]
    N1_train = int(0.6*N1)
    N2_train = int(0.6*N2)
    X1_train = X1.iloc[:N1_train, :]
    X2_train = X2.iloc[:N2_train, :]
    X1_test = X1.iloc[N1_train:, :]
    X2_test = X2.iloc[N2_train:, :]

    M1p = np.mean(X1_train, axis=0)
    S1p = np.cov(X1_train.T)

    M2p = np.mean(X2_train, axis=0)
    S2p = np.cov(X2_train.T)

    p1 = N1_train/(N1_train+N2_train)
    p2 = N2_train/(N1_train+N2_train)
    T = np.log(p1 / p2)

    decision1 = np.zeros((N1-N1_train, 1))


    for i in range(N1-N1_train):
        x1 = X1_test.iloc[i, :]
        f1 = calculate_pdf_gaussian(x1, M1p, S1p)
        f2 = calculate_pdf_gaussian(x1, M2p, S2p)
        h1 = -np.log(f1) + np.log(f2)
        if h1 < T:
            decision1[i] = 0
        else:
            decision1[i] = 1

    decision2 = np.zeros((N2 - N2_train, 1))

    for i in range(N2 - N2_train):
        x2 = X2_test.iloc[i, :]
        f1 = calculate_pdf_gaussian(x2, M1p, S1p)
        f2 = calculate_pdf_gaussian(x2, M2p, S2p)
        h2 = -np.log(f1) + np.log(f2)
        if h2 < T:
            decision2[i] = 0
        else:
            decision2[i] = 1

    decision = np.append(decision1, decision2, axis=0)
    Xtest = np.append(X1_test, X2_test, axis=0)
    Ytest = np.append(np.zeros((N1 - N1_train, 1)), np.ones((N2 - N2_train, 1)))

    return Ytest, decision

def conf_matrix(Ytest, decision):
    conf_matrix = confusion_matrix(Ytest, decision)
    plt.figure()
    sns.heatmap(conf_matrix, annot=True, fmt='g', cbar=False)
    plt.show()

def calculate_pdf_gaussian(x, m, s):
    det = np.linalg.det(s)
    inv = np.linalg.inv(s)
    x_mu = x - m
    cdf_const = 1/np.sqrt(2*np.pi*det)
    cdf_rest = np.exp(0.5*x_mu.T@inv@x_mu)
    return cdf_const * cdf_rest

def decision_tree(X, Y, N):
    X_training, X_test, Y_training, Y_test = train_test_split(X, Y, train_size=0.6, random_state=42, stratify=Y)

    M = np.mean(X_training, axis=0)
    S = np.std(X_training, axis=0)

    X_training = (X_training - M) / S
    X_test = (X_test - M) / S

    threshold = np.linspace(np.min(X), np.max(X), N)
    acc = np.zeros((len(threshold), 1))
    for p in range(len(threshold)):
        pred = (X >= threshold[p])
        acc[p] = f1_score(Y, pred)

    best_threshold = threshold[np.argmax(acc)]

    return (threshold, acc, best_threshold)

def calculateIG(X, Y):
    infoD = calculate_entropy(Y)
    IG = np.zeros((X.shape[1] - 1,))
    brojKoraka = 10
    for ob in range(X.shape[1] - 1):
        kol = X.iloc[:, ob]
        korak = (max(kol) - min(kol)) / brojKoraka
        kol = np.floor(kol / korak) * korak

        f = np.unique(kol)
        infoDA = 0
        for i in f:
            temp = Y[kol == i]

            infoDi = calculate_entropy(temp)
            Di = sum(kol == i)
            D = len(kol)

            infoDA += Di * infoDi / D

        IG[ob] = infoD - infoDA

    return IG

def preprocess_data(data):
    data.drop(columns="id", inplace=True)
    replace_values_in_bmi(data)
    one_hot_encoding(data, data.ever_married)
    one_hot_encoding(data, data.Residence_type)
    one_hot_encoding(data, data.work_type)
    classifier_for_smoking_status(data)
    data = remove_examples_with_spec_value(data, data.gender, 'Other')
    one_hot_encoding(data, data.smoking_status)
    one_hot_encoding(data, data.gender)
    data = swap_columns(data, 'stroke', data.columns[-1])
    return data

def get_correlation(data, string):
    correlation = data.corr(method=string)
    plt.figure()
    sns.heatmap(correlation, annot=True)
    plt.show()

def resample_data(data):
    X = data.drop(columns="stroke")
    Y = data.stroke
    oversample = imblearn.over_sampling.SMOTE()
    X, Y = oversample.fit_resample(X, Y)
    new_data = pd.concat([X, Y], axis=1)
    new_data.age = new_data.age.round()

    new_data = new_data.sample(frac=1).reset_index(drop=True)

    return new_data

def undersample_data(data):
    X = data.drop(columns="stroke")
    Y = data.stroke
    undersample = imblearn.under_sampling.RandomUnderSampler(random_state=0)
    X, Y = undersample.fit_resample(X, Y)
    new_data = pd.concat([X, Y], axis=1)
    new_data = new_data.sample(frac=1).reset_index(drop=True)

    return new_data

class Knn_classifier:
    def __init__(self, X_training, X_test, Y_training, Y_test, number_of_neighbors = None):
        self.data = data
        self.number_of_neighbors = number_of_neighbors
        self.predictions = []
        self.X_training, self.X_test, self.Y_training, self.Y_test = X_training, X_test, Y_training, Y_test





    def make_classification(self):
        mean_value_training = np.mean(self.X_training, axis=0)
        std_value_training = np.std(self.X_training, axis=0)
        X_norm_training = (self.X_training - mean_value_training) / std_value_training

        mean_value_test = np.mean(self.X_test, axis=0)
        std_value_test = np.std(self.X_test, axis=0)
        X_norm_test = (self.X_test - mean_value_test) / std_value_test

        for i in range(X_norm_test.shape[0]):
            curr_test = X_norm_test.iloc[i, :]
            dist = np.zeros((X_norm_training.shape[0],))
            for j in range(X_norm_training.shape[0]):
                curr_training = X_norm_training.iloc[j, :]
                dist[j] = np.sqrt(np.sum((curr_test - curr_training) ** 2))

            idx = np.argsort(dist)[:self.number_of_neighbors]
            closest_instance = self.Y_training.iloc[idx]
            values, count = np.unique(closest_instance, return_counts=True)

            self.predictions.append(values[np.argmax(count)])
        self.predictions = pd.Series(self.predictions)




    def evaluate_the_model(self):
        print("Rezultat modela", f1_score(self.Y_test, self.predictions))
        conf_mat = confusion_matrix(self.Y_test, self.predictions)
        sns.heatmap(conf_mat, annot=True, fmt='g', cbar=False)
        plt.show()

    def find_best_number_of_neighbors(self):
        number_of_neighbors_array = np.array([1, 2, 5, 10, 15])
        f1_average = np.zeros(len(number_of_neighbors_array))
        k_fold = KFold(n_splits=10)

        for i in range(len(number_of_neighbors_array)):
            f1_scores = []
            for train_index, test_index in k_fold.split(self.X_training):

                X_train_temp, X_test_temp, Y_train_temp, Y_test_temp = self.X_training.iloc[train_index, :],self.X_training.iloc[test_index, :], self.Y_training.iloc[train_index], self.Y_training.iloc[test_index]
                knn = Knn_classifier(X_train_temp, X_test_temp, Y_train_temp, Y_test_temp, number_of_neighbors_array[i])
                knn.make_classification()
                f1_scores.append(f1_score(knn.Y_test, knn.predictions))
            f1_average[i] = np.mean(f1_scores)

        best_number_of_neighbors = number_of_neighbors_array[np.argmax(f1_average)]
        plt.plot(number_of_neighbors_array, f1_average)
        plt.xlabel("number_of_neighbors")
        plt.ylabel("F1_score")
        plt.show()

        #print(best_number_of_neighbors)
        self.number_of_neighbors = best_number_of_neighbors

class Node:
    def __init__(self, data, empty_node = False):

        self.data =  data
        self.empty_node = empty_node
        self.number_of_thresholds = 10
        self.find_feature()
        self.find_threshold()
        self.children = []


    def find_feature(self):
        if(self.empty_node == True):
            return
        information_gains = calculate_information_gain(self.data)
        #self.feature = list(information_gains.keys())[0]
        self.feature = information_gains[0][0]
        #print("Izabrana osobina:", self.feature)

    def find_threshold(self):
        if(self.empty_node == True):
            return
        label_entropy = calculate_entropy(self.data.stroke)
        threshold_array = np.linspace(np.min(self.data[self.feature]), np.max(self.data[self.feature]), num=self.number_of_thresholds)
        information_gains = np.zeros(len(threshold_array))
        gini_indices = np.zeros(len(threshold_array))

        for i in range(len(threshold_array)):
            new_column = np.where(self.data[self.feature] < threshold_array[i], 0, 1)
            entropy_of_the_column = entropy_of_feature(new_column, self.data)
            information_gains[i] = label_entropy - entropy_of_the_column

        for i in range(len(threshold_array)):

            column_of_label_left = self.data.stroke.iloc[np.where(self.data[self.feature] < threshold_array[i])[0]]
            gini_index_left = calculate_gini_index(column_of_label_left)

            column_of_label_right = self.data.stroke.iloc[np.where(self.data[self.feature] >= threshold_array[i])[0]]
            gini_index_right = calculate_gini_index(column_of_label_right)


            gini_indices[i] = gini_index_left + gini_index_right



        self.threshold = threshold_array[np.argmax(information_gains)]
        #print("Izabrana granica:", self.threshold)

    def make_children(self):
        data1 = self.data[self.data[self.feature] <= self.threshold]
        data2 = self.data[self.data[self.feature] > self.threshold]
        if(data1.shape[0] == 0):
            self.children.append(Node(data1, empty_node=True))
        else:
            self.children.append(Node(data1))

        if(data2.shape[0] == 0):

            self.children.append(Node(data2, empty_node=True))
        else:
            self.children.append(Node(data2))

class Tree:
    def __init__(self, data, tree_depth = None):
        self.data = data
        self.tree_depth = tree_depth
        self.root = Node(data)
        self.make_tree()
        self.predictions = []

    def make_tree(self):
        if(self.tree_depth == None):
            return
        height = 0
        current_level = [self.root]
        next_level = []

        while(height < self.tree_depth):
            while(len(current_level) != 0):
                current_node = current_level.pop()
                if(current_node.empty_node == True):
                    continue
                current_node.make_children()
                next_level.append(current_node.children[0])
                next_level.append(current_node.children[1])
            current_level = next_level
            next_level = []
            height += 1

    def make_decision(self, test_data):
        N = test_data.shape[0]
        for i in range(N):
            current_node = self.root
            while(len(current_node.children) != 0):
                values = test_data.loc[:, current_node.feature]
                value = values.iloc[i]
                if(value <= current_node.threshold):
                    current_node = current_node.children[0]
                else:
                    current_node = current_node.children[1]
            values, counts = np.unique(current_node.data.stroke, return_counts=True)
            if(len(counts) == 0):
                self.predictions.append(0)
            else:
                self.predictions.append(values[np.argmax(counts)])
        self.predictions = pd.Series(self.predictions)

    def evaluate_the_model(self, test_data):
        print("Rezultat modela", f1_score(test_data, self.predictions))
        conf_mat = confusion_matrix(test_data, self.predictions)
        sns.heatmap(conf_mat, annot=True, fmt='g', cbar=False)
        plt.show()


    def find_best_depth_k_fold(self):
        depth_array = np.array([2, 3, 5, 7])
        f1_average = np.zeros(len(depth_array))
        k_fold = KFold(n_splits=10)
        X_temp = self.data.drop(columns="stroke")
        Y_temp = self.data.stroke


        for i in range(len(depth_array)):
            f1_scores = []
            for train_index, test_index in k_fold.split(X_temp):

                X_train_temp, X_test_temp, Y_train_temp, Y_test_temp = X_temp.iloc[train_index, :], X_temp.iloc[test_index, :], Y_temp.iloc[train_index], Y_temp.iloc[test_index]
                training_data_temp = pd.concat([X_train_temp, Y_train_temp], axis=1)
                training_tree = Tree(training_data_temp, depth_array[i])
                training_tree.make_decion(X_test_temp)
                f1_scores.append(f1_score(Y_test_temp, training_tree.predictions))
            f1_average[i] = np.mean(f1_scores)

        best_depth = depth_array[np.argmax(f1_average)]

        self.tree_depth = best_depth

data = pd.read_csv("07_stroke_prediction_dataset.csv")
data = pd.DataFrame(data)
quantitative = ['age', 'avg_glucose_level', 'bmi']

print(data)
print(data.head())
print(data[quantitative].describe().T)
data[quantitative].hist(bins=15, figsize=(12,7))
plt.show()

data=preprocess_data(data)


X = data.drop('stroke', axis=1)
Y = data.stroke
data_smote = resample_data(data)
X_smote = data_smote.drop('stroke', axis=1)
Y_smote = data_smote.stroke
X_training, X_test, Y_training, Y_test = train_test_split(X, Y, train_size=0.7, random_state=42, stratify=Y)
X_training_smote, X_test_smote, Y_training_smote, Y_test_smote = train_test_split(X_smote, Y_smote, train_size=0.7, random_state=42, stratify=Y_smote)

X_training_smote[quantitative].hist(bins=15, figsize=(12,7))
plt.show()



#INFORMATION GAIN; CORRELATION

information_gains = calculate_information_gain(data)
first_10_features = []
for i in information_gains[0:10]:
    first_10_features.append(i[0])
first_10_features.append('stroke')
get_correlation(data[first_10_features], 'pearson')
get_correlation(data[first_10_features], 'spearman')



#LDA

data_LDA = dimensionality_reduction_LDA(data[quantitative + ['stroke']], 1)
X_LDA = data_LDA.drop('Outcome', axis=1)
Y_LDA = data_LDA.Outcome
plt.figure(1)
plt.scatter(X_LDA.loc[Y_LDA==0], np.zeros((1, np.sum(Y_LDA==0))), marker='*')
plt.scatter(X_LDA.loc[Y_LDA==1], np.zeros((1, np.sum(Y_LDA==1))), marker='.')
plt.show()

#PCA

data_PCA = dimensionality_reduction_PCA(data[quantitative + ['stroke']], 2)
data_smote_PCA = dimensionality_reduction_PCA(data_smote[quantitative + ['stroke']], 2)
X_PCA = data_PCA.drop('Outcome', axis=1)
Y_PCA = data_PCA.Outcome
X_PCA_smote = data_smote_PCA.drop('Outcome', axis=1)
Y_PCA_smote = data_smote_PCA.Outcome

#LINEAR CLASSIFIER

first_3_features = []
for i in information_gains[0:3]:
    first_3_features.append(i[0])
plot_3d(data[first_3_features], data.stroke, 1000)
#Ytest, decision = bayes(data, first_3_features, 'stroke')
Ytest, decision = bayes(data_smote_PCA, ['PC1', 'PC2'], 'Outcome')
conf_matrix(Ytest, decision)

#KNN CLASSIFIER

undersample_data_new = undersample_data(data)
X_under = undersample_data_new.drop(columns="stroke")
Y_under = undersample_data_new.stroke
X_training_under, X_test_under, Y_training_under, Y_test_under = train_test_split(X_under, Y_under, train_size=2/3, random_state=42, stratify=Y_under)
knn = Knn_classifier(X_training_under, X_test_under, Y_training_under, Y_test_under)
knn.find_best_number_of_neighbors()
knn.make_classification()
knn.evaluate_the_model()



#DECISION TREE - BUILT IN

print("Ugradjeni Decision tree: \n")
classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(X_training_smote, Y_training_smote)
y_pred_dt = classifier.predict(X_test_smote)

conf_mat = confusion_matrix(Y_test_smote, y_pred_dt)
plt.figure()
sns.heatmap(conf_mat, annot=True, fmt='g', cbar=False)
plt.show()

print('F-score stabla odlučivanja iznosi: ' + str(f1_score(Y_test_smote, y_pred_dt)*100) + '%.')

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(classifier,
                   feature_names=X_training_smote.columns,
                   class_names=['0', '1'],
                   filled=True)
plt.show()

print("Grid Search Decision tree: \n")
criterion = ['gini', 'entropy']
max_depth = [4,2,6,8,10,12]
tree_param = {'criterion':['gini','entropy'],'max_depth':max_depth}
clf = GridSearchCV(tree.DecisionTreeClassifier(random_state=42), tree_param, cv=5)
clf.fit(X_training_smote, Y_training_smote)

print('Best Criterion:', clf.best_estimator_.get_params()['criterion'])
print('Best max_depth:', clf.best_estimator_.get_params()['max_depth'])

classifier = tree.DecisionTreeClassifier(criterion=clf.best_estimator_.get_params()['criterion'], max_depth=clf.best_estimator_.get_params()['max_depth'], random_state=42)
classifier = classifier.fit(X_training_smote, Y_training_smote)
y_pred_dt = classifier.predict(X_test_smote)

conf_mat = confusion_matrix(Y_test_smote, y_pred_dt)
plt.figure()
sns.heatmap(conf_mat, annot=True, fmt='g', cbar=False)
plt.show()

print('F-score stabla odlučivanja iznosi: ' + str(f1_score(Y_test_smote, y_pred_dt)*100) + '%.')
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(classifier,
                   feature_names=X_training_smote.columns,
                   class_names=['0', '1'],
                   filled=True)
plt.show()


#DECISION TREE - MADE

training_data_smote = pd.concat([X_training_smote, Y_training_smote], axis=1)
decision_tree = Tree(training_data_smote, 5)
decision_tree.find_best_depth_k_fold()
decision_tree.make_tree()
decision_tree.make_decision(X_test_smote)
decision_tree.evaluate_the_model(Y_test_smote)



#NEURAL NETWORK

model = Sequential()
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.build((None, min(X_training_smote.shape)))
model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()

history = model.fit(x= X_training_smote, y = Y_training_smote, epochs = 500, batch_size=500,  verbose=1)


YpredTrening = model.predict(X_training_smote, verbose=0)
YpredTrening = np.round(YpredTrening)

YpredTest = model.predict(X_test_smote)
YpredTest = np.round(YpredTest)

conf_mat = confusion_matrix(Y_test_smote, YpredTest)
sns.heatmap(conf_mat, annot=True, fmt='g', cbar=False)
plt.show()

Atrening = f1_score(Y_training_smote, YpredTrening)
print('F-score na trening skupu iznosi: ' + str(Atrening*100) + '%.')

Atest = f1_score(Y_test_smote, YpredTest)
print('F-score na test skupu iznosi: ' + str(Atest*100) + '%.')



