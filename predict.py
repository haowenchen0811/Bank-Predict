import pandas as pd
from sklearn import model_selection
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np

pd.set_option("display.max_columns", None)
df = pd.read_csv("new_bank.csv")
print(df.info())
print(df.isnull().sum())
df.Geography.fillna(method="ffill", inplace=True)
x = df.groupby("Geography").transform("mean")
r = ["CreditScore", "Tenure", "NumOfProducts", "EstimatedSalary"]
# fill the missing value of their average value in different country
for row in r:
    df[row].fillna(x[row], inplace=True)

_, axss = plt.subplots(2, 3, figsize=[20, 10])
sns.boxplot(x="Exited", y="CreditScore", data=df, ax=axss[0][0])
sns.boxplot(x="Exited", y="Age", data=df, ax=axss[0][1])
sns.boxplot(x="Exited", y="Tenure", data=df, ax=axss[0][2])
sns.boxplot(x="Exited", y="Balance", data=df, ax=axss[1][0])
sns.boxplot(x="Exited", y="EstimatedSalary", data=df, ax=axss[1][1])
sns.boxplot(x="Exited", y="NumOfProducts", data=df, ax=axss[1][2])

_, axss = plt.subplots(2, 2, figsize=[20, 10])
sns.countplot(x="Exited", hue="Geography", data=df, ax=axss[0][0])
sns.countplot(x="Exited", hue="Gender", data=df, ax=axss[0][1])
sns.countplot(x="Exited", hue="HasCrCard", data=df, ax=axss[1][0])
sns.countplot(x="Exited", hue="IsActiveMember", data=df, ax=axss[1][1])
cor = df[["CreditScore", "Age", "Tenure", "NumOfProducts", "Balance", "Geography", "EstimatedSalary"]].corr()

plt.figure(3)
sns.heatmap(cor)
plt.show()
# feature preprocessing
df["Gender"] = df["Gender"] == "Male"
df = pd.get_dummies(df, columns=["Geography"])
y = df["Exited"]
x = df.drop(["RowNumber", "CustomerId", "Surname", "Exited"], axis=1)

# split the data for test
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, stratify=y, random_state=1)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

R = RandomForestClassifier()
K = KNeighborsClassifier()
L = LogisticRegression()
R.fit(x_train, y_train)

model_names = ["RandomForest", "KNN", "Logistic"]
model_list = [R, K, L]
count = 0
for classifier in model_list:
    score = model_selection.cross_val_score(classifier, x_train, y_train, cv=5)
    print(score)
    print("model accuracy of " + model_names[count] + str(score.mean()))
    count += 1

from sklearn.model_selection import GridSearchCV


def print_grid_search_metrics(gs):
    print ("Best score: " + str(gs.best_score_))
    print ("Best parameters set:")
    best_parameters = gs.best_params_
    for param_name in sorted(best_parameters.keys()):
        print(param_name + ':' + str(best_parameters[param_name]))

parameters = {
    'penalty': ("l1", "l2"),
    "C": (0.01, 0.1, 1, 5, 10)
}
gridLR = GridSearchCV(LogisticRegression(solver="liblinear"), parameters, cv=5)
gridLR.fit(x_train, y_train)
print_grid_search_metrics(gridLR)
bestLR=gridLR.best_estimator_

parameters={
    'n_estimators':[40,40,45,50,55,60]
}

GridRF = GridSearchCV(RandomForestClassifier(),parameters,cv=5)
GridRF.fit(x_train,y_train)
print_grid_search_metrics(GridRF)
bestRF=GridRF.best_estimator_

parameters = {
    'n_neighbors':[1,2,3,4,5,6,7,8,9]
}
Grid_KNN = GridSearchCV(KNeighborsClassifier(),parameters, cv=5)
Grid_KNN.fit(x_train,y_train)
print_grid_search_metrics(Grid_KNN)
bestKNN = Grid_KNN.best_estimator_


# calculate accuracy, precision and recall, [[tn, fp],[]]
def cal_evaluation(classifier, cm):
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]
    accuracy  = (tp + tn) / (tp + fp + fn + tn + 0.0)
    precision = tp / (tp + fp + 0.0)
    recall = tp / (tp + fn + 0.0)
    print (classifier)
    print ("Accuracy is: " + str(accuracy))
    print ("precision is: " + str(precision))
    print ("recall is: " + str(recall))
    print ()

# print out confusion matrices
def draw_confusion_matrices(confusion_matricies):
    class_names = ['Not','Churn']
    for cm in confusion_matrices:
        classifier, cm = cm[0], cm[1]
        cal_evaluation(classifier, cm)

confusion_matrices = [
    ("Random Forest", confusion_matrix(y_test,bestRF.predict(x_test))),
    ("Logistic Regression", confusion_matrix(y_test,bestLR.predict(x_test))),
    ("K nearest neighbor", confusion_matrix(y_test, bestKNN.predict(x_test)))
]

draw_confusion_matrices(confusion_matrices)

from sklearn.metrics import roc_curve
from sklearn import metrics

# Use predict_proba to get the probability results of Random Forest
y_pred_rf = bestRF.predict_proba(x_test)[:, 1]
fpr_rf, tpr_rf, thresh = roc_curve(y_test, y_pred_rf)
bestRF.predict_proba(x_test)

# ROC curve of Random Forest result
plt.figure(11)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='RF')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve of  RF model')
plt.legend(loc='best')
plt.show()

from sklearn import metrics

# AUC score
print(metrics.auc(fpr_rf,tpr_rf))


# Use predict_proba to get the probability results of Logistic Regression
y_pred_lr = bestLR.predict_proba(x_test)[:, 1]
fpr_lr, tpr_lr, thresh = roc_curve(y_test, y_pred_lr)
bestLR.predict_proba(x_test)


# ROC Curve
plt.figure(10)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_lr, tpr_lr, label='LR')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve of LR Model')
plt.legend(loc='best')
plt.show()

# AUC score
print(metrics.auc(fpr_lr,tpr_lr))

x_with_corr = x.copy()
x_with_corr['SalaryInRMB'] = x['EstimatedSalary'] * 6.91
x_with_corr.head()

# add L1 regularization to logistic regression
# check the coef for feature selection
scaler = StandardScaler()
x_l1 = scaler.fit_transform(x_with_corr)
LRmodel_l1 = LogisticRegression(penalty="l1", C = 0.1, solver='liblinear')
LRmodel_l1.fit(x_l1, y)

indices = np.argsort(abs(LRmodel_l1.coef_[0]))[::-1]

print ("Logistic Regression (L1) Coefficients")
for ind in range(x_with_corr.shape[1]):
  print ("{0} : {1}".format(x_with_corr.columns[indices[ind]],round(LRmodel_l1.coef_[0][indices[ind]], 4)))

# add L2 regularization to logistic regression
# check the coef for feature selection
np.random.seed()
scaler = StandardScaler()
x_l2 = scaler.fit_transform(x_with_corr)
LRmodel_l2 = LogisticRegression(penalty="l2", C = 0.1, solver='liblinear', random_state=42)
LRmodel_l2.fit(x_l2, y)
LRmodel_l2.coef_[0]

indices = np.argsort(abs(LRmodel_l2.coef_[0]))[::-1]

print ("Logistic Regression (L2) Coefficients")
for ind in range(x_with_corr.shape[1]):
  print ("{0} : {1}".format(x_with_corr.columns[indices[ind]],round(LRmodel_l2.coef_[0][indices[ind]], 4)))

# check feature importance of random forest for feature selection
forest = RandomForestClassifier()
forest.fit(x, y)

importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature importance ranking by Random Forest Model:")
for ind in range(x.shape[1]):
  print ("{0} : {1}".format(x.columns[indices[ind]],round(importances[indices[ind]], 4)))