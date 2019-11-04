import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
iris_data = pd.read_csv("iris_data.csv" ) # , names=column_names )
titanic = pd.read_csv("titanic-train.csv" ) # , names=column_names )
titanic_test = pd.read_csv("titanic-test.csv" ) # , names=column_names )

"""
Name: Katherine Tung
Date: 11/4/19

This program uses the classic Iris data set. The program uses a Perceptron to classify Iris setosa with petal width and
length, to classify Iris virginica with sepal width and length, to classify Iris virginica with petal width and length,
and to classify Iris virginica with all four inputs (petal width, petal length, sepal width, petal length). The program 
also takes in the Titanic data set and uses a Perceptron to classify survivors of the Titanic. Then, the program 
makes predictions using a test data set and outputs a .csv with predictions. 
"""
# Data cleaning stuff
titanic['Embarked'].fillna('S',inplace=True) #fill missing with most common
le = LabelEncoder()
le.fit(titanic['Sex'])
titanic['SexNum'] = le.transform(titanic['Sex'])
titanic['NameLength'] = titanic['Name'].str.len()
titanic = pd.concat([titanic, pd.get_dummies(titanic['Embarked'])],axis=1)
titanic['Age'].fillna(titanic['Age'].mean(), inplace=True)

titanic_test['SexNum'] = le.transform(titanic_test['Sex'])
titanic_test['NameLength'] = titanic_test['Name'].str.len()
titanic_test = pd.concat([titanic_test, pd.get_dummies(titanic_test['Embarked'])],axis=1)
titanic_test['Age'].fillna(titanic['Age'].mean(), inplace=True)
titanic_test['Fare'].fillna(titanic['Fare'].mean(), inplace=True)




def print_conf_matrix(targets, outputs):
    cm = confusion_matrix(targets, outputs)
    print("Confusion Matrix:")
    print("     PN PP")
    print("AN: "+ str(cm[0]))
    print("AP: "+ str(cm[1]))

def perceptron_iris(flower, predict, inputs, no_change, input_type, data_frame, verbose = True):

    def class_to_targets(target):
        if target == flower:
            return 1
        else:
            return 0


    # Full fit with two inputs
    iris_inputs = data_frame[ inputs ]
    iris_targets = data_frame[predict].apply(class_to_targets)

    percey = Perceptron(n_iter_no_change=no_change)
    if (verbose):
        print("Training a Perceptron to classify " + str(flower) + " using "+ str(input_type) +" inputs.")
    percey.fit(iris_inputs, iris_targets)
    percey_outputs = percey.predict(iris_inputs)
    if (verbose):
        print("Found weights=" + str(percey.coef_) + " and threshold: " + str(percey.intercept_) +
              " in " + str(percey.n_iter_) + " epochs.")
        print("Mean accuracy:", percey.score(iris_inputs, iris_targets))
        print_conf_matrix(iris_targets, percey_outputs)
        print("Precision = TP / (TP + FP) = ", precision_score(iris_targets, percey_outputs))
        print("Recall = TP / (TP + FN) = ", recall_score(iris_targets, percey_outputs))
    return percey.score(iris_inputs, iris_targets)

# 1. use a Perceptron to classify setosa with the petal inputs.
perceptron_iris('Iris-setosa', 'class', ['petal width', 'petal length'], 5, 'petal width and petal length', iris_data)

# 2. use a Perceptron to classify virginica with the sepal inputs.
perceptron_iris('Iris-virginica', 'class', ['sepal width', 'sepal length'], 19, 'sepal width and sepal length',
                iris_data) #best to 23

# 3. use a Perceptron to classify virginica with the petal inputs.
perceptron_iris('Iris-virginica', 'class', ['petal width', 'petal length'], 5, 'petal width and petal length',
                iris_data)

# 4. use a Perceptron to classify virginica with all four inputs.
perceptron_iris('Iris-virginica', 'class', ['petal width', 'petal length', 'sepal width', 'sepal length'], 5,
                'petal width, petal length, sepal width, sepal length', iris_data)

# 4. use a Perceptron to classify virginica with all four inputs.
perceptron_iris('Iris-virginica', 'class', ['petal width', 'petal length', 'sepal width', 'sepal length'], 5,
                'petal width, petal length, sepal width, sepal length', iris_data)

# 5. use a Perceptron to classify survivors of the Titanic.

running_best = 0
running_index = 2
for i in range(2, 301):
    result_to_try = perceptron_iris(1, 'Survived', ['Pclass', 'SexNum', 'SibSp', 'Parch', 'Fare', 'C', 'Q', 'S',
                                        'Age', 'NameLength'],
                                    i, 'Pclass, SexNum, SibSp, Parch, Fare, C, Q, S, Age, NameLength',
                                    titanic, verbose=False)
    if (running_best < result_to_try):
        running_best = result_to_try
        running_index = i
perceptron_iris(1, 'Survived', ['Pclass', 'SexNum', 'SibSp', 'Parch', 'Fare', 'C', 'Q', 'S', 'Age', 'NameLength'],
                running_index, 'Pclass, SexNum, SibSp, Parch, Fare, C, Q, S, Age, NameLength', titanic)


#Test inputs

titanic_inputs = titanic[ ['Pclass', 'SexNum', 'SibSp', 'Parch', 'Fare', 'C', 'Q', 'S',
                                    'Age', 'NameLength'] ]
titanic_targets = titanic['Survived']

percey = Perceptron(n_iter_no_change=running_index)
percey.fit(titanic_inputs, titanic_targets)
percey_outputs = percey.predict(titanic_test[ ['Pclass', 'SexNum', 'SibSp', 'Parch', 'Fare', 'C', 'Q', 'S',
                                    'Age', 'NameLength'] ])
pd.concat([titanic_test['PassengerId'], pd.DataFrame(percey_outputs)],axis=1).to_csv('KT_titanic_test_pred.csv')



