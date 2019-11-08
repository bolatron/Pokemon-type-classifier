import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier

def ploot(var_x, var_y, var_y2):

    fig, ax = plt.subplots()
    ax.plot(var_x, var_y, label='Test Line')
    ax.plot(var_x, var_y2, label='Train Line')
    ax.set(xlabel='Quantidade de Camadas Intermediárias', ylabel='Erro (%)',
           title='Erro Relativo ao Algoritmo MLP')
    ax.grid()
    plt.legend()
    plt.show()

def setData(file, size_lock):
    # Read file
    lb = LabelEncoder()
    pokedata = pd.read_csv(file)

    pokedata["Type_"] = lb.fit_transform(pokedata["Type 1"])
    pokedata["Habitat_"] = lb.fit_transform(pokedata["Habitat"])
    pokedata["Color_"] = lb.fit_transform(pokedata["Color"])
    pokedata["BodyStyle_"] = lb.fit_transform(pokedata["Body Style"])

    x = pokedata.iloc[:size_lock, 15:18]
    y = pokedata.iloc[:size_lock, 14]
    a = pokedata["Type 1"].value_counts() + pokedata["Type 2"].value_counts()
    a.plot.barh()
    plt.legend()
    plt.show()
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
    return x_train, x_test, y_train, y_test

def entropy(labels, base=None):
  vc = pd.Series(labels).value_counts(normalize=True, sort=False)
  return -(vc * np.log(vc)/np.log(base)).sum()

''' Treinamento baseado em Arvore de Decisao '''
def pokeTree(size_lock):

    x_train, x_test, y_train, y_test = setData("Pokemon.csv", size_lock)

    clf = tree.DecisionTreeClassifier(criterion="entropy")
    # Create decision tree based on data
    clf = clf.fit(x_train, y_train)

    y_test = y_test.to_numpy()
    y_train = y_train.to_numpy()
    y_predict = clf.predict(x_test)
    y_predict2 = clf.predict(x_train)
    error_1 = 0
    error_2 = 0

    for i in range(len(y_test)):
        if y_predict[i] != y_test[i]:
            error_1 = error_1 + 1

    for i in range(len(y_train)):
        if y_predict2[i] != y_train[i]:
            error_2 = error_2 + 1
    #print(error)
    return error_1/len(y_test)*100, error_2/len(y_train)*100

def pokeNN(size_lock):

    x_train, x_test, y_train, y_test = setData("Pokemon.csv", 717)

    clf = MLPClassifier(hidden_layer_sizes=(500, 3), max_iter=1000, solver='adam', activation='logistic')
    clf.fit(x_train, y_train)

    y_test = y_test.to_numpy()
    y_train = y_train.to_numpy()
    y_predict = clf.predict(x_test)
    y_predict2 = clf.predict(x_train)
    error_1 = 0
    error_2 = 0

    for i in range(len(y_test)):
        if y_predict[i] != y_test[i]:
            error_1 = error_1 + 1

    for i in range(len(y_train)):
        if y_predict2[i] != y_train[i]:
            error_2 = error_2 + 1
    #print(error)
    return error_1/len(y_test)*100, error_2/len(y_train)*100

def main():
    '''
    var_x = []
    var_y = []
    var_y2 = []
    x = 1
    while(x <= 1):
        y, y2 = pokeNN(x)
        print(y, y2)
        x = x + 1
        var_x.append(x)
        var_y.append(y)
        var_y2.append(y2)
    ploot(var_x, var_y, var_y2)
    '''
    setData("Pokemon.csv", 718)

if __name__ == '__main__':
    main()
