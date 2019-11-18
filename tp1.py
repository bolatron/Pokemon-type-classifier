import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier

import graphviz

from id3 import Id3Estimator
from id3 import export_graphviz
from sklearn.datasets import load_breast_cancer

def id3(size):
    x_train, x_test, y_train, y_test = setData("Pokemon.csv", 718, size)

    est = Id3Estimator(gain_ratio=True)
    est.fit(x_train, y_train)
    y_test = y_test.to_numpy()
    y_train = y_train.to_numpy()
    y_predict = est.predict(x_test)
    y_predict2 = est.predict(x_train)
    error_1 = 0
    error_2 = 0

    for i in range(len(y_test)):
        if y_predict[i] != y_test[i]:
            error_1 = error_1 + 1

    for i in range(len(y_train)):
        if y_predict2[i] != y_train[i]:
            error_2 = error_2 + 1

    #dot = export_graphviz(est.tree_, 'tree.dot', bunch.feature_names)

    return error_1/len(y_test)*100, error_2/len(y_train)*100

def ploot(var_x, var_y, var_y2):

    fig, ax = plt.subplots()
    ax.plot(var_x, var_y, label='Teste')
    ax.plot(var_x, var_y2, label='Treinamento')
    ax.set(xlabel='Proporção dos dados de teste', ylabel='Erro (%)',
           title='Erro Relativo ao Algoritmo ID3')
    ax.grid()
    plt.legend()
    plt.show()

def setData(file, size_lock, size):
    # Read file
    lb = LabelEncoder()
    pokedata = pd.read_csv(file)

    pokedata["Type_"] = lb.fit_transform(pokedata["Type 1"])
    pokedata["Habitat_"] = lb.fit_transform(pokedata["Habitat"])
    pokedata["Color_"] = lb.fit_transform(pokedata["Color"])
    pokedata["BodyStyle_"] = lb.fit_transform(pokedata["Body Style"])
    pokedata["Strong_"] = lb.fit_transform(pokedata["Strong Against"])
    pokedata["Weak_"] = lb.fit_transform(pokedata["Weak Against"])
    pokedata["Resistant_"] = lb.fit_transform(pokedata["Resistant to"])
    pokedata["Vulnerable_"] = lb.fit_transform(pokedata["Vulnerable to"])

    df = pokedata.iloc[:size_lock, 12:19]

    pokedata = pokedata.sort_values(by=["Type_"])

    x = pokedata.iloc[:size_lock, 12:17]
    y = pokedata.iloc[:size_lock, 11]

    bug_type = pokedata.iloc[:63]
    dark_type = pokedata.iloc[63:91]
    dragon_type = pokedata.iloc[91:115]
    eletric_type = pokedata.iloc[115:151]
    fairy_type = pokedata.iloc[151:168]
    fighting_type = pokedata.iloc[168:193]
    fire_type = pokedata.iloc[193:239]
    flying_type = pokedata.iloc[239:242]
    ghost_type = pokedata.iloc[242:265]
    grass_type = pokedata.iloc[265:331]
    ground_type = pokedata.iloc[331:361]
    ice_type = pokedata.iloc[361:384]
    normal_type = pokedata.iloc[384:477]
    poison_type = pokedata.iloc[477:505]
    psychic_type = pokedata.iloc[505:551]
    rock_type = pokedata.iloc[551:591]
    steel_type = pokedata.iloc[591:613]
    water_type = pokedata.iloc[613:]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=size, random_state=1)
    print(len(x_test)/(len(x_train)+len(x_test)))
    return x_train, x_test, y_train, y_test

def entropy(labels, base=None):
  vc = pd.Series(labels).value_counts(normalize=True, sort=False)
  return -(vc * np.log(vc)/np.log(base)).sum()

''' Treinamento baseado em Arvore de Decisao '''
def pokeTree(size):

    x_train, x_test, y_train, y_test = setData("Pokemon.csv", 718, size)

    clf = tree.DecisionTreeClassifier(criterion="gini")
    # Create decision tree based on data
    clf = clf.fit(x_train, y_train)

    #tree.plot_tree(clf)
    #plt.show()

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
    print(error_2)
    print(error_1)
    return error_1/len(y_test)*100, error_2/len(y_train)*100

def pokeNN(x):

    x_train, x_test, y_train, y_test = setData("Pokemon.csv", 717, 0.3)

    clf = clf = MLPClassifier(max_iter=x)
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
    print('---------------------')
    print('-------- MENU -------')
    print('---------------------')

    key = 0

    while(key != 4):
        print('Escolha o algoritmo: ')
        print('    1 -  CART')
        print('    2 -  ID3 ')
        print('    3 -  MLP ')
        print('    4 -  SAIR')
        key = int(input('Digite: '))

        var_x = []
        var_y = []
        var_y2 = []
        if(key == 1):
            x = 0.2
            while(x <= 0.9):
                y, y2 = pokeTree(x)
                x = x + 0.1
                var_x.append(x)
                var_y.append(y)
                var_y2.append(y2)
            ploot(var_x, var_y, var_y2)

        if(key == 2):
            x = 0.2
            while(x <= 0.9):
                y, y2 = id3(x)
                x = x + 0.1
                var_x.append(x)
                var_y.append(y)
                var_y2.append(y2)
            ploot(var_x, var_y, var_y2)

        if(key == 3):
            x = 100
            while(x <= 2000):
                y, y2 = pokeNN(x)
                x = x + 100
                var_x.append(x)
                var_y.append(y)
                var_y2.append(y2)
            ploot(var_x, var_y, var_y2)

    print('PROGRAMA ENCERRADO!')

if __name__ == '__main__':
    main()
