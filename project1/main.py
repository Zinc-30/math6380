from sklearn.cross_validation import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn import svm, tree
from sklearn.model_selection import KFold
import pandas as pd
from sklearn import linear_model
import numpy as np
from sklearn.feature_selection import *
# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# from sklearn.neural_network import MLPRegressor


def test(X_train, X_test, y_train, y_test):
    '''run test test dataset'''
    names = ["lasso", "ridge", "Elastic Net",
             "lars", "lasso Lars", "bayes ridge", "kernel ridge", "boost", "svr", "decision tree"]
    classifiers = [linear_model.Lasso(alpha=1),
                   linear_model.Ridge(alpha=1),
                   linear_model.ElasticNet(alpha=0.4, l1_ratio=0.55),
                   linear_model.Lars(n_nonzero_coefs=1),
                   linear_model.LassoLars(alpha=0.1),
                   linear_model.BayesianRidge(),
                   KernelRidge(alpha=0.5),
                   GradientBoostingRegressor(),
                   svm.SVR(),
                   tree.DecisionTreeRegressor()
                   ]

    res = dict()
    # print '==================='
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_pred_t = clf.predict(X_train)
        # print name, 'test', mean_squared_error(y_test, y_pred)
        # print name, 'train', mean_squared_error(y_train, y_pred_t)
        res[name] = mean_squared_error(y_test, y_pred)
        res[name + '_train'] = mean_squared_error(y_train, y_pred_t)
    return res


def main():
    #   read data
    dfx = pd.read_csv('data/CNV_train.csv', index_col=0)
    for name in ['mRNA', 'mutation']:
        df1 = pd.get_dummies(pd.read_csv(
            'data/' + name + '_train.csv', index_col=0).fillna(0))
        dfx = pd.concat([dfx, df1], axis=1)
    X = dfx.values
    dfy = pd.read_csv('data/values_train.csv', index_col=0)
    y = dfy['72hr'].values
    # print X.shape, y.shape
    result = []
    #  data split
    for train_index, test_index in KFold(n_splits=5).split(X):
        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y, test_size=0.3)
        # data split
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # outlier detection
        # index = IsolationForest().fit(X_train).predict(X_train) == 1
        # X_train = X_train[index]
        # y_train = y_train[index]
        # standard scale
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        # feature selection
        # 1
        pca = PCA(n_components=200)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        # 2
        # model = SelectKBest(f_regression, k=100).fit(X_train, y_train)
        # 3
        # model = SelectFromModel(linear_model.Lasso(
        #     alpha=0.4).fit(X_train, y_train), prefit=True)

        # X_train = model.transform(X_train)
        # X_test = model.transform(X_test)
        # 4
        # selected_columns = np.array([23785, 31268, 19121, 32447, 6401, 17821, 16546, 25689, 23925,
        #                              29175, 31200, 29976, 23590, 25139, 17109, 20798, 24812, 20066, 22101,
        #                              21786, 17576, 30569])
        # X_train = X_train[:, selected_columns]
        # X_test = X_test[:, selected_columns]

#  training and test
        res = test(X_train, X_test, y_train, y_test)
        result.append(res)
    res_df = pd.DataFrame(result, index=range(len(result)))
    print res_df.mean()


def predict():
    clf = GradientBoostingRegressor()
    dfx = pd.read_csv('data/CNV_train.csv', index_col=0)
    for name in ['mRNA', 'mutation']:
        df1 = pd.get_dummies(pd.read_csv(
            'data/' + name + '_train.csv', index_col=0).fillna(0))
        dfx = pd.concat([dfx, df1], axis=1)
    X = dfx.values
    dfy = pd.read_csv('data/values_train.csv', index_col=0)
    y = dfy['72hr'].values

    dfx = pd.read_csv('data/CNV_test.csv', index_col=0)
    for name in ['mRNA', 'mutation']:
        df1 = pd.get_dummies(pd.read_csv(
            'data/' + name + '_test.csv', index_col=0).fillna(0))
        dfx = pd.concat([dfx, df1], axis=1)
    X_test = dfx.values
    print X.shape, X_test.shape

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X_test = scaler.transform(X_test)
    # model = SelectKBest(f_regression, k=100).fit(X, y)
    # X_new = model.transform(X)
    # X_test_new = model.transform(X_test)
    # selected_columns = np.array([23785, 31268, 19121, 32447, 6401, 17821, 16546, 25689, 23925,
    #                              29175, 31200, 29976, 23590, 25139, 17109, 20798, 24812, 20066, 22101,
    #                              21786, 17576, 30569])
    # X_new = X[:, selected_columns]
    # X_test_new = X_test[:, selected_columns]
    clf.fit(X, y)
    y_pred = clf.predict(X_test)
    with open('data/ans.csv', 'w') as f:
        print >>f, 'index,IC50_72hrs'
        for i in range(len(dfx.index)):
            print>>f, dfx.index[i][5:], ',', y_pred[i]

if __name__ == '__main__':
    main()
    # predict()
