import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
import numpy as np
from sklearn.feature_selection import *
# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn import svm, tree
# from sklearn.neural_network import MLPRegressor
predict_index = None


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


def read_data():
    data_frame = pd.read_csv('data/DrugEfficacy_train.csv', index_col=0)
    data_train = data_frame.loc[~np.isnan(data_frame['Viability'])]
    data_predict = data_frame.loc[np.isnan(data_frame['Viability'])]
    train_x = data_train.filter(regex='^D', axis=1)
    train_y = data_train.filter(regex='Viability', axis=1)
    predict_x = data_predict.filter(regex='^D', axis=1)
    global predict_index
    predict_index = data_predict.index.values
    return train_x.values, train_y.values.T[0], predict_x.values


def write_result(predict_y):
    result_data_frame = pd.DataFrame(
        predict_y, index=predict_index, columns=['Viability'])
    result_data_frame.index.name = 'ID'
    result_data_frame.to_csv('data/predict.csv', encoding='utf-8')


if __name__ == '__main__':
    result = []
    X, y, predict_x = read_data()
    for train_index, test_index in KFold(n_splits=5).split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        res = test(X_train, X_test, y_train, y_test)
        result.append(res)
    res_df = pd.DataFrame(result, index=range(len(result)))
    print res_df.mean()
    # predict_y = regressor.predict(predict_x)
    # write_result(predict_y)
