import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

predict_index = None


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
    result_data_frame = pd.DataFrame(predict_y, index=predict_index, columns=['Viability'])
    result_data_frame.index.name = 'ID'
    result_data_frame.to_csv('data/predict.csv', encoding='utf-8')

if __name__ == '__main__':
    train_x, train_y, predict_x = read_data()
    regressor = GradientBoostingRegressor()
    regressor.fit(train_x, train_y)
    predict_y = regressor.predict(predict_x)
    write_result(predict_y)
