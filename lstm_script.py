import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from argparse import ArgumentParser
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

WINDOW_SIZE = 16
REMOVE_COLUMNS = [-5, -2, -1]
SELECT_COLUMNS = [0, 1, 3, 4]
TRAIN_PERCENTAGE = 0.5
T_AFTER_CUT_DIVISOR = 10
LSTM_UNITS = 50
OUTPUT_SERIES = 4


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def stateful_cut(arr, batch_size, T_after_cut):
    if len(arr.shape) != 3:
        # N: Independent sample size,
        # T: Time length,
        # m: Dimension
        print("ERROR: please format arr as a (N, T, m) array.")

    N = arr.shape[0]
    T = arr.shape[1]

    # We need T_after_cut * nb_cuts = T
    nb_cuts = int(T / T_after_cut)
    if nb_cuts * T_after_cut != T:
        print("ERROR: T_after_cut must divide T")

    # We need batch_size * nb_reset = N
    # If nb_reset = 1, we only reset after the whole epoch, so no need to reset
    nb_reset = int(N / batch_size)
    if nb_reset * batch_size != N:
        print("ERROR: batch_size must divide N")

    # Cutting (technical)
    cut1 = np.split(arr, nb_reset, axis=0)
    cut2 = [np.split(x, nb_cuts, axis=1) for x in cut1]
    cut3 = [np.concatenate(x) for x in cut2]
    cut4 = np.concatenate(cut3)
    return cut4


def plotting(history):
    plt.plot(history.history['loss'], color="red")
    plt.plot(history.history['val_loss'], color="blue")
    red_patch = mpatches.Patch(color='red', label='Training')
    blue_patch = mpatches.Patch(color='blue', label='Test')
    plt.legend(handles=[red_patch, blue_patch])
    plt.xlabel('Epochs')
    plt.ylabel('MSE loss')
    plt.show()


def main(csv_filename):
    df = read_csv(csv_filename, index_col=0, parse_dates=['_date'])
    n_features = df.shape[1]

    values = df.values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, WINDOW_SIZE, 1)
    reframed.drop(reframed.columns[REMOVE_COLUMNS], axis=1, inplace=True)

    values = reframed.values
    X = np.delete(values, [values.shape[1] - i - 1 for i in range(OUTPUT_SERIES)], axis=1)
    X_temp = np.zeros((WINDOW_SIZE, reframed.shape[0], n_features))
    for i in range(int(X.shape[1] / n_features)):
        X_temp[i] = X[:, [x + i * n_features for x in range(n_features)]]
    X = X_temp

    y_columns = []
    for i in range(int(values.shape[1] / n_features) - 1):
        y_columns.append([x + (i + 1) * n_features for x in SELECT_COLUMNS])
    last_elements = [values.shape[1] - i - 1 for i in range(OUTPUT_SERIES)]
    last_elements.reverse()
    y_columns.append(last_elements)
    y_columns = np.array(y_columns).reshape(OUTPUT_SERIES * len(y_columns))
    y = values[:, y_columns]
    y_temp = np.zeros((WINDOW_SIZE, reframed.shape[0], OUTPUT_SERIES))
    for i in range(int(y.shape[1] / OUTPUT_SERIES)):
        y_temp[i] = y[:, [x + i * OUTPUT_SERIES for x in range(OUTPUT_SERIES)]]
    y = y_temp
    print(X.shape, y.shape)

    Nsequence = WINDOW_SIZE
    Ntrain = int(Nsequence * TRAIN_PERCENTAGE)
    X_train = X[:Ntrain, :, :]
    X_test = X[Ntrain:, :, :]
    y_train = y[:Ntrain, :, :]
    y_test = y[Ntrain:, :, :]

    N = X_train.shape[0]  # size of samples
    T = X_train.shape[1]  # length of each time series
    batch_size = N  # number of time series considered together: batch_size | N
    T_after_cut = T  # length of each cut part of the time series: T_after_cut | T
    dim_in = X_train.shape[2]  # dimension of input time series
    dim_out = y_train.shape[2]  # dimension of output time series

    inputs, outputs, inputs_test, outputs_test = \
        [stateful_cut(arr, batch_size, T_after_cut) for arr in \
         [X_train, y_train, X_test, y_test]]

    np.random.seed(1337)
    model = Sequential()
    model.add(LSTM(batch_input_shape=(batch_size, None, dim_in),
                   return_sequences=True, units=LSTM_UNITS, stateful=True))
    model.add(TimeDistributed(Dense(activation='linear', units=dim_out)))
    model.compile(loss='mse', optimizer='rmsprop')

    epochs = 50
    nb_reset = int(N / batch_size)
    if nb_reset > 1:
        print("ERROR: We need to reset states when batch_size < N")

    # When nb_reset = 1, we do not need to reinitialize states
    history = model.fit(inputs, outputs, epochs=epochs,
                        batch_size=batch_size, shuffle=False,
                        validation_data=(inputs_test, outputs_test))
    plt.figure(figsize=(10, 8))
    plotting(history)

    model_stateless = Sequential()
    model_stateless.add(LSTM(input_shape=(None, dim_in),
                             return_sequences=True, units=LSTM_UNITS))
    model_stateless.add(TimeDistributed(Dense(activation='linear', units=dim_out)))
    model_stateless.compile(loss='mse', optimizer='rmsprop')
    model_stateless.set_weights(model.get_weights())

    i = 0  # time series selected (between 0 and N-1)
    x = X_train[i]
    y = y_train[i]
    inv_y = np.copy(x)
    inv_y[:, SELECT_COLUMNS] = y
    inv_y = scaler.inverse_transform(inv_y)
    y_hat = model_stateless.predict(np.array([x]))[0]
    inv_yhat = np.copy(x)
    inv_yhat[:, SELECT_COLUMNS] = y_hat
    inv_yhat = scaler.inverse_transform(inv_yhat)
    i = 0
    for dim in SELECT_COLUMNS:
        plt.figure(figsize=(10, 8))
        plt.plot(df.index[-T:], inv_y[:, dim], color='b', label='True')
        plt.plot(df.index[-T:], inv_yhat[:, dim], color='orange', label='Prediction')
        rmse = np.sqrt(mean_squared_error(y[:, i], y_hat[:, i]))
        i += 1
        print('Test RMSE on column {}: {}'.format(df.columns[dim], rmse))
        plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-c", "--csv", dest="csv_filename",
                        help="CSV prepared to be processed as multivar timeserie", metavar="CSV")

    args = parser.parse_args()
    main(args.csv_filename)
