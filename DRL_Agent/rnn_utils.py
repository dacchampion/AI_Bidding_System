from keras.layers import Dense, LSTM, TimeDistributed, GRU
from keras.models import Sequential
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
import numpy as np


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


def convert_to_timeseries(observations_df, predictor_cols, target_cols):
    values = observations_df.values.astype('float32')
    min_values = np.min(values, axis=0) * 0.5
    max_values = np.max(values, axis=0) * 1.5
    values = np.vstack([values, min_values])
    values = np.vstack([values, max_values])
    scaler = MinMaxScaler(feature_range=(0, 1))
    values = values[:-2, :]
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    num_features = values.shape[1]
    reframed = series_to_supervised(scaled, 1, 1)
    inv_target_cols = [num_features + x for x in target_cols]
    reframed = reframed.iloc[:, np.concatenate((predictor_cols, np.array(inv_target_cols)))]
    return reframed, scaler


def split_train_test(ts_df, num_pred_vars, num_targets, training_split=0.75):
    # split into train and test sets
    values = ts_df.values
    n_train_days = int(values.shape[0] * training_split)
    train = values[:n_train_days, :]
    test = values[n_train_days:, :]
    # split into input and outputs
    train_X, train_y = train[:, :num_pred_vars], train[:, -num_targets:]
    test_X, test_y = test[:, :num_pred_vars], test[:, -num_targets:]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    return train_X, train_y, test_X, test_y


def train_lstm(tr_X, tr_y, te_X, te_y, lstm_units=20, num_epochs=25, batch_sz=8):
    model = Sequential()
    model.add(LSTM(lstm_units, input_shape=(tr_X.shape[1], tr_X.shape[2])))
    if tr_X.shape[1] == 1:
        model.add(Dense(activation='linear', units=tr_y.shape[1]))
    else:
        model.add(TimeDistributed(Dense(activation='linear', units=tr_y.shape[1])))
    model.compile(loss='mse', optimizer='adam')
    train_history = model.fit(tr_X, tr_y, epochs=num_epochs, batch_size=batch_sz, validation_data=(te_X, te_y),
                              verbose=0,
                              shuffle=False)

    return model, train_history


def train_gru(tr_X, tr_y, te_X, te_y, gru_units=20, num_epochs=25, batch_sz=8):
    model = Sequential()
    model.add(GRU(gru_units, input_shape=(tr_X.shape[1], tr_X.shape[2])))
    if tr_X.shape[1] == 1:
        model.add(Dense(activation='linear', units=tr_y.shape[1]))
    else:
        model.add(TimeDistributed(Dense(activation='linear', units=tr_y.shape[1])))
    model.compile(loss='mse', optimizer='adam')
    train_history = model.fit(tr_X, tr_y, epochs=num_epochs, batch_size=batch_sz, validation_data=(te_X, te_y),
                              verbose=0,
                              shuffle=False)

    return model, train_history

