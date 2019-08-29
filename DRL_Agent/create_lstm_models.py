import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from argparse import ArgumentParser

from DRL_Agent.rnn_utils import convert_to_timeseries, split_train_test, train_lstm, train_gru


def main(a):
    a.predictors = np.array(a.predictors.strip('[]').split(','), dtype=np.int)
    a.targets = np.array(a.targets.strip('[]').split(','), dtype=np.int)
    df = pd.read_csv(a.csv_filename, index_col=a.date_field_name, parse_dates=[a.date_field_name])
    report_df = pd.DataFrame()
    kwds_dt_groups = df.groupby([a.k_field, a.d_field])
    for grp_values, _ in kwds_dt_groups:
        kwd = grp_values[0]
        device_type = grp_values[1]
        kwd_df = df[(df[a.k_field] == kwd) & (df[a.d_field] == device_type)].copy()
        del kwd_df[a.k_field], kwd_df[a.d_field]
        time_steps = kwd_df.shape[0]
        num_features = kwd_df.shape[1]
        model_results = {}
        if time_steps >= 32:
            timeseries_df, scaler_obj = convert_to_timeseries(kwd_df, a.predictors, a.targets)
            time_steps = timeseries_df.shape[0]
            print("Processing keyword '{}' from device '{}' with {} timesteps".format(kwd, device_type, time_steps))
            train_X, train_y, test_X, test_y = split_train_test(timeseries_df, len(a.predictors), len(a.targets))
            if a.rnn_type == 'LSTM':
                rnn, history_log = train_lstm(train_X, train_y, test_X, test_y, lstm_units=30, num_epochs=50)
            else:
                rnn, history_log = train_gru(train_X, train_y, test_X, test_y, gru_units=30, num_epochs=50)

            X = np.concatenate((train_X, test_X))
            y = np.concatenate((train_y, test_y))

            y_hat = rnn.predict(X)

            inv_data = np.zeros((time_steps, num_features))
            X = X.reshape((X.shape[0], X.shape[2]))
            inv_data[:, a.predictors] = X
            inv_data[:, a.targets] = y
            inv_pred = np.copy(inv_data)
            inv_data = scaler_obj.inverse_transform(inv_data)
            inv_pred[:, a.targets] = y_hat
            inv_pred = scaler_obj.inverse_transform(inv_pred)

            ## tweak for processing the prediction in case of 'tenerife holidays'
            if kwd == 'tenerife holidays':
                compare_file = 'model_comparison_{}.csv'
                results_df = pd.read_csv(compare_file.format(device_type))
                results_df['sessions_gru'] = inv_pred[:, 0]
                results_df.to_csv(compare_file.format(device_type), index=False)
            ##

            for i, col in enumerate(a.targets):
                rmse_scaled = np.sqrt(mean_squared_error(y[:, i], y_hat[:, i]))
                rmse_inverse = np.sqrt(mean_squared_error(inv_data[:, col], inv_pred[:, col]))
                col_name = kwd_df.columns[col]
                model_results['{}_{}'.format(col_name, 'min')] = inv_data[:, col].min()
                model_results['{}_{}'.format(col_name, 'max')] = inv_data[:, col].max()
                model_results['{}_{}'.format(col_name, 'rmse_s')] = rmse_scaled
                model_results['{}_{}'.format(col_name, 'rmse_i')] = rmse_inverse
            model_results['loss'] = history_log.history['loss'][-1]
            model_results['val_loss'] = history_log.history['val_loss'][-1]
        else:
            for i, col in enumerate(a.targets):
                col_name = kwd_df.columns[col]
                model_results['{}_{}'.format(col_name, 'min')] = np.nan
                model_results['{}_{}'.format(col_name, 'max')] = np.nan
                model_results['{}_{}'.format(col_name, 'rmse_s')] = np.nan
                model_results['{}_{}'.format(col_name, 'rmse_i')] = np.nan
            model_results['loss'] = np.nan
            model_results['val_loss'] = np.nan

        model_results['keyword'] = kwd
        model_results['device_type'] = device_type
        model_results['time_steps'] = time_steps
        filename = '{}{}_{}_{}.h5'.format(a.models_dir, a.rnn_type, kwd, device_type.split(' ')[0])
        model_results['model_file'] = filename
        rnn.save(filename)
        report_df = report_df.append(model_results, ignore_index=True)
        del kwd_df

    report_df.to_csv(a.report_filename, index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-c", "--csv", dest="csv_filename",
                        help="Source CSV to get features for training", metavar="CSV")
    parser.add_argument("-r", "--report", dest="report_filename",
                        help="CSV to report values on different models performance", metavar="Report CSV")
    parser.add_argument("-d", "--date_field", dest="date_field_name",
                        help="Name of the date field to pick the time series from", metavar="Date field")
    parser.add_argument("-p", "--predictor_fields", dest="predictors",
                        help="Predictor column indexes", metavar="Predictor columns")
    parser.add_argument("-y", "--target_fields", dest="targets",
                        help="Target column indexes", metavar="Target columns")
    parser.add_argument("-k", "--keyword_field", dest="k_field",
                        help="Keyword field name", metavar="Keyword")
    parser.add_argument("-dt", "--device_type_field", dest="d_field",
                        help="Device type field name", metavar="Device type")
    parser.add_argument("-md", "--models_dir", dest="models_dir",
                        help="Directory name to store models", metavar="Models directory")
    parser.add_argument("-rnn", "--rnn_type", dest="rnn_type",
                        help="Recurrent neural network model (LSTM, GRU)", metavar="RNN model type")
    args = parser.parse_args()
    main(args)
