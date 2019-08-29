import numpy as np
from keras.models import load_model
from DRL_Agent.rnn_utils import convert_to_timeseries


class BiddingSimulator(object):
    def __init__(self, context_df, action_column, predictor_columns, target_columns, model_file_name,
                 discount=0.9, explore_factor=0.5, episode_length=7):
        self._context = context_df
        _, self._scaler = convert_to_timeseries(context_df, [8], list(range(0, 8)))
        self._t = 0
        self._state = self._context.iloc[[self._t]].copy()
        self._predictor_cols = predictor_columns
        self._predictor_idxs = [self._state.columns.get_loc(x) for x in self._predictor_cols]
        self._target_cols = target_columns
        self._target_idxs = [self._state.columns.get_loc(x) for x in self._target_cols]
        self._discount = discount
        self._model = load_model(model_file_name)
        self._episode_end = episode_length
        self._t_episode = 0
        # Actions
        self._action_col = action_column
        min_bid = self._context[self._action_col].min()
        max_bid = self._context[self._action_col].max()
        step = ((1+explore_factor) * max_bid - explore_factor * min_bid) / 100
        self._actions = np.arange(explore_factor * min_bid, (1+explore_factor)*max_bid, step)

    def get_observation(self):
        temp_state = np.zeros((1, 8))
        temp_state[0, 0] = np.mean([self._scaler.data_max_[0], self._scaler.data_min_[0]])
        temp_state[0, 1] = np.mean([self._scaler.data_max_[1], self._scaler.data_min_[1]])
        temp_state[0, 2] = np.mean([self._scaler.data_max_[2], self._scaler.data_min_[2]]) / temp_state[0, 0]
        temp_state[0, 3] = temp_state[0, 0] / np.mean([self._scaler.data_max_[6], self._scaler.data_min_[6]])
        temp_state[0, 4] = np.mean([self._scaler.data_max_[3], self._scaler.data_min_[3]])
        temp_state[0, 5] = np.mean([self._scaler.data_max_[5], self._scaler.data_min_[5]])
        temp_state[0, 6] = np.mean([self._scaler.data_max_[6], self._scaler.data_min_[6]])
        temp_state[0, 7] = np.mean([self._scaler.data_max_[7], self._scaler.data_min_[7]])
        # temp_state[0, 8] = np.mean([self._scaler.data_max_[4], self._scaler.data_min_[4]])

        return temp_state

    def step(self, bidding_action):
        self._state.ix[0, self._action_col] = self._actions[bidding_action]

        # TODO: It needs decoupling the state according to the model
        scaled_state = self._scaler.fit_transform(self._state.values.reshape(-1, 1))
        scaled_input = scaled_state[self._predictor_idxs]
        scaled_input = scaled_input.reshape((1, 1, len(self._predictor_idxs)))
        prediction = self._model.predict(scaled_input)

        inv_data = np.zeros((self._state.shape[0], self._state.shape[1]))
        scaled_input = scaled_input.reshape((scaled_input.shape[0], scaled_input.shape[2]))
        inv_data[0, self._predictor_idxs] = scaled_input
        inv_data[0, self._target_idxs] = prediction
        inv_pred = self._scaler.inverse_transform(inv_data.reshape(-1, 1))

        self._state[self._target_cols] = inv_pred.T[:, self._target_idxs]

        # TODO: Calculate the reward according to the new state, it may need decoupling
        sessions = inv_pred[0, 0]
        cpc = inv_pred[1, 0]
        reattr_margin = inv_pred[4, 0]
        reward = reattr_margin - sessions * cpc

        discount = self._discount
        self._t_episode += 1
        if self._t_episode == self._episode_end:
            self._t_episode = 0
            discount = 0

        self._t += 1
        if self._t == self._context.shape[0]:
            self._t = 0
            self._t_episode = 0
            discount = 0

        for i, column in enumerate(self._target_cols):
            self._state.ix[0, column] = prediction[0, i]

        new_state = np.zeros((1, 8))
        new_state[0, 0] = sessions
        new_state[0, 1] = cpc
        new_state[0, 2] = inv_pred[2, 0] / sessions
        new_state[0, 3] = sessions / inv_pred[6, 0]
        new_state[0, 4] = inv_pred[3, 0]
        new_state[0, 5] = inv_pred[5, 0]
        new_state[0, 6] = inv_pred[6, 0]
        new_state[0, 7] = inv_pred[7, 0]
        # new_state[0, 8] = reattr_margin

        return reward, discount, new_state
