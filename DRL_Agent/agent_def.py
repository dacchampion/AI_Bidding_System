import numpy as np

from DRL_Agent.model_def import QNeuralModel

EPSILON = 0.1


class DeepLearningAgent(object):
    def __init__(self, number_of_features, number_of_actions, initial_state, steps_update, batch_size):
        self.q_model = QNeuralModel(number_of_features, [150, 120], number_of_actions)
        # Instance variables
        self._num_actions = number_of_actions
        self._state = initial_state
        self._steps_update = steps_update
        self._n = batch_size
        self._optimal_value = 0
        self._losses = []
        self._replay_buffer = []
        self._step = 0

    def initial_action(self):
        return np.random.randint(self._num_actions)

    def epsilon_greedy(self, q_values):
        return [np.argmax(q_values[x]) if EPSILON < np.random.random() else np.random.randint(self._num_actions) for x
                in range(q_values.shape[0])]

    def q(self, obs, model="episode"):
        q = self.q_model.run(obs, model)
        q = np.array(q)
        return q[0, :, :]

    def step(self, r, g, s):
        next_q = self.q(s, model="episode")
        next_action = self.epsilon_greedy(next_q)
        next_action = next_action[0]
        # Store transition in replay buffer
        self._replay_buffer.append((self._state, next_action, r, g, s))
        # Update Q action-values n-times
        batch_size = self._n if len(self._replay_buffer) >= self._n else len(self._replay_buffer)
        old_states = np.zeros((batch_size, s.shape[1]))
        actions = np.zeros((batch_size, 1))
        rewards = np.zeros((batch_size, 1))
        discounts = np.zeros((batch_size, 1))
        new_states = np.zeros((batch_size, s.shape[1]))

        i = 0
        for idx in np.random.randint(len(self._replay_buffer), size=batch_size):
            s, a, r, d, s_ = self._replay_buffer[idx]
            old_states[i] = s
            actions[i] = a
            rewards[i] = r
            discounts[i] = d
            new_states[i] = s_
            i += 1

        next_qs = self.q(new_states, model="target")
        # Offline and batch Q-Learning
        loss = self.q_model.train(old_states,
                                  actions,
                                  np.array(np.max(next_qs, axis=1)).reshape(batch_size, 1),
                                  rewards,
                                  discounts)
        self._losses.append(loss)

        if g == 0:
            self.q_model.copy_train_vars(to_model="episode")

        if self._step == self._steps_update:
            self.q_model.copy_train_vars(to_model="target")
            self._step = 0

        self._state = s
        self._optimal_value = next_q[0, next_action]
        self._step += 1
        return next_action
