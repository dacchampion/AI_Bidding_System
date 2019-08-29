import numpy as np
import tensorflow as tf

MIN_INIT_VALUE = -0.5
MAX_INIT_VALUE = 0.5
EPSILON = 0.1


class DQNAgent(object):
    def __init__(self, number_of_features, n_hidden_units, number_of_actions, initial_state, num_offline_updates=30,
                 step_size=0.01):
        # Variables declaration for neural network
        self._x = tf.placeholder(tf.float32, shape=(None, number_of_features))
        initializer = tf.random_uniform_initializer(minval=MIN_INIT_VALUE, maxval=MAX_INIT_VALUE)
        if not isinstance(n_hidden_units, list):
            t_last_layer = tf.contrib.layers.fully_connected(self._x,
                                                             n_hidden_units,
                                                             weights_initializer=initializer,
                                                             activation_fn=tf.nn.relu)
        else:
            previous_layer = self._x
            for n_units in n_hidden_units:
                t_h_layer = tf.contrib.layers.fully_connected(previous_layer,
                                                              n_units,
                                                              weights_initializer=initializer,
                                                              activation_fn=tf.nn.relu)
                previous_layer = t_h_layer
            t_last_layer = previous_layer
        self._t_q = tf.contrib.layers.fully_connected(t_last_layer,
                                                      number_of_actions,
                                                      weights_initializer=initializer,
                                                      activation_fn=None)
        # Loss calculation
        self._action = tf.placeholder(tf.int32, shape=(None, 1))
        self._next_action_value = tf.placeholder(tf.float32, shape=(None, 1))
        self._reward = tf.placeholder(tf.float32, shape=(None, 1))
        self._discount = tf.placeholder(tf.float32, shape=(None, 1))
        target = self._reward + self._discount * self._next_action_value
        action_values = tf.diag_part(tf.squeeze(tf.gather(tf.transpose(self._t_q), self._action), axis=1))
        t_delta = tf.squeeze(target) - action_values
        self._q_loss = tf.reduce_mean(tf.square(t_delta))
        # Gradient calculation
        self._t_optimizer = tf.train.RMSPropOptimizer(step_size).minimize(self._q_loss)
        # Session declaration and initialization
        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())

        # Instance variables
        self._num_actions = number_of_actions
        self._state = initial_state
        self._n = num_offline_updates
        self._optimal_value = 0
        self._losses = []
        self._replay_buffer = []

    def epsilon_greedy(self, q_values):
        return [np.argmax(q_values[x]) if EPSILON < np.random.random() else np.random.randint(self._num_actions) for x in range(q_values.shape[0])]

    def q(self, obs):
        feed_dict = {self._x: obs}
        q = self._sess.run([self._t_q], feed_dict)
        q = np.array(q)
        return q[0, :, :]

    def step(self, r, g, s):
        next_q = self.q(s)
        next_action = self.epsilon_greedy(next_q)
        next_action = next_action[0]
        # Store transition in replay buffer
        self._replay_buffer.append((s, next_action, r, g, s))

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

        next_qs = self.q(new_states)
        next_actions = self.epsilon_greedy(next_qs)
        # Offline and batch Q-Learning
        feed_dict = {self._x: old_states,
                     self._action: actions,
                     self._next_action_value: np.array([next_qs[i, j] for i, j in enumerate(next_actions)]).reshape(batch_size, 1),
                     self._reward: rewards,
                     self._discount: discounts}
        _, q, loss = self._sess.run([self._t_optimizer, self._t_q, self._q_loss], feed_dict)
        self._losses.append(loss)

        self._state = s
        self._optimal_value = next_q[0, next_action]
        return next_action
