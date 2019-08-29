import tensorflow as tf

MIN_INIT_VALUE = -0.5
MAX_INIT_VALUE = 0.5


class QNeuralModel(object):
    def __init__(self, n_input_units, n_hidden_units, n_output_units, step_size=0.01):
        self._g = tf.Graph()
        with self._g.as_default():
            with tf.variable_scope('input'):
                self._x = tf.placeholder(tf.float32, shape=(None, n_input_units), name="X")

            with tf.variable_scope('parameters'):
                self._action = tf.placeholder(tf.int32, shape=(None, 1), name="action")
                self._next_action_value = tf.placeholder(tf.float32, shape=(None, 1), name="next_action_value")
                self._reward = tf.placeholder(tf.float32, shape=(None, 1), name="reward")
                self._discount = tf.placeholder(tf.float32, shape=(None, 1), name="discount")

            self._t_q = self.define_model('train', n_hidden_units, n_output_units)
            self._t_q_episode = self.define_model('episode', n_hidden_units, n_output_units, trainable_flg=False)
            self._t_q_target = self.define_model('target', n_hidden_units, n_output_units, trainable_flg=False)

            with tf.variable_scope('cost'):
                target = self._reward + self._discount * self._next_action_value
                action_values = tf.diag_part(tf.squeeze(tf.gather(tf.transpose(self._t_q), self._action), axis=1))
                t_delta = tf.squeeze(target) - action_values
                self._q_loss = tf.reduce_mean(tf.square(t_delta))

            with tf.variable_scope('train'):
                self._t_optimizer = tf.train.RMSPropOptimizer(step_size).minimize(self._q_loss)

            self._session = tf.Session(graph=self._g)
            self._session.run(tf.global_variables_initializer())

            self._saver = tf.train.Saver()

    def define_model(self, namespace, hidden_units, output_units, trainable_flg=True):
        with tf.name_scope(namespace):
            # Hidden layers
            initializer = tf.random_uniform_initializer(minval=MIN_INIT_VALUE, maxval=MAX_INIT_VALUE)
            if not isinstance(hidden_units, list):
                with tf.variable_scope('{}_hidden_layer'.format(namespace)):
                    t_last_layer = tf.contrib.layers.fully_connected(self._x,
                                                                     hidden_units,
                                                                     weights_initializer=initializer,
                                                                     activation_fn=tf.nn.relu,
                                                                     trainable=trainable_flg)
            else:
                previous_layer = self._x
                for i, n_units in enumerate(hidden_units):
                    with tf.variable_scope('{}_hidden_layer_{}'.format(namespace, i)):
                        t_h_layer = tf.contrib.layers.fully_connected(previous_layer,
                                                                      n_units,
                                                                      weights_initializer=initializer,
                                                                      activation_fn=tf.nn.relu,
                                                                      trainable=trainable_flg)
                        previous_layer = t_h_layer
                t_last_layer = previous_layer

            # Output Layer
            with tf.variable_scope('{}_output'.format(namespace)):
                t_q = tf.contrib.layers.fully_connected(t_last_layer,
                                                        output_units,
                                                        weights_initializer=initializer,
                                                        activation_fn=tf.nn.relu,
                                                        trainable=trainable_flg)

        return t_q

    def restore(self, checkpoint):
        self._saver.restore(self._session, checkpoint)

    def save(self, checkpoint):
        self._saver.save(self._session, checkpoint)

    def copy_train_vars(self, to_model):
        train_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='train')
        to_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=to_model)
        assign_ops = []
        with self._g.as_default():
            for target_var in to_variables:
                target_var_name = target_var.name.replace('{}_'.format(to_model), '')
                for train_var in train_variables:
                    train_var_name = train_var.name.replace('train_', '')
                    if target_var_name == train_var_name:
                        assign_ops.append(tf.assign(target_var, tf.identity(train_var)))
            copy_operation = tf.group(*assign_ops)
            self._session.run(copy_operation)

    def run(self, features, model="target"):
        feed_dict = {self._x: features}
        return self._session.run([self._t_q_target if model == "target" else self._t_q_episode], feed_dict)

    def train(self, states, actions, next_action_values, rewards, discounts):
        feed_dict = {self._x: states,
                     self._action: actions,
                     self._next_action_value: next_action_values,
                     self._reward: rewards,
                     self._discount: discounts}
        _, loss = self._session.run([self._t_optimizer, self._q_loss], feed_dict)
        return loss
