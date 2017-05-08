import threading
import multiprocessing
import scipy.signal
import sys
from helper import *
from vizdoom import *
from time import sleep
from gym_torcs import TorcsEnv


# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


# Processes Doom screen image to produce cropped and resized image.
def process_frame(frame):
    # s = frame[10:-10, 30:-30] #Get the frame quadratic 120 - 20 and 160 - 60
    # s = scipy.misc.imresize(s, [84, 84]) #resize 100x100 to 84x84
    # s = np.reshape(s, [np.prod(s.shape)]) / 255.0 #make it one tuple size 7056 and makes values between 0-1 by dividing 255
    r, g, b = frame[:, 0], frame[:, 1], frame[:, 2]  # find r,g,b values
    img_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b  # make rgb to grayscale
    s = np.reshape(img_gray, [np.prod(img_gray.shape)]) / 255.0
    return s


# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


class AC_Network:
    def __init__(self, s_size, a_size, scope, trainer, continuous=False):
        with tf.variable_scope(scope):
            # Input and visual encoding layers
            self.inputs = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
            self.imageIn = tf.reshape(self.inputs, shape=[-1, 64, 64, 1])
            self.conv1 = slim.conv2d(activation_fn=tf.nn.elu,
                                     inputs=self.imageIn, num_outputs=16,
                                     kernel_size=[8, 8], stride=[4, 4], padding='VALID')
            self.conv2 = slim.conv2d(activation_fn=tf.nn.elu,
                                     inputs=self.conv1, num_outputs=32,
                                     kernel_size=[4, 4], stride=[2, 2], padding='VALID')
            hidden = slim.fully_connected(slim.flatten(self.conv2), 256, activation_fn=tf.nn.elu)

            # Recurrent network for temporal dependencies
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]  # initial state of the rnn
            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)
            rnn_in = tf.expand_dims(hidden, [0])
            step_size = tf.shape(self.imageIn)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 256])

            # Output layers for policy and value estimations
            if not continuous:
                self.discrete_policy = slim.fully_connected(rnn_out, a_size,
                                                            activation_fn=tf.nn.softmax,
                                                            weights_initializer=normalized_columns_initializer(0.01),
                                                            biases_initializer=None)
            else:
                self.variance = slim.fully_connected(rnn_out, 3,
                                                     activation_fn=None,
                                                     weights_initializer=normalized_columns_initializer(1.0),
                                                     biases_initializer=None)
                self.mean = slim.fully_connected(rnn_out, 3,
                                                 activation_fn=None,
                                                 weights_initializer=normalized_columns_initializer(1.0),
                                                 biases_initializer=None)
                self.variance = tf.squeeze(self.variance)
                self.variance = tf.nn.softplus(self.variance) + 1e-5
                self.normal_dist = tf.contrib.distributions.Normal(self.mean, self.variance)
                self.actions = self.normal_dist._sample_n(1)
                self.actions = [tf.clip_by_value(self.actions[0][0][0], -1, 1),
                                tf.clip_by_value(self.actions[0][0][1], 0, 1),
                                tf.clip_by_value(self.actions[0][0][2], 0, 1)]

            self.value = slim.fully_connected(rnn_out, 1,
                                              activation_fn=None,
                                              weights_initializer=normalized_columns_initializer(1.0),
                                              biases_initializer=None)

            # Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

                if not continuous:
                    self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
                    self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)

                    self.responsible_outputs = tf.reduce_sum(self.discrete_policy * self.actions_onehot, [1])

                    self.entropy_loss = - tf.reduce_sum(self.discrete_policy * tf.log(self.discrete_policy))
                    self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs) * self.advantages)
                    self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))

                    self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy_loss * 0.01
                else:
                    self.steer = tf.placeholder(shape=[None], dtype=tf.float32)
                    self.accelerate = tf.placeholder(shape=[None], dtype=tf.float32)
                    self.brake = tf.placeholder(shape=[None], dtype=tf.float32)

                    # self.logprob = self.normal_dist.log_prob(tf.stack([self.steer, self.accelerate, self.brake], axis=1))
                    #
                    # self.entropy = tf.reduce_sum(self.normal_dist.entropy())
                    # self.entropy_loss = - tf.reduce_sum(self.entropy)
                    #
                    # self.steering_policy_loss = - tf.reduce_sum((self.logprob[:, 0] + self.entropy) * self.advantages)
                    # self.accelerating_policy_loss = - tf.reduce_sum((self.logprob[:, 1] + self.entropy) * self.advantages)
                    # self.brake_policy_loss = - tf.reduce_sum((self.logprob[:, 2] + self.entropy) * self.advantages)
                    #
                    # self.policy_loss = (self.steering_policy_loss + self.accelerating_policy_loss + self.braking_policy_loss)
                    # self.value_loss = tf.reduce_sum(tf.squared_difference(self.target_v, self.value))
                    #
                    # self.loss = (self.policy_loss + 0.5 * self.value_loss + 1e-4 * self.entropy_loss)

                    epsilon = 1e-10
                    actions = tf.stack([self.steer, self.accelerate, self.brake], axis=1)
                    entropy = tf.reduce_sum(tf.log(self.variance + epsilon))
                    self.entropy_loss = -tf.reduce_sum(entropy)

                    squared_difference = tf.squared_difference(actions, self.mean)
                    squared_distance = tf.reduce_sum(squared_difference / (self.variance + epsilon), axis=1)

                    self.policy_loss = -tf.reduce_sum((entropy + squared_distance) * self.advantages)

                    self.value_loss = tf.reduce_sum(tf.squared_difference(self.target_v, self.value))

                    self.loss = (self.policy_loss + 0.5 * self.value_loss + 1e-4 * self.entropy_loss)

                # Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

                # Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))


class Worker:
    def __init__(self, game, name, s_size, a_size, trainer, model_path, global_episodes, continuous=False):
        self.continuous = continuous
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer_train = tf.summary.FileWriter("train_" + str(self.number))
        self.summary_writer_play = tf.summary.FileWriter("play_" + str(self.number))

        # Create the local copy of the network and the tensorflow op to copy global params to local network
        self.local_AC = AC_Network(s_size, a_size, self.name, trainer, continuous)
        self.update_local_ops = update_target_graph('global', self.name)
        self.env = game
        if not continuous:
            self.actions = np.identity(a_size, dtype=bool).tolist()    #To have same format as doom
            #self.actions = np.array([[-0.5, 0, 0], [0.5, 0, 0], [0, 1, 0], [0, 0, 1]])

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = np.asarray(rollout[:, 1].tolist())
        rewards = rollout[:, 2]
        next_observations = rollout[:, 3]
        values = rollout[:, 5]

        # We take the rewards and values from rollout, and use them to generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        rnn_state = self.local_AC.state_init
        if not self.continuous:
            feed_dict = {self.local_AC.target_v: discounted_rewards,
                         self.local_AC.inputs: np.vstack(observations),
                         self.local_AC.actions: actions,
                         self.local_AC.advantages: advantages,
                         self.local_AC.state_in[0]: rnn_state[0],
                         self.local_AC.state_in[1]: rnn_state[1]}
        else:
            feed_dict = {self.local_AC.target_v: discounted_rewards,
                         self.local_AC.inputs: np.vstack(observations),
                         self.local_AC.steer: actions[:, 0],
                         self.local_AC.accelerate: actions[:, 1],
                         self.local_AC.brake: actions[:, 2],
                         self.local_AC.advantages: advantages,
                         self.local_AC.state_in[0]: rnn_state[0],
                         self.local_AC.state_in[1]: rnn_state[1]}
        v_l, p_l, e_l, loss_f, g_n, v_n, _ = sess.run([self.local_AC.value_loss,
                                                       self.local_AC.policy_loss,
                                                       self.local_AC.entropy_loss,
                                                       self.local_AC.loss,
                                                       self.local_AC.grad_norms,
                                                       self.local_AC.var_norms,
                                                       self.local_AC.apply_grads],
                                                      feed_dict=feed_dict)
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), loss_f / len(rollout), g_n, v_n

    def work(self, max_episode_length, gamma, sess, coord, saver, training):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = 0
                episode_step_count = 0
                d = False

                if np.mod(episode_count, 20) == 0:
                    # Sometimes you need to relaunch TORCS because of the memory leak error
                    ob = self.env.reset(relaunch=True)
                    sleep(0.5)
                else:
                    ob = self.env.reset(relaunch=False)
                s = ob.img

                # For creating gifs
                img = np.ndarray((64, 64, 3))
                for z in range(3):
                    img[:, :, z] = 255 - s[:, z].reshape((64, 64))
                red, green, blue = img[:, :, 0], img[:, :, 1], img[:, :, 2]
                img_gray = 0.2989 * red + 0.5870 * green + 0.1140 * blue

                # For state
                episode_frames.append(img_gray)
                s = process_frame(s)
                rnn_state = self.local_AC.state_init

                while not d:
                    # Take an action using probabilities from policy network output.
                    if not self.continuous:
                        a_dist, v, rnn_state = sess.run(
                            [self.local_AC.discrete_policy, self.local_AC.value, self.local_AC.state_out],
                            feed_dict={self.local_AC.inputs: [s],
                                       self.local_AC.state_in[0]: rnn_state[0],
                                       self.local_AC.state_in[1]: rnn_state[1]})
                        a_t = np.random.choice(a_dist[0], p=a_dist[0])  # a random sample is generated given probabs
                        a_t = np.argmax(a_dist == a_t)
                        ob, reward, d, info = self.env.step_discrete(self.actions[a_t])
                        r = reward/100
                    else:
                        a_t, v, rnn_state = sess.run(
                            [self.local_AC.actions,
                             self.local_AC.value,
                             self.local_AC.state_out],
                            feed_dict={self.local_AC.inputs: [s],
                                       self.local_AC.state_in[0]: rnn_state[0],
                                       self.local_AC.state_in[1]: rnn_state[1]})

                        ob, r, d, info = self.env.step(a_t)

                    if not d:
                        s1 = ob.img

                        # For creating gifs
                        img1 = np.ndarray((64, 64, 3))
                        for z1 in range(3):
                            img1[:, :, z] = 255 - s1[:, z].reshape((64, 64))
                        red1, green1, blue1 = img1[:, :, 0], img1[:, :, 1], img1[:, :, 2]
                        img_gray1 = 0.2989 * red1 + 0.5870 * green1 + 0.1140 * blue1
                        episode_frames.append(img_gray1)

                        # for state
                        s1 = process_frame(s1)
                        # episode_frames.append(s1)
                    else:
                        s1 = s

                    episode_buffer.append([s, a_t, r, s1, d, v[0, 0]])
                    episode_values.append(v[0, 0])

                    episode_reward += r
                    s = s1
                    total_steps += 1
                    episode_step_count += 1

                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if training and len(episode_buffer) == 100 and d is not True:  #batch to 100
                        # Since we don't know what the true final return is, we "bootstrap" from our current
                        # value estimation
                        v1 = sess.run(self.local_AC.value,
                                      feed_dict={self.local_AC.inputs: [s],
                                                 self.local_AC.state_in[0]: rnn_state[0],
                                                 self.local_AC.state_in[1]: rnn_state[1]})[0, 0]
                        v_l, p_l, e_l, loss_f, g_n, v_n = self.train(episode_buffer, sess, gamma, v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)
                    if d:
                        break

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                if episode_reward > 10000:
                    time_per_step = 0.05
                    images = np.ndarray((64, 64))
                    images = np.array(episode_frames)
                    make_gif(images, './frames/r_image' + str(episode_reward) + '.gif',
                             duration=len(images) * time_per_step, true_image=True, salience=False)

                # Update the network using the experience buffer at the end of the episode.
                if training and len(episode_buffer) != 0:
                    v_l, p_l, e_l, loss_f, g_n, v_n = self.train(episode_buffer, sess, gamma, 0.0)

                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0:
                    if training and self.name == 'worker_0' and episode_count % 100 == 0:
                        time_per_step = 0.05
                        images = np.array(episode_frames)
                        make_gif(images, './frames/image' + str(episode_count) + '.gif',
                                 duration=len(images) * time_per_step, true_image=True, salience=False)
                    if training and episode_count % 10 == 0 and self.name == 'worker_0':
                        saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk')
                        print("Saved Model")

                    mean_reward = np.mean(self.episode_rewards[-5:])  # mean over the last 5 elements of episode Rs
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])

                    summary = tf.Summary()
                    summary.value.add(tag='Performance/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Performance/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Performance/Value', simple_value=float(mean_value))
                    if training:
                        summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                        summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                        summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                        summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                        summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                        summary.value.add(tag='Losses/Loss', simple_value=float(loss_f))
                        self.summary_writer_train.add_summary(summary, episode_count)
                        self.summary_writer_train.flush()
                    else:
                        self.summary_writer_play.add_summary(summary, episode_count)
                        self.summary_writer_play.flush()

                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1


def initialize_variables(saver, sess, load_model):
    if load_model:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())


def play_training(training=True, load_model=True):
    with tf.device("/cpu:0"):
        global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        trainer = tf.train.RMSPropOptimizer(learning_rate=1e-4, decay=0.99, epsilon=1)
        master_network = AC_Network(s_size, a_size, 'global', None, False)

        if training:
            #num_workers = multiprocessing.cpu_count()  # Set workers at number of available CPU threads
            num_workers = 4
        else:
            num_workers = 1

        workers = []
        for i in range(num_workers):
            workers.append(
                Worker(TorcsEnv(vision=True, throttle=True, gear_change=False, port=3101 + i), i, s_size, a_size,
                       trainer, model_path, global_episodes, False))
        saver = tf.train.Saver()

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        initialize_variables(saver, sess, load_model)
        # Asynchronous magic happens: start the "work" process for each worker in a separate thread.
        worker_threads = []
        for worker in workers:
            worker_work = lambda: worker.work(max_episode_length, gamma, sess, coord, saver, training)
            t = threading.Thread(target=worker_work)
            t.start()
            sleep(0.5)
            worker_threads.append(t)
        coord.join(worker_threads)  # waits until the specified threads have stopped.


if __name__ == "__main__":
    max_episode_length = 300
    gamma = .99  # discount rate for advantage estimation and reward discounting
    s_size = 4096  # Observations are greyscale frames of 84 * 84 * 1
    a_size = 4  # Left, Right, Forward, Brake
    model_path = './model'

    tf.reset_default_graph()

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Create a directory to save episode playback gifs to
    if not os.path.exists('./frames'):
        os.makedirs('./frames')

    if len(sys.argv) == 1:  # run from PyCharm
        play_training(training=True, load_model=False)
    elif sys.argv[1] == "1":  # lunch from Terminal and specify 0 or 1 as arguments
        play_training(training=True, load_model=False)
    elif sys.argv[1] == "0":
        play_training(training=False, load_model=True)
