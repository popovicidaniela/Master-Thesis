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


def setup_env(game):
    game.set_doom_scenario_path("basic.wad")  # This corresponds to the simple task we will pose our agent
    game.set_doom_map("map01")
    game.set_screen_resolution(ScreenResolution.RES_160X120)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_render_hud(False)
    game.set_render_crosshair(False)
    game.set_render_weapon(True)
    game.set_render_decals(False)
    game.set_render_particles(False)
    game.add_available_button(Button.MOVE_LEFT)
    game.add_available_button(Button.MOVE_RIGHT)
    game.add_available_button(Button.ATTACK)
    game.add_available_game_variable(GameVariable.AMMO2)
    game.add_available_game_variable(GameVariable.POSITION_X)
    game.add_available_game_variable(GameVariable.POSITION_Y)
    game.set_episode_timeout(300)
    game.set_episode_start_time(10)
    game.set_window_visible(False)
    game.set_sound_enabled(False)
    game.set_living_reward(-1)
    game.set_mode(Mode.PLAYER)
    game.init()
    return game


class AC_Network:
    def __init__(self, s_size, a_size, scope, trainer, continuous=False):
        with tf.variable_scope(scope):
            # Input and visual encoding layers
            self.inputs = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
            self.imageIn = tf.reshape(self.inputs, shape=[-1, 84, 84, 1])
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
                self.steering_variance = slim.fully_connected(rnn_out, 1,
                                                              activation_fn=tf.nn.softplus + 1e-4,
                                                              weights_initializer=tf.zeros_initializer,
                                                              biases_initializer=None)
                self.steering_mean = slim.fully_connected(rnn_out, 1,
                                                          activation_fn=None,
                                                          weights_initializer=tf.zeros_initializer,
                                                          biases_initializer=None)
                self.steering_normal_dist = tf.contrib.distributions.Normal(self.steering_mean, self.steering_variance)
                self.steer = self.steering_normal_dist._sample_n(1)
                self.steer = tf.clip_by_value(self.steer, -1, 1)
                self.braking_variance = slim.fully_connected(rnn_out, 1,
                                                             activation_fn=tf.nn.softplus + 1e-4,
                                                             weights_initializer=tf.zeros_initializer,
                                                             biases_initializer=None)
                self.braking_mean = slim.fully_connected(rnn_out, 1,
                                                         activation_fn=None,
                                                         weights_initializer=tf.zeros_initializer,
                                                         biases_initializer=None)
                self.braking_normal_dist = tf.contrib.distributions.Normal(self.braking_mean, self.braking_variance)
                self.brake = self.braking_normal_dist._sample_n(1)
                self.brake = tf.clip_by_value(self.steer, 0, 1)
                self.acceleration_variance = slim.fully_connected(rnn_out, 1,
                                                                  activation_fn=tf.nn.softplus + 1e-4,
                                                                  weights_initializer=tf.zeros_initializer,
                                                                  biases_initializer=None)
                self.acceleration_mean = slim.fully_connected(rnn_out, 1,
                                                              activation_fn=None,
                                                              weights_initializer=tf.zeros_initializer,
                                                              biases_initializer=None)
                self.acceleration_normal_dist = tf.contrib.distributions.Normal(self.acceleration_mean,
                                                                                self.acceleration_variance)
                self.accelerate = self.acceleration_normal_dist._sample_n(1)
                self.accelerate = tf.clip_by_value(self.accelerate, 0, 1)
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
                    self.entropy = - tf.reduce_sum(self.discrete_policy * tf.log(self.discrete_policy))
                    self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs) * self.advantages)
                else:
                    self.actions = tf.placeholder(shape=[None], dtype=tf.float32)
                    self.steer = self.actions[0][0]
                    self.brake = self.actions[0][2]
                    self.accelerate = self.actions[0][1]
                    self.steering_entropy = - tf.reduce_sum(self.steering_normal_dist.entropy())
                    self.braking_entropy = - tf.reduce_sum(self.braking_normal_dist.entropy())
                    self.acceleration_entropy = - tf.reduce_sum(self.acceleration_normal_dist.entropy())
                    self.steering_policy_loss = -tf.reduce_sum(
                        self.steering_normal_dist.log_prob(self.steer) * self.advantages)
                    self.braking_policy_loss = -tf.reduce_sum(
                        self.braking_normal_dist.log_prob(self.steer) * self.advantages)
                    self.acceleration_policy_loss = -tf.reduce_sum(
                        self.acceleration_normal_dist.log_prob(self.steer) * self.advantages)
                    self.policy_loss = self.steering_policy_loss + self.braking_policy_loss + self.acceleration_policy_loss
                    self.entropy = self.steering_entropy + self.braking_entropy + self.acceleration_entropy

                # Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

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
        self.local_AC = AC_Network(s_size, a_size, self.name, trainer)
        self.update_local_ops = update_target_graph('global', self.name)
        # Doom setup
        self.env = game
        #self.actions = np.identity(a_size, dtype=bool).tolist()
        #actions for torcs
        self.actions = self.actions = np.array([[0.5, 0.18], [0, 0.18]])  # for action turn a bit right or a bit left
        # End Doom setup

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:, 0]  # each row in the 0th column, outputs array([...])
        actions = rollout[:, 1]
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
        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.inputs: np.vstack(observations),
                     self.local_AC.actions: actions,
                     self.local_AC.advantages: advantages,
                     self.local_AC.state_in[0]: rnn_state[0],
                     self.local_AC.state_in[1]: rnn_state[1]}
        v_l, p_l, e_l, loss_f, g_n, v_n, _ = sess.run([self.local_AC.value_loss,
                                                       self.local_AC.policy_loss,
                                                       self.local_AC.entropy,
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

                # self.env.new_episode()
                if np.mod(episode_count, 10) == 0:
                    # Sometimes you need to relaunch TORCS because of the memory leak error
                    ob = self.env.reset(relaunch=True)
                else:
                    ob = self.env.reset(relaunch=False)

                #changing to torcs
                # s = self.env.get_state().screen_buffer
                s = ob.img

                # For creating gifs
                img = np.ndarray((64, 64, 3))
                for z in range(3):
                    img[:, :, z] = 255 - s[:, z].reshape((64, 64))
                r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
                img_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

                # For state
                episode_frames.append(img_gray)
                s = process_frame(s)
                # episode_frames.append(s)
                rnn_state = self.local_AC.state_init


                a_t = np.zeros([1, a_size])

                #while not self.env.is_episode_finished():
                while d == False:
                    # Take an action using probabilities from policy network output.
                    if not self.continuous:
                        a_dist, v, rnn_state = sess.run(
                            [self.local_AC.discrete_policy, self.local_AC.value, self.local_AC.state_out],
                            feed_dict={self.local_AC.inputs: [s],
                                       self.local_AC.state_in[0]: rnn_state[0],
                                       self.local_AC.state_in[1]: rnn_state[1]})
                        a_t = np.random.choice(a_dist[0], p=a_dist[0])  # a random sample is generated given the probabs
                        a_t = np.argmax(a_dist == a_t)
                        # make action takes a list of button states and returns a reward - the result of making action:
                        ob, r, d, info = self.env.step(self.actions[a_t])  # ob = observations, r = reward, d = done (episode finish, info = ?
                    else:
                        steer, brake, accelerate, v, rnn_state = sess.run(
                            [self.local_AC.steer, self.local_AC.brake, self.local_AC.accelerate, self.local_AC.value, self.local_AC.state_out],
                            feed_dict={self.local_AC.inputs: [s],
                                       self.local_AC.state_in[0]: rnn_state[0],
                                       self.local_AC.state_in[1]: rnn_state[1]})
                        a_t[0][0] = steer
                        a_t[0][1] = accelerate
                        a_t[0][2] = brake
                        ob, r, d, info = self.env.step(self.actions[a])  # ob = observations, r = reward, d = done (episode finish, info = ?

                    #is_episode_finished = self.env.is_episode_finished()
                    #if not is_episode_finished:
                    if d == False:
                        # s1 = self.env.get_state().screen_buffer
                        s1 = ob.img  # Changed for torcs

                        # For creating gifs
                        img1 = np.ndarray((64, 64, 3))
                        for z1 in range(3):
                            img1[:, :, z] = 255 - s1[:, z].reshape((64, 64))
                        r1, g1, b1 = img1[:, :, 0], img1[:, :, 1], img1[:, :, 2]
                        img_gray1 = 0.2989 * r1 + 0.5870 * g1 + 0.1140 * b1
                        episode_frames.append(img_gray1)

                        # for state
                        s1 = process_frame(s1)
                        # episode_frames.append(s1)
                    else:
                        s1 = s

                    episode_buffer.append([s, a_t, r, s1, is_episode_finished, v[0, 0]])
                    episode_values.append(v[0, 0])

                    episode_reward += r
                    s = s1
                    total_steps += 1
                    episode_step_count += 1

                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if training and len(
                            episode_buffer) == 30 and is_episode_finished is not True and episode_step_count != max_episode_length - 1:
                        # Since we don't know what the true final return is, we "bootstrap" from our current
                        # value estimation
                        v1 = sess.run(self.local_AC.value,
                                      feed_dict={self.local_AC.inputs: [s],
                                                 self.local_AC.state_in[0]: rnn_state[0],
                                                 self.local_AC.state_in[1]: rnn_state[1]})[0, 0]
                        v_l, p_l, e_l, loss_f, g_n, v_n = self.train(episode_buffer, sess, gamma, v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)
                    if is_episode_finished:
                        break

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                # Update the network using the experience buffer at the end of the episode.
                if training and len(episode_buffer) != 0:
                    v_l, p_l, e_l, loss_f, g_n, v_n = self.train(episode_buffer, sess, gamma, 0.0)

                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0:
                    if training and self.name == 'worker_0' and episode_count % 25 == 0:
                        time_per_step = 0.05
                        images = np.array(episode_frames)
                        make_gif(images, './frames/image' + str(episode_count) + '.gif',
                                 duration=len(images) * time_per_step, true_image=True, salience=False)
                    if training and episode_count % 250 == 0 and self.name == 'worker_0':
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
        trainer = tf.train.RMSPropOptimizer(learning_rate=1e-4)  # trainer for the Workers
        master_network = AC_Network(s_size, a_size, 'global', None)  # Generate global network with trainer None

        if training:
            num_workers = multiprocessing.cpu_count()  # Set workers at number of available CPU threads
        else:
            num_workers = 1

        workers = []
        for i in range(num_workers):  # Create worker classes
            workers.append(Worker(Worker(TorcsEnv(vision=True, throttle=True, gear_change=False, port=3101+i), i, s_size, a_size, trainer, model_path, global_episodes))
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
            sleep(1.0)
            worker_threads.append(t)
        coord.join(worker_threads)  # waits until the specified threads have stopped.


if __name__ == "__main__":
    max_episode_length = 300
    gamma = .99  # discount rate for advantage estimation and reward discounting
    s_size = 7056  # Observations are greyscale frames of 84 * 84 * 1
    a_size = 3  # Agent can move Left, Right, or Fire
    model_path = './model'

    tf.reset_default_graph()

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Create a directory to save episode playback gifs to
    if not os.path.exists('./frames'):
        os.makedirs('./frames')

    if len(sys.argv) == 1:  # run in PyCharm, adjust True/ False
        play_training(training=False, load_model=True)
    elif sys.argv[1] == "1":  # lunch in Terminal and specify 0 or 1 as arguments
        play_training(training=True, load_model=True)
    elif sys.argv[1] == "0":
        play_training(training=False, load_model=True)
