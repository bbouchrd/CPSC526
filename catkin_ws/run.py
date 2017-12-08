import subprocess
import numpy as np
import time
import re
import random
from collections import deque

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
import keras.backend as K

import tensorflow as tf

class ActorCritic:
	def __init__(self, env, sess):
		self.env = env
		self.sess = sess

		self.learning_rate = 0.001
		self.epsilon = 1.0
		self.epsilon_decay = .995
		self.gamma = .95
		self.tau = .125
		self.memory = deque(maxlen=2000)

		self.actor_state_input, self.actor_model = self.create_actor_model()
		_, self.target_actor_model = self.create_actor_model()

		self.actor_critic_grad = tf.placeholder(tf.float32, 
			[None, self.env.action_space.shape[0]])

		actor_model_weights = self.actor_model.trainable_weights
		self.actor_grads = tf.gradients(self.actor_model.output,
			actor_model_weights, -self.actor_critic_grad)

		grads = zip(self.actor_grads, actor_model_weights)
		self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

	def create_actor_model(self):
		state_input = Input(shape=self.env.observation_space.shape)
		h1 = Dense(24, activation='relu')(state_input)
		h2 = Dense(48, activation='relu')(h1)
		h3 = Dense(24, activation='relu')(h2)
		output = Dense(self.env.action_space.shape[0], activation='relu')(h3)

		model = Model(input=state_input, output=output)
		adam = Adam(lr=0.001)
		model.compile(loss='mse', optimizer=adam)
		return state_input, model

	def create_critic_model(self):
		state_input = Input(shape=self.env.observation_space.shape)
		state_h1 = Dense(24, activation='relu')(state_input)
		state_h2 = Dense(48)(state_h1)

		action_input = Input(shape=self.env.action_space.shape)
		action_h1 = Dense(48)(action_input)

		merged = Add()([state_h2, action_h1])
		merged_h1 = Dense(24, activation='relu')(merged)
		output = Dense(1, activation='relu')(merged_h1)
		model = Model(input=[state_input, action_input], output=output)
		adam = Adam(lr=0.001)
		model.compile(loss='mse', optimizer=adam)
		return state_input, action_input, model

	def train(self):
		batch_size = 1
		if len(self.memory) < batch_size:
			return

		rewards = []
		samples = random.sample(self.memory, batch_size)
		self._train_critic(samples)
		self._train_actor(samples)

	def _train_critic(self, samples):
		for sample in samples:
			cur_state, action, reward, new_state, done = sample
			if not done:
				target_action = self.target_actor_model.predict(new_state)
				future_rewards = self.target_critic_model.predict(
					[new_state, target_action])[0][0]
				rewards += self.gamma * future_reward
			self.critic_model.fit([cur_state, action], reward, verbose=0)

	def remember(self, cur_state, action, reward, new_state, done):
		self.memory.append([cur_state, action, reward, new_state, done])

	def _train_actor(self, samples):
		for sample in samples:
			cur_state, action, reward, new_state, _ = sample
			predicted_action = self.actor_model.predict(cur_state)
			grads = self.sess.run(self.critic_grads, feed_dict={
				self.critic_state_input: cur_state,
				self.critic_action_input: predicted_action
			})[0]

			self.sess.run(self.optimize, feed_dict={
				self.actor_state_input: cur_state,
				self.actor_critic_grad: grads
			})

	def _update_actor_target(self):
		actor_model_weights = self.actor_model.get_weights()
		actor_target_weights = self.target_critic_model.get_weights()

		for i in range(len(actor_target_weights)):
			actor_target_weights[i] = actor_model_weights[i]
		self.target_critic_model.set_weights(actor_target_weights)

	def act(self, cur_state):
		self.epsilon *= self.epsilon_decay
		if np.random.random() < self.epsilon:
			return self.env.action_space.sample()
		return self.actor_model.predict(cur_state)

class Env:
	def __init__(self):
		pass

	def spawn_figure(self):
		subprocess.check_call(["rosrun", "gazebo_ros", "spawn_model", "-database", "my_robot", "-gazebo",
			"-model", "figure", "-y", "0", "-x", "0"])

	def pause(self):
		subprocess.call(["rosservice", "call", "gazebo/pause_physics"])

	def reset(self):
		subprocess.check_call(["rosservice", "call", "gazebo/"])

	def get_model_state(self):
		output = subprocess.check_output(["rosservice", "call", "gazebo/get_model_state", "figure", "world"])
		return parse_state_str(output)

	def get_link_state(self, link_name):
		output = subprocess.check_output(["rosservice", "call", "gazebo/get_link_state", link_name, "world"])
		return parse_state_str(output)


def parse_state_str(str):
	m = re.match("[\s\S]*x:\ ([\S]*)[\s\S]*y:\ ([\S]*)[\s\S]*z:\ ([\S]*)[\s\S]*x:\ ([\S]*)[\s\S]*y:\ ([\S]*)[\s\S]*z:\ ([\S]*)[\s\S]*w:\ ([\S]*)[\s\S]*x:\ ([\S]*)[\s\S]*y:\ ([\S]*)[\s\S]*z:\ ([\S]*)[\s\S]*x:\ ([\S]*)[\s\S]*y:\ ([\S]*)[\s\S]*z:\ ([\S]*)", str)
	if m:
		match = m.groups(1)
		return np.asarray(match)
		# return {
		# 	'pose': {
		# 		'position': {
		# 			'x': float(match[0]),
		# 			'y': float(match[1]),
		# 			'z': float(match[2])
		# 		},
		# 		'orientation': {
		# 			'x': float(match[3]),
		# 			'y': float(match[4]),
		# 			'z': float(match[5]),
		# 			'w': float(match[6])
		# 		}
		# 	},
		# 	'twist': {
		# 		'linear': {
		# 			'x': float(match[7]),
		# 			'y': float(match[8]),
		# 			'z': float(match[9])
		# 		},
		# 		'angular': {
		# 			'x': float(match[10]),
		# 			'y': float(match[11]),
		# 			'z': float(match[12])
		# 		}
		# 	}
		# }

def standing_loss(model_state):
	return np.sum(model_state)

def main():
	env = Env()
	env.pause()
	env.spawn_figure()
	print env.get_model_state()

	sess = tf.Session()
	K.set_session(sess)
	env = {
		observation_space: 0, #input shape
		action_space: 0 #output shape
	}
	actor_critic = ActorCritic(env, sess)

	num_trials = 10000
	trial_len = 500

	cur_state = env.reset() # set to default variables
	action = env.action_space.sample()
	while True:
		env.render() # update environ variables?
		cur_state = cur_state.reshape((1, env.observation_space.shape[0]))
		action = actor_critic.act(cur_state)
		action = action.reshape((1, env.action_space.shape[0]))

		new_state, reward, done, _ = env.step(action)
		new_state, = new_state.reshape((1, env.observation_space.shape[0]))

		actor_critic.remember(cur_state, action, reward, new_state, done)
		actor_critic.train()

		cur_state = new_state

main()


# child = subprocess.call(["nohup", "roslaunch", "my_robot_control", "my_robot.launch", "&"])
# child.send_signal(signal_SIGINT)

# import rospy
# import roslaunch


# print rospy.get_ros_root()
# rospy.init_node('en_Mapping', anonymous=True)
# # rospy.on_shutdown(self.shutdown)

# print rospy.get_rostime()

# uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
# roslaunch.configure_logging(uuid)
# launch = roslaunch.parent.ROSLaunchParent(uuid, ["/home/julinas/catkin_ws/src/my_robot_control/launch/my_robot.launch"])
# launch.start()

# launch.shutdown()

# rospy.init_node('my_node')
# print rospy.get_rostime()