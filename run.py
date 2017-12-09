import subprocess
import numpy as np 
import time
import re
import random

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model

class ActorCritic:
	def __init__(self, env):
		self.env = env
		self.learning_rate = 0.001
		self.epsilon = 1.0
		self.epsilon_decay = .995
		self.gamma = .95
		self.memory = None

		self.actor_model = self.create_actor_model()
		self.critic_model = self.create_critic_model()

	def create_actor_model(self):
		model = Sequential()
		model.add(Dense(13, activation='relu'))
		model.add(Dense(26, activation='relu'))
		model.add(Dense(13, activation='relu'))
		model.add(Dense(10, activation='relu'))

		adam = Adam(lr=self.learning_rate)
		model.compile(loss='mse', optimizer=adam)
		return model

	def create_critic_model(self):
		model = Sequential()
		model.add(Dense(23, activation='relu'))
		model.add(Dense(46, activation='relu'))
		model.add(Dense(23, activation='relu'))
		model.add(Dense(1, activation='relu'))

		adam = Adam(lr=self.learning_rate)
		model.compile(loss='mse', optimizer=adam)
		return model

	def train(self):
		batch_size = 200
		if len(self.memory) < batch_size:
			print "not enough data in memory for batch_size = 200"
			return
		samples = random.sample(self.memory, batch_size)
		self._train_critic(samples)
		self._train_actor(samples)

	def _train_critic(self, samples):
		x = samples[:, 0:23] # cur_state, action
		y = samples[:, 24] # new_reward
		self.critic_model.train_on_batch(x, y)

	def _train_actor(self, samples):
		x = samples[:, 0:13] # cur_state
		y = samples[:, 23] # reward
		self.actor_model.train_on_batch(x, y)

	def remember(self, cur_state, action, reward, new_reward):
		memory = np.concatenate((np.asarray(cur_state), np.asarray(action)))
		memory = np.concatenate((memory, [reward, new_reward]))
		memory = memory.reshape((1, 25))
		# memory = memory.reshape((1, memory.shape[0]))
		if self.memory is None:
			self.memory = memory
		else:
			self.memory = np.concatenate((self.memory, memory))

	def act(self, cur_state):
		self.epsilon *= self.epsilon_decay
		if np.random.random() < self.epsilon:
			return self.env.sample_action()
		return self.actor_model.predict(cur_state)

class Env:
	self.state_shape = (13, )
	self.action_shape = (10, )

	def __init__(self):
		pass # in other file

	def pause():
		pass

	def step(self, action):
		pass

def parse_state_str(str):
	pass

def standing_objective(model_state):
	loss = np.sum(np.square(model_state))
	done = True if loss > 1 else False
	return -loss, done

def sitting_objective(model_state):
	pass

def main():
	env = Env()
	env.pause()
	env.spawn_figure()

	actor_critic = ActorCritic(env, sess)

	num_trials = 1 #1000
	trial_len = 200
	for j in range(num_trials):
		cur_state = env.reset()
		for i in range(trial_len):
			action = actor_critic.act(cur_state)
			new_state, reward, done = env.step(action)
			if done:
				break

			target_action = actor_critic.actor_model.predict(new_state)
			new_reward = actor_critic.critic_model.predict(
				np.concatenate((new_state, target_action), axis=1))
			new_reward = reward + actor_critic.gamma * new_reward

			actor_critic.remember(cur_state, action, reward, new_reward)
			actor_critic.train()
			cur_state = new_state
	actor_critic.actor_model.save_weights("actor_model.h5")
	actor_critic.critic_model.save_weights("critic_model.h5")

def validate():
	actor_model = load_model("actor_model.h5")
	env = Env()
	env.pause()
	env.spawn_figure()

	num_trials = 1
	trial_len = 200
	for j in range(num_trials):
		cur_state = env.reset()
		rewards = []
		for i in range(trial_len):
			action = actor_model.predict(cur_state)
			new_state, reward, done = env.step(action)
			if done:
				break

			rewards.append(reward)
			cur_state = new_state
main()
# validate()