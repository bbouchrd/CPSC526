import subprocess
import numpy as np 
import time
import re
import random
import sys
import os

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
        model.add(Dense(13, activation='relu', input_shape=(13, )))
        model.add(Dense(26, activation='relu'))
        model.add(Dense(13, activation='relu'))
        model.add(Dense(10, activation='relu'))

        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)
        return model

    def create_critic_model(self):
        model = Sequential()
        model.add(Dense(23, activation='relu', input_shape=(23, )))
        model.add(Dense(46, activation='relu'))
        model.add(Dense(23, activation='relu'))
        model.add(Dense(1, activation='relu'))

        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)
        return model

    def train(self, batch_size):
        # samples = None
        # # if len(self.memory) < batch_size:
        # #     samples = self.memory
        # # else:
        # indices = np.random.randint(0, len(self.memory))
        # samples = self.memory[indices]
        samples = self.memory
        self._train_critic(samples)
        self._train_actor(samples)

    def _train_critic(self, samples):
        x = samples[:, 0:23] # cur_state, action
        y = samples[:, 24] # new_reward
        self.critic_model.train_on_batch(x, y)

    def _train_actor(self, samples):
        x = samples[:, 0:13] # cur_state
        y = samples[:, 13:23] # action
        self.actor_model.train_on_batch(x, y)

    def clear_memory(self):
        self.memory = None

    def remember(self, cur_state, action, reward, new_reward):
        memory = np.concatenate((cur_state, action), axis=1)
        memory = np.concatenate((memory, np.asarray([reward, new_reward]).reshape((1, 2))), axis=1)
        # memory = memory.reshape((1, 25))
        # memory = memory.reshape((1, memory.shape[0]))
        if self.memory is None:
            self.memory = memory
        else:
            self.memory = np.concatenate((self.memory, memory))

    def act(self, cur_state):
        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            return self.env.sample_action().reshape((1, 10))
        return self.actor_model.predict(cur_state).reshape((1, 10))

class Env:
    def __init__(self):
        self.joint_names = ['joint_5l6l', 'joint_5r6r', 'joint_45l', 'joint_45r', 'joint_01l', 'joint_01r', 'joint_1l2l', 'joint_1r2r', 'joint_2l9l', 'joint_2r9r']
        self.action_space_shape = [len(self.joint_names)]
        self.state_shape = (13, )
        self.action_shape = (10, )

    def spawn_figure(self):
        subprocess.check_call(["rosrun", "gazebo_ros", "spawn_model", "-database", "my_robot", "-gazebo",
            "-model", "figure", "-y", "0", "-x", "0"])

    def pause(self):
        subprocess.call(["rosservice", "call", "gazebo/pause_physics"])

    def unpause(self):
        subprocess.call(["rosservice", "call", "gazebo/unpause_physics"])

    def reset(self):
        subprocess.check_call(["rosservice", "call", "gazebo/reset_simulation"])
        return self.get_model_state()

    def get_model_state(self):
        output = subprocess.check_output(["rosservice", "call", "gazebo/get_model_state", "figure", "world"])
        return parse_state_str(output, True)

    def get_link_state(self, link_name):
        output = subprocess.check_output(["rosservice", "call", "gazebo/get_link_state", link_name, "world"])
        return parse_state_str(output, False)

    def sample_action(self):
        return (np.random.random([self.action_space_shape[0]]) - 0.5) * 20

    def clear_joint_forces(self):
        subprocess.checkout_output(["rosservice", "call", "gazebo/clear_joint_forces", 
            "{joint_name: %(joint_list)s}" % {'joint_list': joint_names}])

    def step(self, action):
        with open(os.devnull, 'w') as f:
            for i in range(len(action[0])):
            # print "trying", i
                subprocess.call(["rosservice", "call", "gazebo/apply_joint_effort", """joint_name: '%(joint_name)s'
effort: %(effort)d
start_time:
    secs: 0
    nsecs: 0
duration:
    secs: 0
    nsecs: 100000000""" % {'joint_name': self.joint_names[i], 'effort': action[0][i]}], stdout=f)
        self.unpause()
        time.sleep(0.1)
        self.pause()
        # print "paused again"
        model_state = self.get_model_state()
        # print "got model state"
        # standing
        reward, done = standing_objective(self)
        # print "calculated objective"
        return model_state, reward, done

def parse_state_str(str, isModel):
    l = str.split()
    match = np.asarray(l)
    if isModel:
        match = match[[13, 15, 17, 20, 22, 24, 26, 30, 32, 34, 37, 39, 41]]
    else:
        match = match[[6, 8, 10, 13, 15, 17, 19, 23, 25, 27, 30, 32, 34]]
    # print match
    # print match.shape
    return match.astype(np.float).flatten().reshape((1, 13))
        # match = m.groups(1)
        # print match
        # sys.stdout.flush()
        # return np.asarray(match).astype(np.float)
        # return {
        #   'pose': {
        #       'position': {
        #           'x': float(match[0]),
        #           'y': float(match[1]),
        #           'z': float(match[2])
        #       },
        #       'orientation': {
        #           'x': float(match[3]),
        #           'y': float(match[4]),
        #           'z': float(match[5]),
        #           'w': float(match[6])
        #       }
        #   },
        #   'twist': {
        #       'linear': {
        #           'x': float(match[7]),
        #           'y': float(match[8]),
        #           'z': float(match[9])
        #       },
        #       'angular': {
        #           'x': float(match[10]),
        #           'y': float(match[11]),
        #           'z': float(match[12])
        #       }
        #   }
        # }

def standing_objective(env):
    head_state = env.get_link_state("link8")
    # print head_state
    loss = 3.75 - head_state[0, 2]
    # loss = np.sum(np.square(model_state))
    done = True if loss > 1 else False
    return -loss, done

def sitting_objective(model_state):
    pass

def main():
    env = Env()
    env.pause()
    env.spawn_figure()

    AC = ActorCritic(env)

    num_trials = 20 #1000
    trial_len = 200
    for j in range(num_trials):
        cur_state = env.reset()
        AC.clear_memory()
        for i in range(trial_len):
            action = AC.act(cur_state)
            print i
            new_state, reward, done = env.step(action)
            if done:
                print "done"
                print env.get_model_state()
                break

            target_action = AC.actor_model.predict(new_state).reshape((1, 10))
            new_reward = AC.critic_model.predict(
                np.concatenate((new_state, target_action), axis=1))
            new_reward = reward + AC.gamma * new_reward

            AC.remember(cur_state, action, reward, new_reward)
            AC.train(i)
            cur_state = new_state
    AC.actor_model.save_weights("actor_model.h5")
    AC.critic_model.save_weights("critic_model.h5")

def validate():
    actor_model = load_model("actor_model.h5")
    env = Env()
    env.pause()
    env.spawn_figure()

    num_trials = 10
    trial_len = 200
    for j in range(num_trials):
        cur_state = env.reset()
        rewards = []
        for i in range(trial_len):
            action = actor_model.predict(cur_state).reshape((1, 10))
            new_state, reward, done = env.step(action)
            if done:
                print "done"
                break

            rewards.append(reward)
            cur_state = new_state
        print "rewards for trial", j, rewards
main()
# validate()