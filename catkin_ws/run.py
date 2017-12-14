import subprocess
import numpy as np 
import time
import re
import random
import sys
import os

from gazebo_msgs.srv import GetModelState, GetLinkState, ApplyJointEffort, JointRequest
from std_srvs.srv import Empty
import roslaunch
import rospy

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
#        self.critic_model.fit(x, y)

    def _train_actor(self, samples):
        x = samples[:, 0:13]
        x = np.concatenate((x, samples[:, 24]), axis=1) # cur_state value
        y = samples[:, 13:23] # action
        self.actor_model.train_on_batch(x, y)
#        self.actor_model.fit(x, y)

    def clear_memory(self):
        self.memory = None

    def remember(self, cur_state, action, reward, new_reward, value):
        memory = np.concatenate((cur_state, action), axis=1)
        memory = np.concatenate((memory, np.asarray([reward, new_reward, value]).reshape((1, 3))), axis=1)
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

        self.pause = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.unpause = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.reset = rospy.ServiceProxy('gazebo/reset_simulation', Empty)


    # def pause(self):
    #     call = rospy.ServiceProxy('gazebo/pause_physics', Empty)
    #     call()

    # def unpause(self):
    #     call = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
    #     call()

    # def reset(self):
    #     call = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
    #     call()
    #     return self.get_model_state()

    def get_model_state(self):
        call_get_state = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)
        output = call_get_state('my_robot', 'world')
        return parse_state(output)

    def get_link_state(self, link_name):
        call_get_state = rospy.ServiceProxy('gazebo/get_link_state', GetLinkState)
        output = call_get_state(link_name, 'world')
        return parse_state(output.link_state)

    def sample_action(self):
        return (np.random.random([self.action_space_shape[0]]) - 0.5) * 5

    def clear_joint_forces(self):
        call = rospy.ServiceProxy('gazebo/clear_joint_forces', JointRequest)
        for i in range(10):
            call({joint_name: self.joint_names[i]})
        # subprocess.check_output(["rosservice", "call", "gazebo/clear_joint_forces", 
        #     "{joint_name: %(joint_list)s}" % {'joint_list': self.joint_names}])

    def step(self, action):
        call = rospy.ServiceProxy('gazebo/apply_joint_effort', ApplyJointEffort)
        for i in range(len(action[0])):
            try: 
                call(self.joint_names[i], action[0][i], rospy.Duration.from_sec(0), rospy.Duration(0.1))
            except rospy.ServiceException, e:
                print e

        self.unpause()
        rospy.sleep(0.1)
        self.pause()
        self.clear_joint_forces
        model_state = self.get_model_state()
        reward, done = standing_objective(self)
        return model_state, reward, done

def parse_state(state):
    p = state.pose.position
    o = state.pose.orientation
    l = state.twist.linear
    a = state.twist.angular
    return np.asarray([p.x, p.y, p.z, o.x, o.y, o.z, o.w, l.x, l.y, l.z, a.x, a.y, a.z]).reshape((1, 13))

def standing_objective(env):
    head_state = env.get_link_state("link8")
    loss = (3.75 - head_state[0, 2])**2
    done = True if loss > 1 else False
    return -loss, done

def sitting_objective(model_state):
    pass

def main():
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    launch = roslaunch.parent.ROSLaunchParent(uuid, ["/home/julinas/catkin_ws/src/my_robot_control/launch/my_robot.launch"])
    launch.start()
    rospy.wait_for_service('gazebo/get_model_state')
    rospy.wait_for_service('gazebo/get_link_state')
    rospy.wait_for_service('gazebo/reset_simulation')
    rospy.wait_for_service('gazebo/pause_physics')
    rospy.wait_for_service('gazebo/unpause_physics')
    rospy.wait_for_service('gazebo/apply_joint_effort')
    env = Env()
    env.pause()

    AC = ActorCritic(env)

    num_trials = 2000
    trial_len = 600 #60 sec = 1 min
    for j in range(num_trials):
        env.reset()
        cur_state = env.get_model_state()
        AC.clear_memory()
        for i in range(trial_len):
            action = AC.act(cur_state)
            # print action
            new_state, reward, done = env.step(action)
            if done or i+1 == trial_len:
                print "trial", j, "lasted", i * 0.1
                break

            value = AC.critic_model.predict(np.concatenate((cur_state, action)), axis=1)

            target_action = AC.actor_model.predict(new_state).reshape((1, 10))
            new_reward = AC.critic_model.predict(
                np.concatenate((new_state, target_action), axis=1))
            new_reward = reward + AC.gamma * new_reward

            AC.remember(cur_state, action, reward, new_reward, value)
            AC.train(i)
            cur_state = new_state
    AC.actor_model.save_weights("actor_model.h5")
    AC.critic_model.save_weights("critic_model.h5")
    launch.shutdown()

def validate(): #needs fixing
    actor_model = load_model("actor_model.h5")
    env = Env()
    env.pause()
    # env.spawn_figure()

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
