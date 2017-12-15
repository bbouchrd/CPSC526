import subprocess
import numpy as np 
import time
import random
import math
import sys
import os

from gazebo_msgs.srv import GetModelState, GetLinkState, ApplyJointEffort, JointRequest
from std_srvs.srv import Empty
import roslaunch
import rospy

from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, ThresholdedReLU, Lambda
from keras.optimizers import Adam
from keras.models import load_model

class ActorCritic:
    def __init__(self, env):
        self.env = env
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = .995
        self.gamma = .95
        # self.memory = None
        self.cur_state = None
        self.cur_target = None
        self.cur_advantages = None

        self.adam = Adam(lr=self.learning_rate)

        self.leaky_alpha = 0.0000001

        self.actor_model = self.create_actor_model()
        self.critic_model = self.create_critic_model()

    def create_actor_model(self): #input: state; output: reward * action val
        model = Sequential()
        # model.add(Dense(13, activation='relu', input_shape=(156, )))
        model.add(Dense(13, input_shape=(156, )))
        model.add(LeakyReLU(alpha=self.leaky_alpha))
        model.add(Dense(26, activation='relu'))
        model.add(LeakyReLU(alpha=self.leaky_alpha))
        model.add(Dense(13, activation='relu'))
        model.add(LeakyReLU(alpha=self.leaky_alpha))
        model.add(Dense(10, activation='linear'))

        model.compile(loss='mse', optimizer=self.adam)
        return model

    def create_critic_model(self): #input: state; output: loss+gamma*new loss corr w/ value 
        model = Sequential()
        model.add(Dense(13, activation='relu', input_shape=(156, )))
        model.add(LeakyReLU(alpha=self.leaky_alpha))
        model.add(Dense(26, activation='relu'))
        model.add(LeakyReLU(alpha=self.leaky_alpha))
        model.add(Dense(13, activation='relu'))
        model.add(LeakyReLU(alpha=self.leaky_alpha))
        model.add(Dense(1, activation='relu'))
        model.add(ThresholdedReLU(theta=0.0))
        model.add(Lambda(lambda x: x - 1.0))
        model.compile(loss='mse', optimizer=self.adam)
        return model

    def train(self, batch_size):
        self._train_critic()
        self._train_actor()

    def _train_critic(self):
        x = self.cur_state
        y = self.advantages
        self.critic_model.train_on_batch(x, y)

    def _train_actor(self):
        x = self.cur_state
        y = self.target
        self.actor_model.train_on_batch(x, y)

    def clear_memory(self):
        self.memory = None

    def remember(self, cur_state, action, loss, new_state, done):
        value = self.critic_model.predict(cur_state)
        new_value = self.critic_model.predict(new_state)

        if not done:
            loss += self.gamma * new_value

        self.cur_state = cur_state
        self.advantages = np.asarray(loss).reshape((1, 1))
        self.target = action * (-value)

    def act(self, cur_state):
        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            return self.env.sample_action().reshape((1, 10))
        policy = self.actor_model.predict(cur_state, batch_size=1)

        return np.asarray([np.random.normal(loc=policy[x], scale=0.1) for x in range(len(policy))]).reshape((1, 10))

        # [np.random.choice(for el in policy]
        # return np.random.choice(self.action)

class Env:
    def __init__(self):
        self.joint_names = ['joint_5l6l', 'joint_5r6r', 'joint_45l', 'joint_45r', 'joint_01l', 'joint_01r', 'joint_1l2l', 'joint_1r2r', 'joint_2l9l', 'joint_2r9r']
        self.action_space_shape = [len(self.joint_names)]
        self.state_shape = (13, )
        self.action_shape = (10, )

        # head, left and right, lower and upper arm, trunk, left and right, upper and lower leg, left and right foot
        self.important_links = ['link8', 'link6l', 'link6r', 'link5l', 'link5r', 'link0', 'link1l', 'link1r', 'link2l', 'link2r', 'link9l', 'link9r']

        self.pause = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.unpause = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.reset = rospy.ServiceProxy('gazebo/reset_simulation', Empty)

    def get_model_state(self):
        call_get_state = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)
        output = call_get_state('my_robot', 'world')
        return parse_state(output)

    def get_link_state(self, link_name):
        call_get_state = rospy.ServiceProxy('gazebo/get_link_state', GetLinkState)
        output = call_get_state(link_name, 'world')
        return parse_state(output.link_state)

    def get_state(self):
        # use all links instead of com
        state_list = []

        for link in self.important_links:
            link_state = self.get_link_state(link)
            state_list.append(link_state)

        states = np.concatenate(state_list).reshape(1, 156)
        return states

    def sample_action(self):
        sampled = (np.random.random([self.action_space_shape[0]]) - 0.5) * 20

        # feet
        sampled[8] *= 0.1 
        sampled[9] *= 0.1
        return sampled

    def clear_joint_forces(self):
        call = rospy.ServiceProxy('gazebo/clear_joint_forces', JointRequest)
        for i in range(10):
            call({joint_name: self.joint_names[i]})

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
        self.clear_joint_forces()
        model_state = self.get_state()
        # loss, done = standing_objective(self)
        loss, done = sitting_objective(self)
        return model_state, loss, done

    def try_sit(self, effort, isfeet, isknee):
        self.unpause()
        call = rospy.ServiceProxy('gazebo/apply_joint_effort', ApplyJointEffort)
        if isfeet:
            call(self.joint_names[8], effort, rospy.Duration.from_sec(0), rospy.Duration(0.01))
            call(self.joint_names[9], effort, rospy.Duration.from_sec(0), rospy.Duration(0.01))
        elif isknee:
            call(self.joint_names[6], effort, rospy.Duration.from_sec(0), rospy.Duration(0.01))
            call(self.joint_names[7], effort, rospy.Duration.from_sec(0), rospy.Duration(0.01))
        else:
            call(self.joint_names[4], effort, rospy.Duration.from_sec(0), rospy.Duration(0.01))
            call(self.joint_names[5], effort, rospy.Duration.from_sec(0), rospy.Duration(0.01))

        # self.unpause()
        # self.clear_joint_forces()


def parse_state(state):
    p = state.pose.position
    o = state.pose.orientation
    l = state.twist.linear
    a = state.twist.angular
    return np.asarray([p.x, p.y, p.z, o.x, o.y, o.z, o.w, l.x, l.y, l.z, a.x, a.y, a.z]).reshape((1, 13))

def standing_objective(env):
    head_state = env.get_link_state("link8")
    pelvis_state = env.get_link_state("link0")
    leftknee_state = env.get_link_state("link1l")
    rightknee_state = env.get_link_state("link1r")
    loss = (3.75 - head_state[0, 2])**2 + math.sqrt((0 - head_state[0, 0])**2 + (0 - head_state[0, 1])**2) + math.sqrt((0 - pelvis_state[0, 0])**2 + (0 - pelvis_state[0, 1])**2) + math.sqrt((0 - leftknee_state[0, 1])**2 + (0 - rightknee_state[0, 1])**2)
    done = True if loss > 0.2 else False
    return loss - 1, done

def sitting_objective(env):
    leftknee_state = env.get_link_state("link1l")
    rightknee_state = env.get_link_state("link1r")
    loss = math.sqrt((0 - leftknee_state[0, 1])**2 + (0 - rightknee_state[0, 1])**2)
    done = True if loss > 0.1 else False
    return loss - 1, done

def sit():
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    launch = roslaunch.parent.ROSLaunchParent(uuid, ["src/my_robot_control/launch/my_robot.launch"])
    launch.start()
    rospy.wait_for_service('gazebo/reset_simulation')
    rospy.wait_for_service('gazebo/pause_physics')
    rospy.wait_for_service('gazebo/unpause_physics')
    rospy.wait_for_service('gazebo/apply_joint_effort')
    env = Env()
    env.pause()

    isfeet = True

    while True:
        string = raw_input("Input")
        if string == "Q":
            break
        elif string == "R":
            env.reset()
            env.pause()
        elif string == "F":
            isfeet = True
            isknee = False
        elif string == "K":
            isfeet = False
            isknee = True
        elif string == "H":
            isfeet = False
            isknee = False
        else:
            try:
                effort = float(string)
                env.try_sit(effort, isfeet, isknee)
            except:
                pass


def main():
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    launch = roslaunch.parent.ROSLaunchParent(uuid, ["src/my_robot_control/launch/my_robot.launch"])
    launch.start()
    rospy.wait_for_service('gazebo/get_model_state')
    rospy.wait_for_service('gazebo/get_link_state')
    rospy.wait_for_service('gazebo/reset_simulation')
    rospy.wait_for_service('gazebo/pause_physics')
    rospy.wait_for_service('gazebo/unpause_physics')
    rospy.wait_for_service('gazebo/apply_joint_effort')
    env = Env()
    env.pause()

    global AC
    AC = ActorCritic(env)

    num_trials = 2000
    trial_len = 600 #60 sec = 1 min
    for j in range(num_trials):
        env.reset()
        cur_state = env.get_state()
        for i in range(trial_len):
            action = AC.act(cur_state)
            new_state, loss, done = env.step(action)
            if done or i+1 == trial_len:
                print "trial", j, "lasted", i * 0.1
                print action
                break

            AC.remember(cur_state, action, loss, new_state, done)
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
sit()
# main()
# validate()
