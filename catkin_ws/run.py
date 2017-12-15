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
        # self.memory = None
        self.cur_state = None
        self.cur_target = None
        self.cur_advantages = None

        self.adam = Adam(lr=self.learning_rate)

        self.actor_model = self.create_actor_model()
        self.critic_model = self.create_critic_model()

    def create_actor_model(self): #input: state; output: reward * action val
        model = Sequential()
        model.add(Dense(13, activation='relu', input_shape=(156, )))
        model.add(Dense(26, activation='relu'))
        model.add(Dense(13, activation='relu'))
        model.add(Dense(10, activation='linear'))

        model.compile(loss='mse', optimizer=self.adam)
        return model

    def create_critic_model(self): #input: state; output: reward+gamma*new_reward corr w/ value 
        model = Sequential()
        model.add(Dense(13, activation='relu', input_shape=(156, )))
        model.add(Dense(26, activation='relu'))
        model.add(Dense(13, activation='relu'))
        model.add(Dense(1, activation='linear'))

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

    def remember(self, cur_state, action, reward, new_state, done):
        value = self.critic_model.predict(cur_state)
        new_value = self.critic_model.predict(new_state)

        if not done:
            reward += self.gamma * new_value

        self.cur_state = cur_state
        self.target = action * (reward - value)
        self.advantages = np.asarray(reward).reshape((1, 1))

    def act(self, cur_state):

        # policy = self.actor_model.predict(cur_state, batch_size=1)
        # print type(policy)
        # print policy.shape
        # print policy


        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            return self.env.sample_action().reshape((1, 10))
        return self.actor_model.predict(cur_state, batch_size=1)

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
        j5l6l = random.uniform(-2.5, 2.5)
        j5r6r = random.uniform(-2.5, 2.5)
        j45l = random.uniform(-2.5, 2.5)
        j45r = random.uniform(-2.5, 2.5)
        j01l = random.uniform(-2.5, 2.5)
        j01r = random.uniform(-2.5, 2.5)
        j1l2l = random.uniform(-2.5, 2.5)
        jlr2r =random.uniform(-2.5, 2.5)
        j2l9l = random.uniform(-0.5, 0.5)
        j2r9r = random.uniform(-0.5, 0.5)
        np.array([])
        return np.array([j5l6l,j5r6r,j45l,j45r,j01l,j01r,j1l2l,jlr2r,j2l9l,j2r9r])

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
        self.clear_joint_forces
        model_state = self.get_state()
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
    pelvis_state = env.get_link_state("link0")
    #lower_l_leg = env.get_link_state("link2l")
    #lower_r_leg = env.get_link_state("link2r")
    #upper_l_leg = env.get_link_state("link2l")
    #upper_r_leg = env.get_link_state("link2r")
    lfoot = env.get_link_state("link9l")
    rfoot = env.get_link_state("link9r")
    midpointx = (lfoot[0,0] + rfoot[0,0])/2
    midpointy = (lfoot[0,1] + rfoot[0,1])/2
    
    loss = (3.75 - head_state[0, 2])**2 + ( midpointx - pelvis_state[0,0])**2 + ( midpointy - pelvis_state[0,1])**2
    loss += abs(pelvis_state[0,10]) + abs(pelvis_state[0,11]) + abs(pelvis_state[0,12])
    
    done = True if loss > 1 else False
    return -loss, done

def sitting_objective(model_state):
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

    AC = ActorCritic(env)

    num_trials = 2000
    trial_len = 600 #60 sec = 1 min
    for j in range(num_trials):
        env.reset()
        cur_state = env.get_state()
        # AC.clear_memory()
        for i in range(trial_len):
            action = AC.act(cur_state)
            # print action
            new_state, reward, done = env.step(action)
            if done or i+1 == trial_len:
                print "trial", j, "lasted", i * 0.1
                break

            AC.remember(cur_state, action, reward, new_state, done)
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
