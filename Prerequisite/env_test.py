##################################
##### 1. Check numpy, torch  #####
##################################

import numpy
import torch

print('numpy' + numpy.__version__)
print('torch' + torch.__version__)

##################################################################

########################
##### 2. Check gym #####
########################

import gym

env = gym.make('CartPole-v1')

for episode in range(10000):
    done = False
    obs = env.reset()

    while not done:
        env.render()

        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)

        print('observation: {} | action: {} | reward: {} | next_observation: {} | done: {}'.format(
                obs, action, reward, next_obs, done))
        
        obs = next_obs