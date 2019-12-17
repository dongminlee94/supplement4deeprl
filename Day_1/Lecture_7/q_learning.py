import gym
import time
import argparse
import numpy as np

class Agent(object):
   """An implementation of the SARSA, Q-Learning agents."""

   def __init__(self,
                env,
                args,
                obs_num,
                act_num,
                steps=0,
                gamma=0.99,
                epsilon=0.2,
                lr=0.01,
                test_mode=False,
   ):

      self.env = env
      self.args = args
      self.obs_num = obs_num
      self.act_num = act_num
      self.steps = steps
      self.gamma = gamma
      self.epsilon = epsilon
      self.lr = lr
      self.test_mode = test_mode

      # Initialize Q-function table
      self.q_table = np.zeros((self.obs_num, self.act_num))

   def select_action(self, obs):
      if np.random.rand() <= self.epsilon:
         # Choose a random action with probability epsilon
         return np.random.randint(self.act_num)
      else:
         # Choose the action with highest Q-value at the current state
         return np.argmax(self.q_table[obs])
   
   def update_q_table(self, obs, action, reward, next_obs, next_action):
      q = self.q_table[obs][action]
      # The next four line shows the difference between SARSA and Q-Learning
      if self.args.algo == 'sarsa':
         q_backup = reward + self.gamma*self.q_table[next_obs][next_action]
      elif self.args.algo == 'q-learning':
         q_backup = reward + self.gamma*max(self.q_table[next_obs])
      self.q_table[obs][action] += self.lr * (q_backup - q)

   def run(self):
      total_reward = 0.

      obs = self.env.reset()
      done = False

      while not done:
         if self.test_mode:
            action = np.argmax(self.q_table[obs])
            next_obs, reward, done, _ = env.step(action)
         else:
            action = self.select_action(obs)
            next_obs, reward, done, _ = env.step(action)
            next_action = self.select_action(next_obs)
            self.update_q_table(obs, action, reward, next_obs, next_action)

         total_reward += reward
         self.steps += 1
         obs = next_obs
      return total_reward

if __name__ == "__main__":
   # Configurations
   parser = argparse.ArgumentParser(description='Traditional RL algorithms in FrozenLake environment')
   parser.add_argument('--algo', type=str, default='q-learning', 
                        help='select an algorithm among sarsa, q-learning')
   args = parser.parse_args()

   # Initialize environment
   env = gym.make('FrozenLake-v0')
   obs_num = env.observation_space.n
   act_num = env.action_space.n
   print('State number:', obs_num)
   print('Action number:', act_num)
   env.render()

   agent = Agent(env, args, obs_num, act_num)

   start_time = time.time()

   sum_returns = 0.
   num_episodes = 0
   agent.test_mode = False

   # Perform the training phase, during which the agent learns
   for episode in range(10000):
      # Run one episode
      episode_return = agent.run()

      sum_returns += episode_return
      num_episodes += 1

      average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0

      print('---------------------------------------')
      print('Steps:', agent.steps)
      print('Episodes:', num_episodes)
      print('SumReturns:', sum_returns)
      print('AverageReturn:', average_return)
      print('Time:', int(time.time() - start_time))
      print('---------------------------------------')

   sum_returns = 0.
   agent.test_mode = True

   # Perform the test phase -- no learning
   for episode in range(100):
      # Run one episode
      episode_return = agent.run()
      sum_returns += episode_return

   env.render()
   print('---------------------------------------')
   print('EpisodeReturn:', episode_return)
   print('SumReturns:', sum_returns)
   print('---------------------------------------')
