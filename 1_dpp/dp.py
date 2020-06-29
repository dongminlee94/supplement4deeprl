import gym
import time
import argparse
import numpy as np

class DynamicProgramming(object):
   """An implementation of the policy iteration, value iteration."""

   def __init__(self,
                env,
                obs_num,
                act_num,
                gamma=0.99,
                epsilon=1e-6,
   ):

      self.env = env
      self.obs_num = obs_num
      self.act_num = act_num
      self.gamma = gamma
      self.epsilon = epsilon

      # Dynamics
      self.dynamics = self.env.unwrapped.P

   def policy_evaluation(self, pi_table):
      # Initialize value table
      v_table = np.random.uniform(size=(self.obs_num, 1))
      # Run while loop until value table converge
      while True:
         v_prime = np.zeros((self.obs_num, 1))
         for s in self.dynamics.keys():
            for a in self.dynamics[s].keys():
               for trans_prob, next_obs, reward, done in self.dynamics[s][a]:
                  v_prime[s][0] += pi_table[s][a]*trans_prob*(reward + self.gamma*v_table[next_obs])
         distance = np.max(np.abs(v_table-v_prime))
         v_table = v_prime
         # If the distance between value table and value prime is not smaller than epsilon, reiterate loop
         if distance < self.epsilon:
            break
      return v_table

   def policy_improvement(self, v_table):
      # Initialize Q-function table and policy prime
      q_table = np.zeros((self.obs_num, self.act_num))
      pi_prime = np.zeros((self.obs_num, self.act_num))
      # Update Q-function table through policy improvement
      for s in self.dynamics.keys():
         for a in self.dynamics[s].keys():
            for trans_prob, next_obs, reward, done in self.dynamics[s][a]:
               q_table[s][a] += trans_prob*(reward + self.gamma*v_table[next_obs])
      # Update policy table from the action with highest Q-value as 1 at the current state
      pi_prime[np.arange(self.obs_num), np.argmax(q_table, axis=1)] = 1
      return pi_prime

   def policy_iteration(self, start_time):
      # Initialize a policy table
      pi_table = np.random.uniform(size=(self.obs_num, self.act_num))
      # Nomalize the policy table
      pi_table = pi_table/np.sum(pi_table, axis=1, keepdims=True)
      
      iterations = 0
      # Run policy iteration until policy table converge
      while True:
         v_table = self.policy_evaluation(pi_table)
         pi_prime = self.policy_improvement(v_table)
         # If the policy table is not equal to the policy prime, reiterate loop
         if (pi_table == pi_prime).all():
            break
         iterations += 1
         # Change the policy table to the policy prime
         pi_table = pi_prime
         
         print('---------------------------------------')
         print('Iterations:', iterations)
         print('Time:', int(time.time() - start_time))
         print('---------------------------------------')
      return pi_table

   def value_iteration(self, start_time):
      # Initialize value table and policy table
      v_table = np.random.uniform(size=(self.obs_num, 1))
      pi_table = np.zeros((self.obs_num, self.act_num))

      iterations = 0
      # Run value iteration until value table converge
      while True:
         q_table = np.zeros((self.obs_num, self.act_num))
         for s in self.dynamics.keys():
            for a in self.dynamics[s].keys():
               for trans_prob, next_obs, reward, done in self.dynamics[s][a]:
                  q_table[s][a] += trans_prob*(reward + self.gamma*v_table[next_obs])
         # Update value prime from the highest Q-value at the Q-function table
         v_prime = np.max(q_table, axis=1)
         
         distance = np.max(np.abs(v_table-v_prime))
         v_table = v_prime
         # If the distance between value table and value prime is not smaller than epsilon, reiterate loop
         if distance < self.epsilon:
            break
         iterations += 1
         
         print('---------------------------------------')
         print('Iterations:', iterations)
         print('Time:', int(time.time() - start_time))
         print('---------------------------------------')
      
      # Update policy table from the action with highest Q-value as 1 at the current state
      pi_table[np.arange(self.obs_num), np.argmax(q_table, axis=1)] = 1
      return pi_table

   def run(self, pi_table):
      total_reward = 0.

      obs = self.env.reset()
      done = False

      while not done:
         action = np.random.choice(self.act_num, 1, p=pi_table[obs][:])[0]
         next_obs, reward, done, _ = env.step(action)
         
         total_reward += reward
         obs = next_obs
      return total_reward

if __name__ == "__main__":
   # Configurations
   parser = argparse.ArgumentParser(description='Traditional RL algorithms in FrozenLake environment')
   parser.add_argument('--algo', type=str, default='pi',
                        help='select an algorithm among policy iteration (pi), value iteration (vi)')
   args = parser.parse_args()

   # Initialize environment
   env = gym.make('FrozenLake-v0', is_slippery=False)
   obs_num = env.observation_space.n
   act_num = env.action_space.n
   print('State number:', obs_num)
   print('Action number:', act_num)
   env.render()

   start_time = time.time()
   dp = DynamicProgramming(env, obs_num, act_num)
   if args.algo == 'pi':
      pi_table = dp.policy_iteration(start_time)
   elif args.algo == 'vi':
      pi_table = dp.value_iteration(start_time)

   sum_returns = 0.
   for episode in range(100):
      episode_return = dp.run(pi_table)
      sum_returns += episode_return

   env.render()
   print('---------------------------------------')
   print('EpisodeReturn:', episode_return)
   print('SumReturns:', sum_returns)
   print('---------------------------------------')