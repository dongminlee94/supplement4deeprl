import gym

env = gym.make('FrozenLake-v0', is_slippery=False)

for episode in range(1):
    done = False
    obs = env.reset()
    
    env.render()
    print('\n')

    while not done:
        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)

        env.render()
        print('observation: {} | action: {} | reward: {} | next_observation: {} | done: {}\n'.format(
                obs, action, reward, next_obs, done))
        
        obs = next_obs