import gym

env = gym.make('Pendulum-v0')

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