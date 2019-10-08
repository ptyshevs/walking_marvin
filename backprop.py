import numpy as np


def train_backprop(nn, env):
	for i_episode in range(50):
		losses = []
		observation_arr = []
		cum_reward = 0
		done = False

		print("Epoch #", i_episode)

		observation = env.reset()
		while not done:
			env.render()

			observation = np.array(observation).reshape((24,))
			observation_arr.append(observation)

			action = nn.predict(observation).reshape((4))
			observation, reward, done, info = env.step(action)

			cum_reward = reward - cum_reward
			nn.nn_backprop_one(observation, cum_reward)

			losses.append(cum_reward)

			if done:
				# print("Episode finished after {} timesteps".format(t+1))
				break

def render_backprop(nn, env):
	done = False
	observation = env.reset()

	while not done:
		env.render()
		observation = np.array(observation).reshape((24,1))

		action = nn.predict(observation).reshape((4))
		observation, reward, done, info = env.step(action)
