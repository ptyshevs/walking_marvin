import argparse
import pickle as pcl
import sys
import gym

from nn import *
from ESSolver import *
from ESRaySolver import *

from viz import render_env

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
		observation_arr.append(observation)

		action = nn.nn_prop(observation).reshape((4))
		observation, reward, done, info = env.step(action)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--walk', '-w', default=False, help="Display only walking process", action='store_true')
	parser.add_argument('--load', '-l', help="Load weights for Marvin agent from a file, skipping training process")
	parser.add_argument('--save', '-s', help="Save weights to a file after running the program")
	parser.add_argument('--multiprocess', '-m', help="Spread training across multiple processes", default=False, action='store_true')
	parser.add_argument('--detailed-log', '-d', default=False, help="Display detailed log", action='store_true')
	parser.add_argument('--sonic', default=False, help="--walk, but swiftly", action='store_true')
	parser.add_argument('--backprop', default=False, help="Use Backprop-based training", action='store_true')
	args = parser.parse_args()

	layer_sizes = [24, 24, 4]

	if args.backprop:
		agent = NNBackProp()
		agent.init_nn()
	else:
		agent = NN(layer_sizes, seed=0)

	env_name = "Marvin-v0"
	env = gym.make(env_name)

	train_phase = True
	if args.load is not None:
		if args.backprop:
			agent.load_weights(args.load)
		else:
			try:
				with open(args.load, 'rb') as f:
					weights = pcl.load(f)
					agent.set_weights(weights)
					train_phase = False
			except FileNotFoundError:
				print(f"Failed to load file: {args.load}")
				sys.exit(1)
	if args.sonic and not args.backprop:
		with open("pre_trained/fast_marvin.pcl", 'rb') as f:
			weights = pcl.load(f)
			agent.set_weights(weights)
			train_phase = False

	if args.walk:
		train_phase = False

	# Train the agent
	if train_phase:
		if args.backprop:
			train_backprop(agent, env)
		else:
			if args.multiprocess:
				solver = ESRaySolver(agent, env_name, verbose=True)
			else:
				solver = ESSolver(agent, env_name, verbose=True)
		
			optimal_weights = solver.solve(detailed_log=args.detailed_log)
			agent.set_weights(optimal_weights)

	if args.save:
		if args.backprop:
			agent.save_weights(args.save)
		else:
			with open(args.save, 'wb') as f:
				pcl.dump(agent.get_weights(), f)
	
	if args.backprop:
		render_backprop(agent, env)
	else:
		render_env(agent, env)
	env.close()
