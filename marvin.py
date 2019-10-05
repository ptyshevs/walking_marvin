import argparse
import pickle as pcl
import sys
import gym

from nn import *
from ESSolver import *
from viz import render_env

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--walk', '-w', default=False, help="Display only walking process", action='store_true')
    parser.add_argument('--load', '-l', help="Load weights for Marvin agent from a file, skipping training process")
    parser.add_argument('--save', '-s', help="Save weights to a file after running the program")
    parser.add_argument('--multiprocess', '-m', help="Spread training across multiple processes", default=False, action='store_true')
    parser.add_argument('--detailed-log', '-d', default=False, help="Display detailed log", action='store_true')
    args = parser.parse_args()
    print(args, args.load)

    layer_sizes = [24, 24, 4]
    agent = NN(layer_sizes, seed=0)

    env_name = "Marvin-v0"
    env = gym.make(env_name)

    train_phase = True
    if args.load is not None:
        try:
            with open(args.load, 'rb') as f:
                weights = pcl.load(f)
                agent.set_weights(weights)
                train_phase = False
        except FileNotFoundError:
            print(f"Failed to load file: {args.load}")
            sys.exit(1)
    
    if args.walk:
        train_phase = False


    if train_phase:
        # Train the agent
        if args.multiprocess:
            # Sorry, shitcode here to bypass unncessary activation of ray
            from ESRaySolver import *
            solver = ESRaySolver(agent, env_name, verbose=True)
        else:
            solver = ESSolver(agent, env_name, verbose=True)
        
        weights = solver.solve(n_generations=1)
        agent.set_weights(weights)

    if args.save:
        with open(args.save, 'wb') as f:
            pcl.dump(agent.get_weights(), f)
    
    render_env(agent, env)
    
    


    


