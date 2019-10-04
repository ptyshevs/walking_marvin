### Fast Evolution Strategy for Walking Marvin

This is a design doc for the implementation that I've come up with.

## Server

The purpose of Server is to synchronize progress across multiple Clients as well as distribute work to each of the Client.
In order to save make transmission lighter, Server work package consists of:
1. Random seed to be used by a client in order to generate perturbations. This seed is remembered on server-side in order to replicate
   perturbations during update phase.
2. Number of iterations to perform.
3. Population size
4. Current model architecture (weights) to be recreated locally
5. Time limit

Client returns:
1. Status

* [Evolution Strategies as a Scalable Alternative to RL](https://openai.com/blog/evolution-strategies/)
* [Mirrored Sampling](https://hal.inria.fr/inria-00530202v2/document)