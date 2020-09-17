Google deepmind achieved human-level performance on 49 Atari games using the Arcade Learning Environment (ALE). This repository contains the code I used to reproduce this performance on Breakout and Space Invaders with the exact same network architecture (DQN). 

The methodology is fully described in the [article](https://medium.com/@nicolasmaquaire/are-the-space-invaders-deterministic-or-stochastic-595a30becae2) where I discuss the efficiency of the mechanisms used by Deepmind and Open AI for injecting stochasticity in the ALE.

I also shared two notebooks. The first is an in-depth explanation of the algorithm I used.The second, an explanation of the two soft policies I implemented: e-greedy and softmax. 

To run your own experiments, modify the global hyperparameters at the beginning of each file. Additionally, I used argparse for a few settings: 

--new to create a new model
--

  
--name name_of_the_model to use an existing model
--env name_of_the_environment to set the OpenAI Gym environment
--render to render the games
--target to set the device where the tensorflow operations are executed. Use -1 if you don’t have a GPU.
--debug to see where the tensorflow operations are executed
--policy to select a different policy or algorithm. The default is Q-Learning with an e-greedy policy. Other options are “sarsa” to use expected sarsa with an e-greedy policy and “softmax” to use expected sarsa with a softmax policy.

Example: 

python GYM_BREAKOUT.py --new --target -1 --env BreakoutDeterministic-v4
python GYM_SPACE_INVADERS.py --new --target 1 --env SpaceInvaders-v4 --render
python GYM_SPACE_INVADERS.py --new --target -1 --env SpaceInvaders-v4 --render --policy softmax


