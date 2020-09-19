# Are the space invaders deterministic or stochastic?

Google deepmind achieved [human-level performance on 49 Atari games](https://www.nature.com/articles/nature14236) using the Arcade Learning Environment ([ALE](https://arxiv.org/abs/1207.4708)). I discuss in this [article](https://medium.com/@nicolasmaquaire/are-the-space-invaders-deterministic-or-stochastic-595a30becae2) the efficiency of the mechanisms used by Deepmind and Open AI for injecting stochasticity in the ALE.<br>

This repository contains the code I used to reproduce this performance on Breakout and Space Invaders with the exact same network architecture (DQN).<br>

I also shared two notebooks. The [first](https://github.com/NicMaq/Reinforcement-Learning/blob/master/Breakout_explained.ipynb) is an in-depth explanation of the algorithm I used. The second is an explanation of the two soft policies I implemented: e-greedy and softmax. <br>

To run your own experiments, modify the global hyperparameters at the beginning of each file. Additionally, I used argparse for a few settings: <br>

--new to create a new model <br>
--name name_of_the_model to use an existing model<br>
--env name_of_the_environment to set the OpenAI Gym environment<br>
--render to render the games<br>
--target to set the device where the tensorflow operations are executed. Use -1 if you don’t have a GPU.<br>
--debug to see where the tensorflow operations are executed<br>
--policy to select a different policy or algorithm. The default is Q-Learning with an e-greedy policy. Other options are “sarsa” to use expected sarsa with an e-greedy policy and “softmax” to use expected sarsa with a softmax policy.<br>

Example: <br>

python GYM_BREAKOUT.py --new --target -1 --env BreakoutDeterministic-v4<br>
python GYM_SPACE_INVADERS.py --new --target 1 --env SpaceInvaders-v4 --render<br>
python GYM_SPACE_INVADERS.py --new --target -1 --env SpaceInvaders-v4 --render --policy softmax<br>


