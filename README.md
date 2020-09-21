# Are the space invaders deterministic or stochastic?

<table bordercolor:red>
    <tr>
        <td><img src="GymBreakout-20200906194629-15893-427.0_small.gif" width="400" /></td>
        <td valign="top">
          Google deepmind achieved <a href="https://www.nature.com/articles/nature14236">human-level performance on 49 Atari games</a> 
          using the Arcade Learning Environment (<a href="https://arxiv.org/abs/1207.4708">ALE</a>). 
          I discuss in this <a href="https://medium.com/@nicolasmaquaire/are-the-space-invaders-deterministic-or-stochastic-595a30becae2">article</a> 
          the efficiency of the mechanisms used by Deepmind and Open AI for injecting stochasticity in the ALE.<br>
          This repository contains the code I used to reproduce this performance on Breakout and Space Invaders with the exact same network architecture (DQN).<br>
          I also shared two notebooks. The <a href="https://github.com/NicMaq/Reinforcement-Learning/blob/master/Breakout_explained.ipynb">first</a>
          is an in-depth explanation of the algorithm I used. The <a href="https://github.com/NicMaq/Reinforcement-Learning/blob/master/e_greedy_and_softmax_explained.ipynb">second</a> is an explanation of the two soft policies I implemented: e-greedy and softmax.
        </td>
    </tr>
</table>

To run your own experiments, modify the global hyperparameters at the beginning of each file. Additionally, I used argparse for a few settings: <br>

--new to create a new model <br>
--name name_of_the_model to use an existing model<br>
--env name_of_the_environment to set the OpenAI Gym environment<br>
--render to render the games<br>
--target to set the device where the tensorflow operations are executed. Use -1 if you don’t have a GPU.<br>
--debug to see where the tensorflow operations are executed<br>
--policy to select a different policy or algorithm. The default is Q-Learning with an e-greedy policy. Other options are “sarsa” to use expected sarsa with an e-greedy policy and “softmax” to use expected sarsa with a softmax policy.

Example: 
```
python GYM_BREAKOUT.py --new --target -1 --env BreakoutDeterministic-v4
```
```
python GYM_SPACE_INVADERS.py --new --target 1 --env SpaceInvaders-v4 --render
```
```
python GYM_SPACE_INVADERS.py --new --target -1 --env SpaceInvaders-v4 --render --policy softmax
```

You can compare your experiments with the tensorboard runs I added to this repository. You can find the hyperparamters of these experiments in the chapter [methods](https://docs.google.com/document/d/e/2PACX-1vQVP3qsMYCQrchrfmr2zznL_lFt-bHGgbolr40VxdMKab3k3ksDapX7b_XqjZXmnXuZTVOhqR_QJy_n/pub) of the article.
