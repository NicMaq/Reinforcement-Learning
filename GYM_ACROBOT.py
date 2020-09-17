from __future__ import absolute_import, division, print_function, unicode_literals

#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import time
from datetime import datetime
import gym
import tensorflow as tf
#import tensorflow_probability as tfp
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import numpy as np
#import random
import math
import matplotlib.pyplot as plt
import argparse
import sys
import pickle
#import copy
#import cv2
from skimage.transform import rescale, resize, downscale_local_mean
import imageio

# Global constants
MAX_STEPS = 5000000
EVAL_STEPS = 250000 # Evaluate the model every EVAL_STEPS frames
EVAL_GAMES = 100    # For EVAL_GAMES games
MINI_BATCH_SIZE = 32 
MAX_SAMPLES = 1000000

# Policy
# qlearning e-greedy = 0 ; expected sarsa e-greedy = 1 ; expected sarsa softmax = 2
POLICY = 0

# NUM_STATES = len([cos(theta1), sin(theta1), cos(theta2), sin(theta2), thetaDot1, thetaDot2])
NUM_STATES = 6
# NUM_ACTIONS 
# NUM_ACTIONS = len([torque = 1, torque = 0, torque = -1]) 
NUM_ACTIONS = 3

# Epsilon = Greedy Policy
MIN_EPSILON = 0.05
MAX_EPSILON = 1 
EXPLORE_STEPS = 100000
ANNEALING_STEPS = 200000
NO_OP_STEPS = 5

# Tau = Softmax Policy
TAU = 0.00001 

# Network update
MODELUPDATE_TRAIN_STEPS = 10000
START_LEARNING = 5000
UPDATE_FREQ = 4
REPEAT_ACTION = 2 

# Save model
SAVEMODEL_STEPS = 1000000 
# Learning rate (alpha) and Discount factor (gamma) 

ALPHA = 0.00001
GAMMA = 0.99

# Epochs for training the DNN - How many mini batches will be sent at each steps for training. 2 = 2 gradient descents at each step
EPOCHS = 1 
 
# Directories
SAVE_DIR = 'models/GymAcrobot/ExpectedSarsa'
ROOT_TF_LOG = 'tf_logs'

#GPU CPU - Use Argparse to modify this 
USE_DEVICE = '/GPU:0' 
USE_CPU = '/CPU:0'
RENDER = False

class Agent:

    def __init__(self, env, model, target_model, optimizer, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self.model = model
        self.target_model = target_model
        self.optimizer = optimizer
        
        with tf.device(USE_DEVICE):
            self.decay = (MAX_EPSILON-MIN_EPSILON) / ANNEALING_STEPS
            self.epsilon = tf.constant(MAX_EPSILON)
            self.epsilon = tf.cast(self.epsilon, dtype=tf.float32)
            self.min_epsilon = tf.constant(MIN_EPSILON, dtype=tf.float16)
            self.min_epsilon = tf.cast(self.min_epsilon, dtype=tf.float32)
            self.epsilon_evaluation = tf.constant(0.05, dtype=tf.float16)
            self.epsilon_evaluation = tf.cast(self.epsilon_evaluation, dtype=tf.float32)

            assert self.epsilon.device[-5:].lower() == USE_DEVICE[-5:].lower(), "epsilon not on : %s" % USE_DEVICE
            assert self.min_epsilon.device[-5:].lower() == USE_DEVICE[-5:].lower(), "min_epsilon not on : %s" % USE_DEVICE
            assert self.epsilon_evaluation.device[-5:].lower() == USE_DEVICE[-5:].lower(), "epsilon_evaluation not on : %s" % USE_DEVICE
        
        self._reset()


    def _reset(self):
        self.states = self.env.reset()
        self.states = np.reshape(self.states,(NUM_STATES,1))
        #print('self.states shape is:', self.states.shape)


    def eval_game(self):

        steps = 0 
        game_reward = 0
        raw_images = []

        self._reset()

        history = np.repeat(self.states, 4, axis=1)
        init_history = history 

        while True:

            # Play next step
            if RENDER: raw = self.env.render(mode='rgb_array')
            
            if steps < NO_OP_STEPS:
                action = np.random.randint(0, high=3, size=1, dtype=int)
                action = action[0]

            elif steps % REPEAT_ACTION == 0:
                
                history_foraction =  np.reshape(history, (1, NUM_STATES,4))
                
                with tf.device(USE_DEVICE):
                    
                    tf_history = tf.constant(history_foraction)
                    tf_history = tf.cast(tf_history, dtype=tf.float32)

                    assert tf_history.device[-5:].lower() == USE_DEVICE[-5:].lower(), "tf_history not on : %s" % USE_DEVICE
                    
                    if POLICY == 2:
                        action_probs = self.choose_action(tf_history, False)
                        probs = action_probs.numpy()
                        action = np.random.choice(NUM_ACTIONS, p=probs.squeeze())
                    else:
                        action = self.choose_action(tf_history, False)
                        action = action.numpy()
               
            action = action - 1
            next_state, step_reward, done, info = self.env.step(action)
            action = action + 1

            if RENDER: raw_images.append(raw)   
            game_reward += step_reward
            next_state = np.reshape(next_state,(NUM_STATES,1))
        
            next_history = np.append(history[:,-3:], next_state, axis=1)

            # if the game is done, break the loop
            if done:
                return game_reward, raw_images

            # move the agent to the next state 
            history = next_history

            steps += 1

    def play_game(self, global_steps):

        loss = np.zeros((1,), dtype=np.float32) 

        steps = 0 
        game_reward = 0
        train_time = 0

        data_states = []
        data_actions = []
        data_rewards = []
        data_dones = []

        self._reset()

        history = np.repeat(self.states, 4, axis=1)

        while True:

            # Play next step
            #if RENDER: self.env.render()
            
            if steps % REPEAT_ACTION == 0:
                
                history_foraction =  np.reshape(history, (1, NUM_STATES, 4))

                with tf.device(USE_DEVICE):
                    
                    tf_history = tf.constant(history_foraction)
                    tf_history = tf.cast(tf_history, dtype=tf.float32)

                    assert tf_history.device[-5:].lower() == USE_DEVICE[-5:].lower(), "tf_history not on : %s" % USE_DEVICE
                    
                    if POLICY == 2:
                        action_probs = self.choose_action(tf_history, True)
                        probs = action_probs.numpy()
                        action = np.random.choice(NUM_ACTIONS, p=probs.squeeze())
                    else:
                        action = self.choose_action(tf_history, True)
                        action = action.numpy()
            
            #action is -1, 0 or 1
            action = action - 1
            next_state, step_reward, done, info = self.env.step(action)
            action = action + 1
            game_reward += step_reward
            next_state = np.reshape(next_state,(NUM_STATES,1))
            
            '''
            if step_reward > 0:
                step_reward = 1
            elif step_reward == 0:
                step_reward = 0
            else:
                step_reward = -1           
            '''

            next_history = np.append(history[:,-3:], next_state, axis=1)
            next_state = np.reshape(next_state,(NUM_STATES,))

            # Decay epsilon
            with tf.device(USE_DEVICE):
                if self.epsilon > self.min_epsilon and global_steps > EXPLORE_STEPS:
                        self.epsilon -= self.decay
                        if self.epsilon < self.min_epsilon: 
                            self.epsilon = tf.constant(self.min_epsilon)
                        assert self.epsilon.device[-5:].lower() == USE_DEVICE[-5:].lower(), "self.epsilon not updated on : %s" % USE_DEVICE
            
            if  done:
                step_reward = -1

            data_actions.append(action) 
            data_states.append(next_state) 
            data_rewards.append(step_reward) 
            data_dones.append(int(done)) 
           
            if steps % UPDATE_FREQ == 0:
            
                if global_steps > START_LEARNING:
                    
                    lap_time = time.time()
                    
                    with tf.device(USE_DEVICE):
                        # Calculate target
                        lossBatch = self.calculate_target_and_train() 
                        lossMean = tf.reduce_mean(lossBatch)
                        loss += lossMean.numpy() 

                    train_time +=  time.time() - lap_time
                
            # if the game is done, break the loop
            if done:

                np_data_states = np.asarray(data_states, dtype=np.float32)
                np_data_rewards = np.asarray(data_rewards, dtype=np.int16)
                np_data_actions = np.asarray(data_actions, dtype=np.int16)
                np_data_dones = np.asarray(data_dones, dtype=np.int16)

                data = (np_data_states, np_data_actions, np_data_rewards, np_data_dones)
                
                return data, steps, game_reward, loss, train_time

            # move the agent to the next state 
            history = next_history

            steps += 1
 
    #@tf.function    
    def calculate_target_and_train(self):

        loss = tf.constant(0)
        loss = tf.cast(loss, dtype=tf.float32)

        #yield history, next_history, action_one_hot, terminals, rewards
        for batch_history, batch_next_history, batch_action_one_hot, batch_terminal, batch_reward in self.exp_buffer.dataset.take(EPOCHS):

            batch_action_all_ones = tf.ones_like(batch_action_one_hot)
            
            # predict Q(s',a') for the Bellman equation
            next_qsa = self.target_model((batch_next_history, batch_action_all_ones), training=True)
            
            if POLICY == 1:

                # e-greedy policy - Expected Sarsa
                sum_piq = egreedy_policy(next_qsa, self.epsilon)
                v_next_vect = batch_terminal * sum_piq

            elif POLICY == 2:

                # Softmax policy - Expected Sarsa
                action_probs = softmax_policy(next_qsa)
                expectation = tf.multiply(action_probs, next_qsa)
                sum_expectation = tf.reduce_sum(expectation, axis=1, keepdims=True)
                v_next_vect = batch_terminal * sum_expectation

            else:

                # e-greedy policy - Q-Learning
                max_q = tf.math.reduce_max(next_qsa, axis=1, keepdims=True)
                v_next_vect = batch_terminal * max_q
            
            target_vec = batch_reward + GAMMA * v_next_vect
            target_mat = tf.multiply(target_vec, batch_action_one_hot)

            # Predict Q(s,a)
            with tf.GradientTape() as tape:

                qsa = self.model((batch_history, batch_action_one_hot), training=True)

                qsa_mat = tf.multiply(qsa, batch_action_one_hot)
                delta_mat = target_mat - qsa_mat
                
                # Huber loss
                squared_loss = 0.5 * tf.square(delta_mat)
                linear_loss = tf.abs(delta_mat) -0.5
                ones = tf.ones_like(delta_mat)
                loss_mat = tf.where(tf.greater(linear_loss, ones), x = linear_loss, y = squared_loss)
                loss_train = tf.reduce_mean(loss_mat, axis=1, keepdims=True)

                grads = tape.gradient(loss_train, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))        
                
                loss = tf.add(loss_train,loss)

        return loss


    #@tf.function    
    def choose_action(self, states, isTraining):
        
        actions_all_ones = tf.ones((1,NUM_ACTIONS))

        if POLICY == 2:
            
            # softmax
            qsa = self.model((states, actions_all_ones), training=isTraining)
            action_probs = softmax_policy(qsa)

            return action_probs 
        
        else:

            # e-greedy
            randomNum = tf.random.uniform((), minval=0, maxval=1, dtype=tf.float32, seed=1)
            if randomNum < self.epsilon:
                random_action = tf.random.uniform((), minval=0, maxval=NUM_ACTIONS, dtype=tf.int32)
                best_action = random_action 

            else:

                qsa = self.model((states, actions_all_ones), training=isTraining)
                best_action = tf.math.argmax(qsa, axis=1, output_type=tf.dtypes.int32)
                #best_action = argmax_ties(qsa)
                best_action = best_action[0]
            
            return best_action 


class ExperienceBuffer:
    def __init__(self):

        self.states = np.empty(shape=(1,NUM_STATES), dtype=np.float32)
        self.actions = np.empty(shape=(1,), dtype=np.int16)
        self.rewards = np.empty(shape=(1,), dtype=np.int16)
        self.dones = np.empty(shape=(1,), dtype=np.int16)

        with tf.device(USE_DEVICE):
            types = tf.float32, tf.float32, tf.float32, tf.float32,tf.float32 
            shapes = (MINI_BATCH_SIZE,NUM_STATES,4), \
                    (MINI_BATCH_SIZE,NUM_STATES,4), \
                    (MINI_BATCH_SIZE,NUM_ACTIONS), \
                    (MINI_BATCH_SIZE,1), \
                    (MINI_BATCH_SIZE,1)  

            fn_generate = lambda: self.generate_data()
            self.dataset = tf.data.Dataset.from_generator(fn_generate, \
                                         output_types= types, \
                                         output_shapes = shapes)
            self.dataset = self.dataset.prefetch(buffer_size=2*EPOCHS)
       
    def count(self):
        return self.states.shape[0]

    def pop(self):
        self.states = self.states[1:,:]
        self.actions = self.actions[1:]
        self.rewards = self.rewards[1:]
        self.dones = self.dones[1:]

    def append(self, experiences):
        self.states = np.append(self.states, experiences[0], axis=0)
        self.actions= np.append(self.actions, experiences[1], axis=0)
        self.rewards = np.append(self.rewards, experiences[2], axis=0)
        self.dones = np.append(self.dones, experiences[3], axis=0)

        if self.states.shape[0] > MAX_SAMPLES:
            self.states = self.states[-MAX_SAMPLES:,:] 
            self.actions = self.actions[-MAX_SAMPLES:] 
            self.rewards = self.rewards[-MAX_SAMPLES:] 
            self.dones = self.dones[-MAX_SAMPLES:] 
    
    def generate_data(self):

        mini_batch_size = MINI_BATCH_SIZE
        mini_batch_size = float(mini_batch_size)
        num_samples = mini_batch_size * 1.7 # We don't know how many samples we'll remove
        num_samples = int(num_samples)

        while True:
           
            replay_states = self.states
            replay_actions = self.actions
            replay_rewards = self.rewards
            replay_dones = self.dones

            indices4 = np.random.randint(low=0, high=self.count()-4, size=num_samples)
            indices4 = indices4 + 4
            indices3 = indices4 -1
            indices2 = indices3 -1
            indices1 = indices2 -1
            indices0 = indices1 -1
            indices = np.stack((indices0,indices1,indices2,indices3,indices4), axis=0)
            indices = np.reshape(np.transpose(indices),(num_samples*5,))
            reshaped_indices= np.reshape(indices,(-1,5))
            reshaped_indices4 = np.reshape(indices4,(-1,1))

            gathered_states = np.take(replay_states, reshaped_indices, axis=0)
            gathered_actions = np.take(replay_actions, reshaped_indices4, axis=0)
            gathered_rewards = np.take(replay_rewards, reshaped_indices4, axis=0)
            gathered_dones = np.take(replay_dones, reshaped_indices4, axis=0)
            first5_dones =  np.take(replay_dones, reshaped_indices, axis=0)

            # Remove bad samples
            first4_dones = first5_dones[:,:-1]
            any_bad_samples = np.any(first4_dones, axis=1)
            indices_ok = np.logical_not(any_bad_samples)

            rewards_filtered = gathered_rewards[indices_ok,:]
            states_filtered = gathered_states[indices_ok,:]
            actions_filtered = gathered_actions[indices_ok,:]
            dones_filtered = gathered_dones[indices_ok,:]

            rewards = rewards_filtered[0:MINI_BATCH_SIZE,:]
            states = states_filtered[0:MINI_BATCH_SIZE:,:,:]
            actions = actions_filtered[0:MINI_BATCH_SIZE,:]
            dones = dones_filtered[0:MINI_BATCH_SIZE,:]

            raw_history = states[:,0:4,:]
            history = np.transpose(raw_history,(0,2,1))
            raw_next_history = states[:,1:5,:]
            next_history = np.transpose(raw_next_history,(0,2,1))

            actions = actions.astype(int)
            actions = np.reshape(actions,(-1,))

            action_one_hot = np.eye(NUM_ACTIONS)[actions]
            action_one_hot = action_one_hot.astype(float)
            action_one_hot

            terminals = 1 - dones

            history = history.astype(np.float32)
            next_history = next_history.astype(np.float32)
            action_one_hot = action_one_hot.astype(np.float32)
            terminals = terminals.astype(np.float32)
            rewards = rewards.astype(np.float32)
            
            yield history, next_history, action_one_hot, terminals, rewards


@tf.function
def argmax_ties(qsa):

    print('Tracing argmax_ties')

    best_action = tf.math.argmax(qsa, axis=1, output_type=tf.dtypes.int32)

    all_ones = tf.ones_like(qsa)
    max_q = tf.math.reduce_max(qsa, axis=1, keepdims=True)
    qsa_max_m = max_q * all_ones
    losers = tf.zeros_like(qsa)
    qsa_maximums = tf.where(tf.equal(qsa_max_m, qsa), x=all_ones, y=losers)
    nb_maximums = tf.math.reduce_sum(qsa_maximums, axis=1, keepdims=True)

    only_one_max = tf.ones_like(nb_maximums)
    isMaxMany = tf.greater(nb_maximums, only_one_max)
    
    if tf.reduce_any(isMaxMany):

      qsa_maximums_ind = tf.where(tf.equal(qsa_max_m, qsa))
      nbr_maximum_int = tf.reshape(nb_maximums,[-1])
      nbr_maximum_int = tf.dtypes.cast(nbr_maximum_int, tf.int32)

      for idx in tf.range(best_action.shape[0]):
        if isMaxMany[idx]:
            selected_idx = tf.random.uniform((), minval=0, maxval=nbr_maximum_int[idx], dtype=tf.int32)
            rows_index = tf.slice(qsa_maximums_ind,[0,0],[-1,1])
            all_actions = tf.slice(qsa_maximums_ind,[0,1],[-1,-1])
            current_index = tf.ones_like(rows_index)
            current_index = current_index * tf.cast(idx, dtype=tf.int64)
            selected_rows = tf.where(tf.equal(rows_index,current_index))
            select_action = tf.slice(selected_rows,[0,0],[-1,1])
            select_action = tf.squeeze(select_action)
            new_action = all_actions[select_action[selected_idx]]
            new_action = tf.cast(new_action, dtype=tf.int32)
            tf.print('***************************************************************************************** \n')
            tf.print('egreedy tie management new_action is: ', new_action)
            tf.print('***************************************************************************************** \n')

            indice = tf.reshape(idx,(1,1))
            tf.tensor_scatter_nd_update(best_action, indice, new_action)

    return best_action


@tf.function
def softmax_policy(qsa):

    print('Tracing softmax_policy')

    preferences = qsa / TAU
    max_preference =  tf.math.reduce_max(qsa, axis=1, keepdims=True) / TAU
    pref_minus_max = preferences - max_preference
    exp_preferences = tf.math.exp(pref_minus_max)
    sum_exp_preferences = tf.reduce_sum(exp_preferences, axis=1, keepdims=True)
    action_probs = exp_preferences / sum_exp_preferences

    return action_probs


@tf.function
def egreedy_policy(qsa, epsilon):

    print('Tracing egreedy_policy')

    all_ones = tf.ones_like(qsa)
    max_q = tf.math.reduce_max(qsa, axis=1, keepdims=True)
    qsa_max_m = max_q * all_ones
    losers = tf.zeros_like(qsa)
    qsa_maximums = tf.where(tf.equal(qsa_max_m, qsa), x=all_ones, y=losers)
    nb_maximums = tf.math.reduce_sum(qsa_maximums, axis=1, keepdims=True)

    num_actions_float = tf.dtypes.cast(NUM_ACTIONS, tf.float32)
    pi_s = tf.dtypes.cast(all_ones, tf.float32)
    pi_s = pi_s * epsilon / num_actions_float
    pi_max = (1 - epsilon)/nb_maximums
    pi = qsa_maximums * pi_max + pi_s

    pi_qsa = tf.multiply(pi, qsa)
    sum_piq = tf.math.reduce_sum(pi_qsa, axis=1, keepdims=True)

    return sum_piq


def build_keras_Seq():
    
    actions = tf.keras.Input(shape=(NUM_ACTIONS,), name='actions')
    states  = tf.keras.Input(shape=(NUM_STATES,4),name='states')

    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='untruncated_normal', seed=None)
    init0 = tf.keras.initializers.Zeros()
    init1 = tf.keras.initializers.Ones()
    init2 = tf.keras.initializers.GlorotUniform(seed=1) #[-limit, limit], where limit = sqrt(6 / (fan_in + fan_out))
    
    x = tf.keras.layers.Dense(64, kernel_initializer=init2)(states)
    x = tf.keras.activations.relu(x)#, max_value=6)
    x = tf.keras.layers.Dense(256, kernel_initializer=init2)(x)
    x = tf.keras.activations.relu(x)#, max_value=6)
    x = tf.keras.layers.Dense(128, kernel_initializer=init2)(x)
    x = tf.keras.activations.relu(x)#, max_value=6)
    #x = tf.keras.layers.Dense(64, kernel_initializer=init)(x)   
    #x = tf.keras.activations.relu(x)#, max_value=6)
    x = tf.keras.layers.Flatten()(x)

    q_values = tf.keras.layers.Dense(NUM_ACTIONS, name='q_values', kernel_initializer=init2)(x)
    output = tf.keras.layers.Multiply(name='Qs')([q_values, actions])

    model = tf.keras.Model(inputs=[states,actions], outputs=output)

    return model       


def run_training(agent, now, modelPath, modelId):

    logdir = "{}/run/{}/".format(ROOT_TF_LOG, now)
    
    with tf.device(USE_DEVICE):
        file_writer = tf.summary.create_file_writer(logdir)

    # Memory replay - Load if any
    if modelId is not None:
        samples = unpickle(SAVE_DIR + '/' + modelId)
        agent.exp_buffer.buffer = np.array(samples)
    else:
        modelId = now

    with tf.device(USE_DEVICE):
        agent.target_model.set_weights(agent.model.get_weights())

    # Metrics - Should be a collections deque with max capacity set to more than last summary scalar successFrame.
    successMemory = np.empty((1,0))
    successFrame = np.empty((1,0))
    previous_global_steps_tn = 0
    previous_global_steps_eval = 0
    
    game_count = 1
    global_steps = 0
    loss =  np.zeros((1,),dtype=np.float32)
    best_score = -500
    
    lap_time = time.time()
   
    try:

        while global_steps <= MAX_STEPS: 

            print('\nGame {} - Run {}'.format(game_count, now))

            #if global_steps % SAVEMODEL_STEPS  > previous_global_steps % SAVEMODEL_STEPS:
            #    samples = [agent.exp_buffer.images, agent.exp_buffer.actions, agent.exp_buffer.rewards, agent.exp_buffer.dones]
            #    save_theModel(model, modelId, game_count, samples)

            # return steps, game_reward, loss, epsilon
            data_game, steps, game_reward, loss, train_time = agent.play_game(global_steps)
            loss /= steps + 1 # steps starts at 0 

            buffer_previous_size = agent.exp_buffer.count()
            agent.exp_buffer.append(data_game)

            global_steps += steps + 1   
            print('Global_steps is: %s' % global_steps)
            
            if buffer_previous_size == 1 :
                print("Experience Replay buffer pop")
                agent.exp_buffer.pop()

            # Update the target network 
            train_steps = (global_steps - previous_global_steps_tn)*EPOCHS*MINI_BATCH_SIZE/UPDATE_FREQ
            if train_steps > MODELUPDATE_TRAIN_STEPS:
                with tf.device(USE_DEVICE):
                    agent.target_model.set_weights(agent.model.get_weights())
                print('Updating target model **************************** Updating target model ****************')
                previous_global_steps_tn = global_steps

            if POLICY == 0 or POLICY == 1: print('Epsilon is: %s' % agent.epsilon)
            
            # Evaluate every EVAL_STEPS frames the performance 
            if (agent.epsilon == agent.min_epsilon and global_steps > previous_global_steps_eval + EVAL_STEPS) or global_steps > MAX_STEPS:
                
                successEval = np.empty((1,0))
                agent.epsilon = agent.epsilon_evaluation 
                remaining_eval_games = EVAL_GAMES
                previous_global_steps_eval = global_steps
                
                while remaining_eval_games > 0:
                    
                    print('Evaluation game %s' % remaining_eval_games)
                    remaining_eval_games -= 1 

                    game_reward, raw_frames = agent.eval_game()
                    print('game_reward is: ', game_reward)
                    successEval = np.append(successEval, game_reward)
        
                    if  game_reward > best_score and RENDER:
                        generate_gif(raw_frames, modelId, game_count, game_reward)
                        best_score = game_reward
                        print('Generating GIF  **************************** Generating Gif ****************')
     
                    if remaining_eval_games == 0:
                        agent.epsilon = agent.min_epsilon 

                    assert agent.epsilon.device[-5:].lower() == USE_DEVICE[-5:].lower(), "agent.epsilon not on : %s" % USE_DEVICE
                
                with file_writer.as_default():
                    with tf.device(USE_DEVICE):
                        tf.summary.scalar('eval', np.mean(successEval), step=global_steps)
                        tf.summary.scalar('eval-var', np.var(successEval), step=global_steps)
                        tf.summary.histogram('scores', successEval, step=global_steps)

            successMemory = np.append(successMemory,game_reward)
            successFrame = np.append(successFrame,np.mean(successMemory[-10:successMemory.size]))
           
            actions_distrib = np.histogram(agent.exp_buffer.actions[-steps:], bins=[0,1,2,3,4,5,6], density=True)

            print('Memory contains %s samples' % agent.exp_buffer.count())       
            print('Reward over 10 games is: %s and loss is: %s' % (successFrame[-1],loss[0]))
            print('Actions distribution (last game, %) is: ', (100 * actions_distrib[0]).astype(int))
            print('Steps survived: %s' % (steps+1))
 
            # Add user custom data to TensorBoard
            with file_writer.as_default():
                with tf.device(USE_DEVICE):
                    tf.summary.scalar('loss', loss[0], step=global_steps)
                    tf.summary.scalar('epsilon', agent.epsilon, step=global_steps)
                    tf.summary.scalar('score', game_reward, step=global_steps)
                    tf.summary.scalar('steps', steps, step=global_steps)
                    tf.summary.histogram('actions', agent.exp_buffer.actions[-steps:], step=global_steps)

            previous_time = lap_time
            lap_time = time.time()
            print("Train time for the last game: ", train_time)
            print("Elapsed time for the last game: ", lap_time - previous_time)
            
            #if game_count == 4:
            #    break
            
            game_count += 1  
            
    except KeyboardInterrupt:

        print('Save the model')
        save_theModel(agent.model, modelId, game_count, agent.exp_buffer)
        file_writer.close()    

        raise

    print('Save the model ', modelId)
    samples = [agent.exp_buffer.states, agent.exp_buffer.actions, agent.exp_buffer.rewards, agent.exp_buffer.dones]
    save_theModel(agent.model, modelId, game_count, samples)
    file_writer.close()   

def generate_gif(frames, pathName, game_count, game_reward):

    for idx, frame_idx in enumerate(frames): 
        frames[idx] = resize(frame_idx, (420, 320, 3), preserve_range=True, order=0).astype(np.uint8)
        
    imageio.mimsave(f'{SAVE_DIR}{"/GymBreakout-{}-{}-{}.gif".format(pathName, game_count, -game_reward)}', frames, duration=1/30)


def save_theModel(model, pathName, game_count, samples):
    
    # pickle the replay memory  
    print('\nPickle the replay memory')
    now_save = pathName + '_' + str(game_count)
    #with open(SAVE_DIR +'/'+ now_save, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        #pickle.dump(samples, f, pickle.HIGHEST_PROTOCOL)
    modelPath = "{}/GymBreakout-{}.h5".format(SAVE_DIR, now_save)
    model.save(modelPath)
    print('Saved model: ', modelPath)
    print(datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S +0000"))


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--new', help='Create new model.', action='store_true')
    parser.add_argument(
      '--render', help='render the env', action='store_true')
    parser.add_argument(
      '--debug', help='create report on model.', action='store_true')
    parser.add_argument(
      '--name', help='Name of the model to load')
    parser.add_argument(
      '--target', help='GPU to use')
    parser.add_argument(
      '--policy', help='Select policy')

    args = parser.parse_args()

    # Set globals
    global USE_DEVICE
    global RENDER
    global POLICY 

    if args.target is not None:
        if args.target == '-1':
            USE_DEVICE = USE_CPU
        else:
            USE_DEVICE = 'gpu:{}'.format(args.target)

    if args.render:
            RENDER = True
    
    # e-greedy = 0 ; softmax = 1
    if args.policy is not None:
        if args.policy.lower() == 'sarsa': # Expected Sarsa with egreedy
            POLICY = 1
        elif args.policy.lower() == 'softmax': # Expected Sarsa with softmax
            POLICY = 2
        else:
            POLICY = 0 # 'QLearning 

    gpus = tf.config.list_physical_devices('GPU')
    print('GPUS are: ', gpus)

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu,True)

    if args.debug:
        tf.debugging.set_log_device_placement(True)

    #policy = mixed_precision.Policy('mixed_float16')
    #mixed_precision.set_policy(policy)            

    #print('Compute dtype: %s' % policy.compute_dtype)
    #print('Variable dtype: %s' % policy.variable_dtype)

    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    with open(SAVE_DIR + '/' + now + '.txt', 'w+') as f:
        f.write("now is: %s\n" % now)
        f.write("BreakoutDeterministic-v4 \n")
        f.write("Policy is %s \n" % POLICY)
        f.write("Target is %s \n" % USE_DEVICE)
        f.write("alpha = %s \n" % ALPHA)
        f.write("gamma = %s \n" % GAMMA)
        f.write("Tau = %s \n" % TAU)
        f.write("annealing steps = %s \n" % ANNEALING_STEPS)
        f.write("explore steps = %s \n" % EXPLORE_STEPS)
        f.write("start learning = %s \n" % START_LEARNING)
        f.write("Epochs is = %s \n" % EPOCHS)
        f.write("repeat_action = %s \n" % REPEAT_ACTION)
        f.write("update network = %s \n" % UPDATE_FREQ)
        f.write("model update train steps = %s \n" % MODELUPDATE_TRAIN_STEPS)
        f.write("max steps = %s \n" % MAX_STEPS)
        f.write("max samples = %s \n" % MAX_SAMPLES)
        f.write("mini batch size = %s \n" % MINI_BATCH_SIZE)
        f.write("adaptative learning rate = %s \n" % False)
        f.write("min epsilon %s \n" % MIN_EPSILON)
        f.write("max epsilon %s \n" % MAX_EPSILON)
        f.write("comment: Expected SARSA.  \n")
        f.write("comment: Changed TN update rule.  \n")
        f.write("comment: init2 on the dense layers \n")
        f.write("comment: three dense layers\n")
        f.close()

    # Seeding the random
    # Don't forget to seed the network activation function if needed
    np.random.seed(seed=42)
    #random.seed(43)
    tf.random.set_seed(44)

    # Create env
    env = gym.make('Acrobot-v1')
    env.seed = 45
    #env.reset()

    print("obs shape is: ", env.observation_space.shape)
    print("actions space is: ", env.action_space.n)
    #actions = env.unwrapped.get_action_meanings()
    #print('actions are: ', actions)

    if args.new:

        modelPath = None
        modelId = None

        # Build Model
        with tf.device(USE_DEVICE):

            model = build_keras_Seq()
            target_model = build_keras_Seq()
        
            # learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam', **kwargs
            optimizer = tf.keras.optimizers.Adam(ALPHA, epsilon=1e-8)
            #optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00001, rho=0.9)

        with open(SAVE_DIR + '/' + now + '.txt', 'a') as f:
            f.write("\n\nModel Summary \n\n")
            model.summary(print_fn=lambda x: f.write(x + '\n'))
            f.close()    

    else:

        if args.name is not None:
            print('Loading existing model %s' % args.name)
            modelPath = "{}/{}".format(SAVE_DIR, args.name)
            modelId = args.name[len(args.name)-17:len(args.name)-3]

            with tf.device(USE_DEVICE):
                model = tf.keras.models.load_model(modelPath)
                target_model = tf.keras.models.clone_model(model)
                
                optimizer = tf.keras.optimizers.Adam(ALPHA, epsilon=1e-8)
                #optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00001, rho=0.9)

        else:
            print("A model name is required")
            sys.exit()


    memory = ExperienceBuffer()
    agent = Agent(env, model, target_model, optimizer, memory)

    print(model.summary())

    #Training
    try:
        run_training(agent, now, modelPath, modelId)

    except KeyboardInterrupt:

        # Close env 
        env.close()
        print('Exit on keyboard interrupt')

    env.close()


if __name__ == '__main__':
    main()
