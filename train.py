#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 15:20:14 2023

@author: ahmadrezafrh
"""

from env.custom_hopper import *

from utils import PixelObservationWrapper
from utils import GrayScaleObservation
from utils import randomize_mass
from env import VecFrameStack
from env import DummyVecEnv
from models import CNNFeatureExtractor

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

import gym
import os

import warnings
import itertools



def main():
    
    
    '''
    Setting parameters for training in configue dictionary,
    
        1- alg : it is the algorithm we use, for this project for sake of simplicity and reducing
                 training time, we decided to choose PPO.
                   
        2- obs : it determines wether we use "mlp"(for raw states) or "cnn"(for pixels state).

        3- dist1/dist2/dist3 : they determine three distributions for three links
                               of the hopper.
        
        4- logs_dir/models_dir : these folders are used for saving all trained models.
                                 mainly used for visualization with tensorboard.
                                            
        5- ignore_warnings : ignores the tensor warnings (for better visualiziation of the progress bar).
        
        6- domain_randomization : if it's True we use domain randomization on.
        
        7- stacked_environment : dir for saving best trained models.
        
        8- gray_scale : changing RGP to gray scale.
        
        9- vec_normalize : wether we normalize input vectors or not.
        
        10- custom_arch : determines wether we use custom architecture or not.
        
        11- policies : list of policies we wanna use in tuning.
        
        12- n_frame_stacks : total frames stacked together for processing in cnn.
        
        13- time_steps : total time_steps.
        
        14- step_per_iter : total steps taken(frames) per iteration.
        
        15- policy_kwargs : are the parameters for custom architecture of actor/critic networks
                            (it can be different from mlp to cnn)
        
    '''

    

    policy_kwargs = dict(
        features_extractor_class=CNNFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=512)
    )
    
    # policy_kwargs = dict(
    #     activation_fn=th.nn.ReLU,
    #     net_arch=dict(pi=[32, 32], vf=[32, 32])
    # )
    
    domain_randomization_space = {'dist1' : [[1.5, 3]],
                                  'dist2' : [[2, 3]],
                                  'dist3' : [[2, 3]]}
    
    configue = {'env' : "CustomHopper-source-v0",
                'alg' : "ppo",
                'obs' : "mlp",
                'feature_extractor' : 'base_mlp',
                'ignore_warnings' : True,
                'domain_randomization' : False,
                'stacked_environment' : False,
                'gray_scale' : False,
                'vec_normalize' : False,
                'custom_arch' : False,
                'policies' : ['MlpPolicy'],
                'n_frame_stacks' : 8,
                'time_steps' : 5e5,
                'step_per_iter' : 10000,
                'learning_rates' : [0.001, 0.0001, 0.00001, 0.000001],
                'gammas' : [0.8, 0.9, 0.99, 1],
                'clip_range' : [0.1, 0.2, 0.3],
                'ent_coefs' : [0, 0.0001, 0.001],
                'gae_lambda' : [0.95],
                'n_steps' : [2048, 4096, 8192],
                "policy_kwargs" : policy_kwargs,
        }
    
    params = [configue['policies'], configue['learning_rates'], configue['gammas'], configue['clip_range'], configue['ent_coefs'], domain_randomization_space['dist1'], domain_randomization_space['dist2'], domain_randomization_space['dist3'], configue['n_steps'], configue['gae_lambda']]
    hyperparams = list(itertools.product(*params))

    logs_dir = os.path.join('./logs', configue['alg'], configue['obs'], 'brute_force', f"{'domain_randomized' if configue['domain_randomization'] else 'not_domain_randomized'}", f"{configue['feature_extractor'] if configue['custom_arch'] else 'emprical'}")
    models_dir = os.path.join('./models', configue['alg'], configue['obs'], 'brute_force', f"{'domain_randomized' if configue['domain_randomization'] else 'not_domain_randomized'}", f"{configue['feature_extractor'] if configue['custom_arch'] else 'emprical'}")
    
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)


    '''
    Ignoring all warnings for better visualization
    '''
    if configue['ignore_warnings']:
        warnings.filterwarnings("ignore")
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    
    
    env = gym.make(configue['env'])
    if configue['mlp'] & configue['vec_normalize'] == 'cnn':
        env = VecNormalize(env, norm_obs=True, norm_reward=True,clip_obs=10.)  
    if configue['obs'] == 'cnn':
        env = PixelObservationWrapper(env)
        if configue['gray_scale']:
            env = GrayScaleObservation(env, keep_dim=True)
        env = DummyVecEnv([lambda: env])
        if configue['stacked_environment']:
            env = VecFrameStack(env, n_stack=configue['n_frame_stacks'], channels_order='last')
    
        

    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space
    print('Dynamics parameters:', env.get_parameters())  # masses of each link of the Hopper
    
    

    
                  
    for c, hp in enumerate(hyperparams):
        
        if configue['domain_randomization']:
            env = randomize_mass(env, hp[5], hp[6], hp[7])
            
        model = PPO(env=env,
                    policy=hp[0],
                    learning_rate=hp[1],
                    gamma=hp[2],
                    clip_range=hp[3],
                    ent_coef=hp[4],
                    n_steps=hp[8],
                    gae_lambda=hp[9],
                    verbose=0,
                    device="cuda",
                    policy_kwargs=configue['policy_kwargs'],
                    tensorboard_log=logs_dir)

        print(f'training {c}/{len(hyperparams)}:\n')
        print(f'policy = {hp[0]}')
        print(f'learning_rate = {hp[1]}')
        print(f'gamma = {hp[2]}')
        print(f'clip_range = {hp[3]}')
        print(f'ent_coef = {hp[4]}\n')
            
        models_dir_steps = os.path.join(models_dir, f"{hp[0]}_{hp[1]}_{hp[2]}_{hp[3]}_{hp[4]}")
        if not os.path.exists(models_dir_steps):
            os.makedirs(models_dir_steps)
        ts_per_iter = configue['step_per_iter']
        for i in range(int(configue['time_steps']/ts_per_iter)):
            model.learn(total_timesteps=ts_per_iter, reset_num_timesteps=False, tb_log_name=f"{models_dir_steps}", progress_bar=True)
            model.save(os.path.join(models_dir_steps, f"{ts_per_iter*i + ts_per_iter}"))
            
        del model
        print('=______________________________________________________________________________\n')




            
if __name__ == '__main__':
    main()
