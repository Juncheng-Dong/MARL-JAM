
from CommuEnv import CommEnv, ENVLIST
from simulation import simulate,simulate_random

import torch
import numpy as np

import argparse
from utils import generate_logger
from evaluation import *

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=4000, help='training epochs')
parser.add_argument('--gpu', type=int, default=0, help='gpu device')
parser.add_argument('--sender',choices=ENVLIST,help='simulation sender type')
args = parser.parse_args()

# GPU
devices = [torch.device(f"cuda:{i}") for i in range(4)]
device = devices[args.gpu]
torch.cuda.set_device(device)
print(f"GPU:{device}")
# Logger
logger = generate_logger(f'Greedy_{args.sender}')


# Simulation

N_CHANNEL = 5
N_JAMMER = 3
S_POWERS = [1,3,5]
J_POWERS = [0,1,3,5]
C_power = 1
T_max = 3000
sender_type=args.sender

# Single-Channel
env=CommEnv(N_CHANNEL, S_POWERS, commu_type=sender_type)
env_name = 'Constant Sender - Single Channel'

logger.info('MAB Greedy')
for _ in range(3):
    logger.info(f'run {_}')
    random_jammers, random_rewards, random_SINR, sender_SNR = simulate_random(env, N_JAMMER, N_CHANNEL, J_POWERS, C_power, T_max)
    greedy_rewards, greedy_rewards_real, greedy_SINR, greedy_SINR_real, sender_SNR,_ = simulate(env, 1, N_CHANNEL, J_POWERS, C_power, random_jammers, device=device, time_range=T_max, greedy=True)

    for threshold in [0.1,0.3,0.5]:
        logger.info(f'success rate of threshold {threshold}: {success_rate(greedy_SINR, env_name, N=100, plot =False,threshold=threshold,maxSNR=sender_SNR[0][0]/0.1)}')

logger.info('RL Greedy')
for _ in range(3):
    logger.info(f'run {_}')
    random_jammers, random_rewards, random_SINR, sender_SNR = simulate_random(env, N_JAMMER, N_CHANNEL, J_POWERS, C_power, T_max)
    greedy_rewards, greedy_rewards_real, greedy_SINR, greedy_SINR_real, sender_SNR,_ = simulate(env, 1, N_CHANNEL, J_POWERS, C_power, random_jammers,device=device, time_range=T_max, EPS_START=0)

    for threshold in [0.1,0.3,0.5]:
        logger.info(f'success rate of threshold {threshold}: {success_rate(greedy_SINR, env_name, N=100, plot =False,threshold=threshold,maxSNR=10)}')


# Multi-Channel
env=CommEnv(N_CHANNEL, S_POWERS, commu_type=sender_type, multi_channel = 2,fix_power=[1,3])
env_name = 'Sweep Sender - Multi Channel'
for _ in range(3):
    logger.info(f'run {_}')
    random_jammers, random_rewards, random_SINR, sender_SNR = simulate_random(env, N_JAMMER, N_CHANNEL, J_POWERS, C_power, T_max)
    greedy_rewards, greedy_rewards_real, greedy_SINR, greedy_SINR_real, sender_SNR,_ = simulate(env, 1, N_CHANNEL, J_POWERS, C_power, random_jammers,device=device, time_range=T_max, greedy=True)

    for threshold in [0.1,0.3,0.5]:
        logger.info(f'success rate of threshold {threshold}: {success_rate(greedy_SINR, env_name, N=100, plot =False,threshold=threshold,maxSNR=40)}')

logger.info('RL Greedy')
for _ in range(3):
    logger.info(f'run {_}')
    random_jammers, random_rewards, random_SINR, sender_SNR = simulate_random(env, N_JAMMER, N_CHANNEL, J_POWERS, C_power, T_max)
    greedy_rewards, greedy_rewards_real, greedy_SINR, greedy_SINR_real, sender_SNR,_ = simulate(env, 1, N_CHANNEL, J_POWERS, C_power, random_jammers,device=device, time_range=T_max, EPS_START=0)

    for threshold in [0.1,0.3,0.5]:
        logger.info(f'success rate of threshold {threshold}: {success_rate(greedy_SINR, env_name, N=100, plot =False,threshold=threshold,maxSNR=40)}')