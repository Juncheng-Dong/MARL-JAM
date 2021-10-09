from CommEnv import CommEnv
from simulation import simulate

# Define Simulation Constant
N_CHANNEL = 5
N_JAMMER = 3
S_POWERS = [1,3,5]
J_POWERS = [0,1,3,5]
C_power = 0.811
T_max = 50

# Generate Communication Environment
env=CommEnv(N_CHANNEL, S_POWERS, commu_type='constant')
env_name = 'Constant Sender - Single Channel'

# 1. Simulate Performance of Random Jammers
jammers, random_rewards, random_SINR = simulate_random(env, N_JAMMER, N_CHANNEL, J_POWERS, T_max)
# 2. Simulate Performance of Single RL jammer with 1 channel
single_rewards, single_rewards_real, single_SINR, sender_SNR = simulate(env, 1, J_POWERS, jammers, 1000,multi_channel=3)
print('\n')
success_rate(single_SINR, env_name, N=100, plot =False, threshold=0.2)