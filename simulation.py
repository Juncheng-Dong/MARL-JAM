import numpy as np
from tqdm import tqdm

from agents import GreedyAgent
from agents import RandomJammer
from RLagents import Agent


def instant_reward(SNR, SINR):
  gain = [10*np.log2(1.+x) - 10*np.log2(1.+y) for x, y in zip(SNR, SINR)]
  return sum(gain)

def simulate_random(env, N_JAMMER, N_CHANNEL, J_POWERS, C_power, T_max, multi_channel=1):
  jammers=[]
  for i in range(N_JAMMER):
    jammers.append(RandomJammer(N_CHANNEL,J_POWERS))

  #Running Jamming and collect rewards
  rewards = []
  SINR=[]
  for t in range(T_max):
    actions = []
    for jammer in jammers:
      action = [jammer.jam() for i in range(multi_channel)]
      actions.append(action)
    # actions = [[jammer.jam() for jammer in jammers]]
    SNR_t, SINR_t, _, real_SINR_t = env.step(actions,t)
    SINR.append(real_SINR_t)
    
    # rewards.append([-C_power*j_action[1] for j_action in actions]+[1/SINR_t])
    reward = instant_reward(SNR_t, SINR_t)
    total_power = 0
    for action in actions:
      for j_action in action:
        total_power = total_power + j_action[1]
    cost = -C_power*total_power
    # cost = sum([-C_power*j_action[0][1] for j_action in actions])
    rewards.append(reward + cost)
  return jammers, rewards, sum(SINR, []), sum(SNR_t,[])


def simulate(env, N_JAMMER, N_CHANNEL, J_POWERS, C_power, jammers, time_range,device, multi_channel=1, greedy=False, EPS_START=1.0,EPS_END=0.01,EPS_D=0.996):
  seeds = np.random.choice(range(10000),N_JAMMER)
  RLjammers = []
  for _ in range(N_JAMMER):
    if not greedy:
      print('Generating Agents')
      RLjammers.append(Agent(1,len(J_POWERS),N_CHANNEL,seeds[_],device=device,multi_channel=multi_channel))
    else:
      RLjammers.append(GreedyAgent(1,len(J_POWERS),N_CHANNEL,seeds[_],multi_channel=multi_channel))

  eps = EPS_START
  state = np.array(env.step([[jammers[0].jam()]],t=0)) #initialize state with random jamming
  cum_rewards=[]
  real_cum_rewards=[]

  SINR=[]
  real_SINR=[]
  SNR=[]
  power=[]

  for t in tqdm(range(time_range)):
    actions=[]
    raw_actions=[]
    for i in range(N_JAMMER):
      raw_action = RLjammers[i].act(state,eps)[0]
      raw_actions.append(raw_action)
      # j_channel = raw_action//len(J_POWERS)
      # j_power = J_POWERS[raw_action%len(J_POWERS)]
      # action = [j_channel,j_power]

      action = raw_action_decode(raw_action,multi_channel=multi_channel,Num_Channel=env.num_channel(),Powers=J_POWERS)
      actions.append(action)
    power.append(sum([action[1] for indi_action in actions for action in indi_action]))
    SNR_t, SINR_t, real_SNR_t, real_SINR_t = env.step(actions,t)
    SINR.append(SINR_t)
    real_SINR.append(real_SINR_t)
    SNR.append(SNR_t)
    
    next_state = np.array(SINR_t)
    # Compute reward and record
    cum_reward=0
    real_cum_reward=0
    real_cum_reward += instant_reward(real_SNR_t,real_SINR_t)
    cum_reward += instant_reward(SNR_t, SINR_t)
    for i, indi_action in enumerate(actions):
      reward = instant_reward(SNR_t, SINR_t)
      for action in indi_action:
        # reward = 1/SINR_t-C_power*j_power; 
        reward +=  -C_power*action[1]
        cum_reward += -C_power*action[1]
        real_cum_reward += -C_power*action[1]
      RLjammers[i].step(state,raw_actions[i],reward,next_state)
    cum_rewards.append(cum_reward)
    real_cum_rewards.append(real_cum_reward)
    
    state = next_state
    eps = eps*EPS_D
  
  return cum_rewards, real_cum_rewards, sum(SINR,[]),sum(real_SINR,[]), SNR, power

def raw_action_decode(raw_action_value, multi_channel, Num_Channel, Powers):
  Num_Power = len(Powers)
  v = raw_action_value
  attacks =[]
  for i in range(multi_channel):
    #compute channel selection
    v_module=1
    for j in range(2*i,multi_channel*2):
      if j%2==0:
        v_module = v_module*Num_Channel
      else:
        v_module = v_module*Num_Power
    
    v_division = 1
    for j in range(2*i+1,multi_channel*2):
      if j%2==0:
        v_division = v_division*Num_Channel
      else:
        v_division = v_division*Num_Power
    channel =(v%v_module//v_division)

    #compute channel selection
    v_module=1
    for j in range(2*i+1,multi_channel*2):
      if j%2==0:
        v_module = v_module*Num_Channel
      else:
        v_module = v_module*Num_Power
    v_division = 1
    for j in range(2*i+1+1,multi_channel*2):
      if j%2==0:
        v_division = v_division*Num_Channel
      else:
        v_division = v_division*Num_Power
    power =(v%v_module//v_division)

    attacks.append([channel,Powers[power]])

  return attacks
