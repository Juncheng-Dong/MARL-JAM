import gym
import numpy as np

def instant_reward(SNR, SINR):
  gain = [10*np.log2(1.+x) - 10*np.log2(1.+y) for x, y in zip(SNR, SINR)]
  return sum(gain)


class CommEnv(gym.Env):
  def __init__(self,n_channel, powers, commu_type='constant', \
               seed=42,noise=0.1, constant_power=True, multi_channel = 1):
    self.t = 0 #initialize time to be 0
    self.state = None
    self.n_channel = n_channel
    self.powers = powers
    self.noise=noise
    self.channel = np.random.choice(range(self.n_channel), multi_channel, replace=False)
    self.power = np.random.choice(self.powers, multi_channel, replace = False)
    self.constant_power = constant_power
    self.multi_channel = multi_channel

    if commu_type == 'constant':
      print(f"Constant Sender with: [{self.channel},{self.power}]")
      self.sender = self.constant_channel_selector
    elif commu_type == 'sweep':
      self.sender = self.sweep_channel_selector
    elif commu_type == 'ar':
      self.sender = self.ar_channel_selector
    elif commu_type == 'pulse':
      self.sender = self.pulse_channel_selector
    elif commu_type == 'random' :
      self.sender = self.random_channel_selector
    else:
      raise ValueError('Sender type is not support.')
    
    if constant_power:
      print(self.power)
    self.seed=seed # seed for random channel selection

    def step(self,actions):
        '''
        Input: actions: subscribable object (list), actions[i] is action of j-th jammer
        '''
        #sender chooses channel and power at time t
        sender_action = self.sender(self.t)
        s_channel, s_power = sender_action

        jamming_power = np.zeros(self.multi_channel)
        
        SNR,SINR = self.compute_SINR(actions,self.multi_channel)
        real_SNR,real_SINR = self.compute_SINR_real(actions,self.multi_channel)
        
        return SNR, SINR, real_SNR, real_SINR #list, list

    def compute_SINR_real(self,actions,multi_channel):
        jamming_power = np.zeros(self.multi_channel)
        SINR = []
        SNR = []

        for i in range(multi_channel):
            SNR.append(s_power[i]*np.random.rayleigh(scale=0.5**0.5)/self.noise)
            #for action of each jammer
            # print(actions)
            for action in actions:
                # for each attack choice [channel, power] of each jammer
                for attack in action:
                    j_channel, j_power = attack
                    if j_channel == s_channel[i]:
                        jamming_power[i] += j_power
            SINR.append(s_power[i]/(self.noise+jamming_power[i]*np.random.rayleigh(scale=0.5**0.5)))
        # SINR = np.array(SINR).sum() # take mean over all occupied channels
        return SNR, SINR #list, list

    
    def compute_SINR(self,actions,multi_channel):
        jamming_power = np.zeros(self.multi_channel)
        SINR = []
        SNR = []

        for i in range(multi_channel):
            SNR.append(s_power[i]/self.noise)
            #for action of each jammer
            # print(actions)
            for action in actions:
                # for each attack choice [channel, power] of each jammer
                for attack in action:
                    j_channel, j_power = attack
                    if j_channel == s_channel[i]:
                        jamming_power[i] += j_power
            SINR.append(s_power[i]/(self.noise+jamming_power[i]))
        # SINR = np.array(SINR).sum() # take mean over all occupied channels
        return SNR, SINR #list, list

