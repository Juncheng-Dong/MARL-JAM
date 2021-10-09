import gym
import numpy as np

ENVLIST=['constant','sweep','ar','pulse','random']

class CommEnv(gym.Env):
  def __init__(self,n_channel, powers, commu_type='constant', \
               seed=42,noise=0.1, constant_power=True, multi_channel = 1,fix_power=None):
    self.t = 0 #initialize time to be 0
    self.state = None
    self.n_channel = n_channel
    self.powers = powers
    self.noise=noise
    self.channel = np.random.choice(range(self.n_channel), multi_channel, replace=False)
    self.power = np.random.choice(self.powers, multi_channel)
    # if fix_power:
    #     self.power = np.array(fix_power).reshape(-1)
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
    # self.n_jammer = n_jammer #self.n_jammer is not called in this class
  
  def step(self,actions,t):
    '''
    Input: actions: subscribable object (list), actions[i] is action of j-th jammer
    '''
    #sender chooses channel and power at time t
    sender_action = self.sender(t)
    s_channel, s_power = sender_action

    SNR,SINR = self.compute_SINR(s_power,s_channel,actions,self.multi_channel)
    real_SNR,real_SINR = self.compute_SINR_real(s_power,s_channel,actions,self.multi_channel,scale=1/1.253)
    
    return SNR, SINR, real_SNR, real_SINR #list, list

  def compute_SINR_real(self,s_power,s_channel,actions,multi_channel,scale=0.5**0.5):
      jamming_power = np.zeros(self.multi_channel)
      SINR = []
      SNR = []
      
      pc_s = np.random.normal(0,0.5**0.5)**2 + np.random.normal(0,0.5**0.5)**2
      pc_j = [np.random.normal(0,0.5**0.5)**2 + np.random.normal(0,0.5**0.5)**2 for _ in range(len(actions))]
      for i in range(multi_channel):
          SNR.append(s_power[i]*pc_s/self.noise)
          #for action of each jammer
          for j,action in enumerate(actions):
              # for each attack choice [channel, power] of each jammer
              for attack in action:
                  j_channel, j_power = attack
                  if j_channel == s_channel[i]:
                      jamming_power[i] += j_power*pc_j[j]
          SINR.append(s_power[i]*pc_s/(self.noise+jamming_power[i]))
      # SINR = np.array(SINR).sum() # take mean over all occupied channels
      return SNR, SINR #list, list

  
  def compute_SINR(self,s_power,s_channel,actions,multi_channel):
      jamming_power = np.zeros(self.multi_channel)
      SINR = []
      SNR = []

      for i in range(multi_channel):
          SNR.append(s_power[i]/self.noise)
          #for action of each jammer
          for action in actions:
              # for each attack choice [channel, power] of each jammer
              for attack in action:
                  j_channel, j_power = attack
                  if j_channel == s_channel[i]:
                      jamming_power[i] += j_power
          SINR.append(s_power[i]/(self.noise+jamming_power[i]))
      # SINR = np.array(SINR).sum() # take mean over all occupied channels
      return SNR, SINR #list, list
    
  def constant_channel_selector(self,t):
    return [self.channel, self.power]

  def random_channel_selector(self,t): #multi one doesn't consider channel conflicts
    channel = np.random.choice(range(self.n_channel), self.multi_channel, replace=False)
    power = np.random.choice(self.powers, self.multi_channel, replace=False)
    return [channel, power]
  
  #power changes randomly if not constant
  def sweep_channel_selector(self, t, constant_power = True):
    for i in range(self.multi_channel):
      self.channel[i] = i + t % self.n_channel
    
    if not self.constant_power:
      self.power = np.random.choice(self.powers, self.multi_channel, replace=False)
    # print(self.channel)
    return [self.channel, self.power]

  def ar_channel_selector(self, t, constant_power = True): #multi one doesn't consider channel conflicts
    for i in range(self.multi_channel):
      if self.channel[i] %2 == 0:
        self.channel[i] += (t % self.n_channel +i)
      else:
        self.channel[i] -= (t % self.n_channel +i)
      if self.channel[i] >= self.n_channel:
        self.channel[i] = np.random.choice([0, (self.n_channel-1)], p = [0.1, 0.9]) - i
      elif self.channel[i] < 0:
        self.channel[i] = np.random.choice([0, (self.n_channel-1)], p = [0.9, 0.1]) + i
      
      if not constant_power:
        self.power = np.random.choice(self.powers, self.multi_channel, replace=False)
    return [self.channel, self.power]

  def pulse_channel_selector(self, t, constant_power = True):
    for i in range(self.multi_channel):
      self.channel[i] = (self.n_channel-1-i) if t % self.n_channel <= self.n_channel // 2 else i
    
    if not constant_power:
      self.power = np.ranodm.choice(self.power, self.multi_channel, replace=False)
    return [self.channel, self.power]
  def num_channel(self):
    return self.n_channel
