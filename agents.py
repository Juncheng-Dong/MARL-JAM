import numpy as np

class RandomJammer():
  def __init__(self,n_channel,powers):
    self.n_channel = n_channel
    self.powers = powers

  def jam(self):
      '''
      action function of a random jammer
      randomly pick one channel and one power level
      return [channel#, power#]
      '''
      channel = np.random.choice(range(self.n_channel))
      power = np.random.choice(self.powers)

      return [channel,power]

class GreedyAgent():
  '''
  OOP for greedy jammer that thinks jamming as Multi-Armed Bandit problem
  '''
  def __init__(self, state_size, NUM_POWERS, NUM_CHANNEL, seed, multi_channel=1):
    #compute action size
    action_size=1
    for _ in range(multi_channel):
      action_size = NUM_POWERS*action_size
      action_size = NUM_CHANNEL*action_size
    self.action_size = action_size

    self.reward_record = np.zeros(action_size)
    self.count_record = np.zeros(action_size)
  
  def act(self,state,eps):
    actions = np.flatnonzero(self.reward_record == np.max(self.reward_record))
    return actions[np.random.choice(len(actions))], np.max(self.reward_record)
  
  def step(self, state, action, reward, next_step, done=False):
    
    self.reward_record[action] = (self.reward_record[action]*self.count_record[action]+reward)/(self.count_record[action]+1)
    # self.reward_record[action] = 1
    self.count_record[action] = self.count_record[action]+1
    