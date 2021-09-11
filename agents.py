import numpy as np

class RandomJammer():
  def __init__(self,n_channel,powers):
    self.n_channel = n_channel
    self.powers = powers

  def jam(self):
    
    channel = np.random.choice(range(self.n_channel))
    power = np.random.choice(self.powers)

    return [channel,power]