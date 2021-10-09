
import numpy as np
import matplotlib.pyplot as plt

def success_rate(SINR, env_name,maxSNR, N=100, plot =False,toprint=True,threshold=0.5):
  success_SINR = maxSNR*threshold
  overall_success_rate = sum(np.array(SINR)<success_SINR)/len(SINR)
  lastN_success_rate = sum(np.array(SINR[-N:])<success_SINR)/N
  
  if toprint:
    print('overll success rate of jamming under {} sender is:'.format(env_name), overall_success_rate)
    print('last {} success rate of jamming under {} sender is:'.format(N, env_name), lastN_success_rate)

  if plot:
    plt.figure(figsize=(20,5))
    success_SINR = max(SINR)*threshold
    plt.plot( (np.array(SINR))<success_SINR)
    plt.ylabel('Instant success of jamming under {} sender'.format(env_name))
    plt.yticks([0,1])
    plt.show()
  
  return overall_success_rate, lastN_success_rate

def smooth_reward(instant_reward, smooth_len=200):
  return [np.mean(instant_reward[0:smooth_len])]*smooth_len+[np.mean(instant_reward[i-smooth_len:i]) for i in range(smooth_len,len(instant_reward))]

