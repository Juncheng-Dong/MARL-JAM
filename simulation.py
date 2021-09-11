def simulate_random(env, N_JAMMER, N_CHANNEL, J_POWERS, T_max):
  jammers=[]
  for i in range(N_JAMMER):
    jammers.append(RandomJammer(N_CHANNEL,J_POWERS))

  #Running Jamming and collect rewards
  rewards = []
  SINR=[]
  for t in range(T_max):
    actions = [[jammer.jam() for jammer in jammers]]
    SNR_t, SINR_t, _, real_SINR_t = env.step(actions)
    SINR.append(real_SINR_t)
    
    # rewards.append([-C_power*j_action[1] for j_action in actions]+[1/SINR_t])
    reward = instant_reward(SNR_t, SINR_t)
    cost = sum([-C_power*j_action[0][1] for j_action in actions])
    rewards.append(reward + cost)
  return jammers, rewards, sum(SINR, [])