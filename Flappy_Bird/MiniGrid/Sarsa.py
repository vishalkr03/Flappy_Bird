import gym
import numpy as np
import random
import matplotlib.pyplot as plt

env = gym.make('MiniGrid-Empty-6x6-v0')
# env = gym.make('MiniGrid-Empty-6x6-v0', render_mode='human')

alpha = 0.5
gamma = 0.9
ne =100
#value = np.zeros(3)
action_value = {}
reward_func = []
epi = []
nos = []

max_steps = 1000

for i in range(ne):
    e = (-1*i/ne) + 1
  #  if i>ne*0.9:
   #     env = gym.make('MiniGrid-Empty-6x6-v0',render_mode='human')
    obs = env.reset()
    n_steps = 0
    rew = 0
    env.render()
    done = False
    


    agent_position = env.agent_pos   
    x= agent_position[0]
    y= agent_position[1]    
    direc = obs[0].get('direction')

    state = tuple((x,y,direc))  

    if state not in action_value:
        action_value[state] = np.zeros(3)

    if np.random.uniform(0,1) < e :
        action = random.randint(0,2)
    else:
        action = np.argmax(action_value.get(state))

   # print(action)    

    while not done:
        next_state, reward, done, _, infor = env.step(action)
       
        obs = next_state
        nextagent_position = env.agent_pos

        x= nextagent_position[0]
        y= nextagent_position[1]
        direc = obs.get('direction')
        
        nextstate = tuple((x,y,direc))
        # print(nextstate)

        if nextstate not in action_value:
            action_value[nextstate] = np.zeros(3)
 
        if np.random.uniform(0,1) < e :
            nextaction = random.randint(0,2)
        else:
            nextaction = np.argmax(action_value.get(nextstate))   

        action_value[state][action] += alpha * (reward + gamma*action_value[nextstate][nextaction] - action_value[state][action])
       # print(action_value)
        action = nextaction
        state=nextstate
        rew += reward
        n_steps += 1
        if done or n_steps>max_steps:
            break 
    #print(rew,n_steps)
    reward_func.append(rew)
    nos.append(n_steps)
    epi.append(i)




#print(reward_func)
# print()
plt.title("MiniGrid-Empty-6x6-v0 Using SARSA Algorithm")
#plt.ylabel("Number of Steps")
plt.ylabel("Rewards")
plt.xlabel("Number of Episodes")

#plt.plot(epi,reward_func)
plt.plot(epi,nos)

plt.show()


env.close()        