import gym
import numpy as np
import random
import matplotlib.pyplot as plt

env = gym.make('MiniGrid-Empty-6x6-v0')
#env = gym.make('MiniGrid-Empty-6x6-v0', render_mode='human')

alpha = 0.5
lmbda = 0.6
gamma = 0.9
ne = 200
action_value = {}
reward_func = []
epi = []
nos = []
Eli = {}

max_steps = 700
for z in range(1,5):
            for j in range(1,5):
                for d in range(3):            
                    action_value[(z,j,d)] = np.zeros(3)
                    Eli[(z,j,d)]= np.zeros(3)

for i in range(ne):
    
    e = (-1*i/ne) + 1
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

    #print(action)    

    while not done:
        next_state, reward, done, _, infor = env.step(action)
        obs = next_state
        nextagent_position = env.agent_pos

        x= nextagent_position[0]
        y= nextagent_position[1]
        direc = obs.get('direction')
        
        nextstate = tuple((x,y,direc))

        if nextstate not in action_value:
            action_value[nextstate] = np.zeros(3)
 
        if np.random.uniform(0,1) < e :
            nextaction = random.randint(0,2)
        else:
            nextaction = np.argmax(action_value.get(nextstate))

        #print(action_value)
        delta = reward + gamma*(action_value[nextstate][nextaction]) -  action_value[state][action] 

        if state not in Eli:
            Eli[state] = np.zeros(3)

        Eli[state][action] += 1 

        for z in range(1,5):
            for j in range(1,5):
                for d in range(3):
                    for a in range(3):
             
                        action_value[(z,j,d)][a] += alpha * delta * Eli[(z,j,d)][a]
                        Eli[(z,j,d)][a] *= gamma * lmbda
        
        state = nextstate
        action = nextaction

        rew += reward
        n_steps += 1
        if done or n_steps>max_steps:
            break 

    reward_func.append(rew)
    nos.append(n_steps)
    epi.append(i)    
    #print(reward_func)
    #print(nos)
    #print(epi)

plt.title("MiniGrid-Empty-6x6-v0 Using SARSA(Lambda) Algorithm")
#plt.ylabel("Number of Steps")
plt.xlabel("Number of Episodes")
plt.ylabel("Rewards")


plt.plot(epi,reward_func)
#plt.plot(epi,nos)
plt.show()


env.close()        