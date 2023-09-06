import gym
import numpy as np
import random
import matplotlib.pyplot as plt

env = gym.make('MiniGrid-Empty-6x6-v0')
# env = gym.make('MiniGrid-Empty-6x6-v0', render_mode='human')

alpha = 0.4
gamma = 0.9
ne = 150
#value = np.zeros(3)
action_value = {}

reward_func = []
epi = []
nos = []
max_steps = 600
for i in range(ne):
    obs = env.reset()
    n_steps = 0
    rew = 0
    episode = [] 
    e = (-1*i/ne) + 1
    env.render()
    done = False

    while not done:
        state = env.agent_pos   
        x= state[0]
        y=state[1]
        try:   
            direc = obs[0].get('direction')
            state = tuple((x,y,direc)) 
        except:
            direc = obs.get('direction')
            state = tuple((x,y,direc))   
        
        info = tuple((x,y,direc))

        if info not in action_value:
            action_value[info] = np.zeros(3)

        if np.random.uniform(0,1) < e :
            action = random.randint(0,2)
            
        else:
            action = np.argmax(action_value.get(info))
            

            
        # print(action)
        next_state, reward, done, _, infor = env.step(action)
        #if reward!=0:
           # print(reward, n_steps,i)
        if reward <0:
            reward = 0   
        episode.append((next_state,info,action,reward))
        
        n_steps += 1
        rew += reward
        
        obs = next_state
        nextagent_position = env.agent_pos

        x= nextagent_position[0]
        y= nextagent_position[1]
        direc = obs.get('direction')
        
        nextstate = tuple((x,y,direc))

        info = nextstate

        if n_steps>max_steps:
            done = True
    # print(rew,reward)
    reward_func.append(rew)
    nos.append(n_steps)
    epi.append(i)
 
   
              
    G = 0  
    for t in reversed(range(len(episode))):
        state, info,action, reward = episode[t]
        G = gamma * G + reward
        action_value[info][action] += alpha * (G - action_value[info][action])
        #print(action_value)

plt.title("MiniGrid-Empty-6x6-v0 Using Monte-Carlo Algorithm")
plt.ylabel("Number of Steps")
plt.xlabel("Number of Episodes")
#plt.ylabel("Rewards")


#plt.plot(epi,reward_func)
plt.plot(epi,nos)

plt.show()


env.close()

