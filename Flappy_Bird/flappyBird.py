
import random
import time
import numpy as np
import pygame
import matplotlib.pyplot as plt
import flappy_bird_gym
env = flappy_bird_gym.make("FlappyBird-v0",pipe_gap=150)

b=0
prev_hfp=0.00
alpha = 0.4
gamma = 0.95
ne = 500000
action_value = {}
reward_func = []
score_func = []
epi = []
nS = 3000
Score = 0
nop = 0
prev_Score = 0
prev_nop=0
max_steps =10000

'''
num_height_bins = nS
num_distance_bins = nS


min_height, max_height = -0.22, 0.55
min_distance, max_distance = 0.0,1.8 ''' 

mv_size = 1000
#Calculating simple moving average
def sma(array, mv_size):
    moving_averages = []
    for i in range(len(array) - mv_size + 1):
        window = array[i:i + mv_size]
        avg = np.mean(window)
        moving_averages.append(avg)

    return moving_averages

'''def discretize_state(height, distance):
    height_bin = int((height - min_height) / (max_height - min_height) * num_height_bins)
    distance_bin = int((distance - min_distance) / (max_distance - min_distance) * num_distance_bins)


    height_bin = np.clip(height_bin, 0, num_height_bins - 1)
    distance_bin = np.clip(distance_bin, 0, num_distance_bins - 1)

    return distance_bin, height_bin'''




for i in range(ne):
    e = (-1*i/ne) + 1
    obs = env.reset()
    n_steps = 0
    rew = 0

    
  
    #env.render()
    done = False

    
    
    while not done:
       
        
        x= obs[0]
        y= obs[1]
        state = tuple((x,y))     
        #print(state)
        if state not in action_value:
            action_value[state] =[0.0,0.0]

        if np.random.uniform(0,1) < e :
            action = random.randint(0,1)
        else:
            action = np.argmax(action_value.get(state))
            # print(action)    

        next_obs, reward, done,info = env.step(action)
        reward = 1
        #Updating reward
        nop = info['score']
        if info['score'] > prev_nop:
            reward += 5

        if done:
            reward -=10

        if b < nop:
            b=nop


        x1= next_obs[0]
        y1= next_obs[1]  

        nextstate = tuple((x1,y1))  

        if nextstate not in action_value:
            action_value[nextstate] = [0.0,0.0]
   
        prev_nop = nop
        

        action_value[state][action] += alpha * (reward + gamma*np.max(action_value[nextstate]) - action_value[state][action])

    
        state = nextstate
        obs = next_obs

        rew += reward
        n_steps += 1
        if done or max_steps == n_steps:
            break 
    if i%10000 == 0:
        print(i)
    reward_func.append(rew)
    score_func.append(nop)
    epi.append(i)
    if i == 100000 or i==300000 or i == 500000: 
        plt.title("Flappy_Bird-v0 Using Q-Learning Algorithm")
        plt.xlabel("Number of Episodes")
        plt.ylabel("Rewards")
        plt.plot(epi,reward_func)
        plt.plot(sma(epi,mv_size),sma(reward_func,mv_size))
        plt.show()

        plt.title("Flappy_Bird-v0 Using Q-Learning Algorithm")
        plt.xlabel("Number of Episodes")
        plt.ylabel("Score")
        plt.plot(epi,score_func)
        plt.plot(sma(epi,mv_size),sma(score_func,mv_size))
        plt.show()   
print(b)



score_func2 = []
reward_func2 = []
note = []
max_steps =20000
mv_size = 50

for i in range(1000):
    rew=0
    obs = env.reset()
    score = 0
    x= obs[0]
    y= obs[1]    
    state = tuple((x,y))   

    while True:
        #env.render()
        if state in action_value:
            action = np.argmax(action_value.get(state))

        next_obs, reward, done,info = env.step(action)

        x= next_obs[0]
        y= next_obs[1]    
        nextstate = tuple((x,y)) 
        state = nextstate
        score = info['score']
        
        rew += reward
        if done or score == 300:
            #print("Crashed")
            break

    print("Score:",score)
    reward_func2.append(rew)
    score_func2.append(score)
    note.append(i)


plt.title("Flappy_Bird-v0 Using Q-Learning Algorithm")
plt.xlabel("Number of Test Episodes")
plt.ylabel("Rewards")
plt.plot(note,reward_func2)
plt.plot(sma(note,mv_size),sma(reward_func2,mv_size))
plt.show()

plt.title("Flappy_Bird-v0 Using Q-Learning Algorithm")
plt.xlabel("Number of Test Episodes")
plt.ylabel("Score")
plt.plot(note,score_func2)
plt.plot(sma(note,mv_size),sma(score_func2,mv_size))
plt.show()


env.close()        
