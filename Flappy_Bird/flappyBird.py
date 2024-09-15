import time
import random
import pygame
import flappy_bird_gym
import numpy as np
import matplotlib.pyplot as plt
env = flappy_bird_gym.make("FlappyBird-v0")

obs = env.reset()

#Q Learning Algorithm is usedhere

def epsilon_greedy(epsilon,state_space):
    if np.random.rand() < epsilon:
        action = np.random.randint(0,2)
    else:
        action = np.argmax(Q[state_space])
    return action

# for calculating Simple Moving Average
def calculate_SMA(data,size_of_window):                
    SMA = []
    for i in range(len(data)):
        if i < size_of_window:
            SMA.append(np.mean(data[:i+1]))
        else:
            SMA.append(np.mean(data[i-size_of_window+1 : i+1]))
    return SMA           

Q = {}

'''DEFINING HYPERPARAMETER'''
number_of_episodes = 500000
epsilon = 1
alpha = 0.4
gamma = 0.9
reward = 0


reward_function = []
score_function = []
test_episodes =[]
test_rew_episode = []
test_scores = []
ToTal_Episodes = []


for i in range(number_of_episodes+5000):
    obs = env.reset()
    
    state_space = (obs[0],obs[1])
    done = False
    policy = []

    re_per_episd = 0
    sc_per_episd = 0
    test_rew = 0
    scores_in_test = 0

    steps_bird = 1
    max_steps_of_bird = 0
    prev_score = 0
    if i%5000==0:
        print(f'{i} - episode')
    if i > number_of_episodes:
        print(f'{i} - episode') 
    while not done:
        
        
        if state_space not in Q:
            Q[state_space] = np.zeros(2,dtype=float)

        action = epsilon_greedy(epsilon,state_space)
        policy.append(action)
        obs2, reward, done, info = env.step(action)    
        new_state_space = (obs2[0],obs2[1])
        score = info['score']
        if prev_score < score:
            reward = reward + 5
          


        if new_state_space not in Q:
            Q[new_state_space] = np.zeros(2,dtype=float)

        maxof_next_action = np.argmax(Q[new_state_space])
        
        re_per_episd += reward
        sc_per_episd += score
        if i > number_of_episodes:
            test_rew += reward
            scores_in_test += score
            


        Q[state_space][action] = Q[state_space][action] + alpha * (reward + gamma*(Q[new_state_space][maxof_next_action]) - Q[state_space][action])
        state_space = new_state_space
        epsilon = ((-i)/number_of_episodes) + 1
        prev_info = info
        steps_bird += 1
        if done: 
            env.reset()

        if max_steps_of_bird == 1000:
            break

        max_steps_of_bird += 1 



    if i > number_of_episodes:
        test_episodes.append(i-number_of_episodes)
        test_rew_episode.append(test_rew)
        test_scores.append(scores_in_test)
        
    
    score_function.append(sc_per_episd)
    reward_function.append(re_per_episd)  
    ToTal_Episodes.append(i)    



plt.figure(figsize=(12,6))

SMA_window_size = 100

SMA_reward = calculate_SMA(reward_function,SMA_window_size)
# Plot the reward_function and its SMA
plt.subplot(2,2,1)
plt.plot(ToTal_Episodes, reward_function, label='Rewards')
plt.plot(ToTal_Episodes, SMA_reward, label=f'SimpleMovingAvg(wind_size-{SMA_window_size})')
plt.title('FlappyBird-v0 using Q-Learning')
plt.xlabel('no of episodes')
plt.ylabel('Rewards')
plt.legend()


SMA_score = calculate_SMA(score_function,SMA_window_size)
# Plot the score_function and its SMA
plt.subplot(2,2,2)
plt.plot(ToTal_Episodes,score_function,label='Scores')
plt.plot(ToTal_Episodes,SMA_score,label=f'SimpleMovingAvg(wind_size-{SMA_window_size})')
plt.title('FlappyBird-v0 using Q-Learning')
plt.xlabel('no of episodes')
plt.ylabel('Scores')
plt.legend()


SMA_test_rewards = calculate_SMA(test_rew_episode,SMA_window_size)
# Plot the rewards in test episode and its SMA
plt.subplot(2,2,3)
plt.plot(test_episodes,test_rew_episode,label='TestRewards')
plt.plot(test_episodes,SMA_test_rewards,label=f'SimpleMovingAvg(wind_size-{SMA_window_size})')
plt.title('FlappyBird-v0 using Q-Learning')
plt.xlabel('No. of Test episodes')
plt.ylabel('Test rewards')
plt.legend()


SMA_test_scores = calculate_SMA(test_scores,SMA_window_size)
# Plot the scores in test episode and its SMA
plt.subplot(2,2,4)
plt.plot(test_episodes,test_scores,label='TestScores')
plt.plot(test_episodes,SMA_test_scores,label=f'SimpleMovingAvg(wind_size-{SMA_window_size})')
plt.title('FlappyBird-v0 using Q-Learning')
plt.xlabel('No. of Test episodes')
plt.ylabel('Test Scores')
plt.legend()

plt.tight_layout()

plt.show()

for i in range(1):
    
    obs = env.reset()
    state_s = (obs[0],obs[1])
    done=False
    score=0
    while not done:
        env.render()
        if state_s in Q:
            action = np.argmax(Q[state_s])
        else:
            print('random')
            action = random.randint(0,1)    
        n_obs, rew, done,info = env.step(action)
        n_state_s = (n_obs[0],n_obs[1])
        time.sleep(1/30)
        state_s = n_state_s
        score = info['score']
        if done:
            print('damaged')
            break
print(score)



env.close()