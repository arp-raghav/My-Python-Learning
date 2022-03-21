
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import argmax
from tqdm import tqdm
#matplot inline


class Multiarm_B:
#Initialising parameters to the environment
    def __init__(slf, arms = 10):
        slf.mu =  np.random.normal(0,1,arms)
        slf.n_arms = arms
        slf.n_count= np.zeros(arms)
        slf.Unique_SUM = np.zeros(arms)
        slf.Q_val = np.zeros(arms)

class Agnt():
#Initialising parameters to the Agent
    def __init__(slf, eps, envmt, play = 1000):
        slf.envmt = envmt
        slf.play = play
        slf.eps = eps
        slf.Increasing_Reward = np.zeros(play)
        slf.Action = np.zeros(play)


#This method pulls the arm in accordance with the epsilon greedy policy with the specified epsilon value and returns the arm as well as the reward for pulling that arm.
    def pull(slf):
      
#Using a random value to determine whether to explore or exploit while comparing with the epsilon value.
        prob = np.random.rand()
        #If chosen random value is less than epsilon then explore
        if prob < slf.eps: 
            a = np.random.choice(slf.envmt.n_arms)
            #And if chosen random value is less than epsilon then exploit
        elif prob > slf.eps: 
            a = argmax(slf.envmt.Q_val)
        reward = np.random.normal(slf.envmt.mu[a] , 1, 1)

        #Updating the mean of arm being pulled
        slf.envmt.n_count[a] += 1
        slf.envmt.Unique_SUM[a] += reward
        slf.envmt.Q_val[a] = slf.envmt.Unique_SUM[a] / slf.envmt.n_count[a] 
        return reward, a
        
    def run(slf):
# This method pulls the arm for given number of episodes, saves the cumulative reward and at the end of all episodes, reinitialise the envmt      
        sum = 0
        for i in range(slf.play):
            reward, a = slf.pull()
            sum += reward
            slf.Increasing_Reward[i] += (sum/(i+1))
            #checking if the optimal action was taken
            if a == argmax(slf.envmt.mu): 
                slf.Action[i] += 1
        slf.reset()

    def reset(slf):
        slf.envmt = Multiarm_B()
        


env = Multiarm_B()
#initialisation of Agnts
greedy = Agnt(0, env)
eGreedy_01 = Agnt(0.1, env)
eGreedy_001 = Agnt(0.01, env)
itr= 2000
#Running the experiment for 2000 itr
for i in tqdm(range(itr)):
    greedy.run()
    eGreedy_01.run()
    eGreedy_001.run()
   

#Calcuting the percentage optimal action
greedy.Action *= (100/itr)
eGreedy_01.Action *= (100/itr)
eGreedy_001.Action *= (100/itr)

#Averaging the reward over number of itr
greedy.Increasing_Reward /= itr
eGreedy_01.Increasing_Reward /= itr
eGreedy_001.Increasing_Reward /= itr



#plotting of graphs

plt.plot(greedy.Increasing_Reward, color = "g", label = "Greedy")
plt.plot(eGreedy_01.Increasing_Reward, color = "r", label = "ε-Greedy with ε =0.1")
plt.plot(eGreedy_001.Increasing_Reward, color = "b", label = "ε-Greedy with ε =0.01")
plt.ylim(0,1.5)
plt.xlabel("Play")
plt.ylabel("Average Reward")
plt.legend()
plt.savefig('Q12.png', dpi=300, bbox_inches='tight')
plt.show()



plt.plot(greedy.Action, color = "g", label = "Greedy")
plt.plot(eGreedy_01.Action, color = "r", label = "ε-Greedy with ε =0.1")
plt.plot(eGreedy_001.Action, color = "b", label = "ε-Greedy with ε =0.01")
plt.ylim(0,100)
plt.xlabel("Play")
plt.ylabel("Optimal Action %")
plt.legend()
plt.savefig('Q11.png', dpi=300, bbox_inches='tight')
plt.show()



'''
Submitted By
1. Arpit Raghav(201584085)
2. Ramit Magan(201597470)
3. Shivam Nareshkumar Sharma(201598809)
4. Keerthi Raj Shashikala Rajanna(201598719)
'''
