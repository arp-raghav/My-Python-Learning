
import numpy as np
from numpy.core.fromnumeric import argmax
from tqdm import tqdm
import matplotlib.pyplot as plt

# initialization of environment

start = [3, 0]
end = [3, 11]
left = 2
up = 0
right = 3
down = 1
Grid_Height = 4
Grid_width= 12
actions = [left, up, right, down]
eps = 0.1
gam = 1
alp = 0.5




#This function takes the current Q values and state and, based on the epsilon greedy policy, proposes an action.
def dec_action(Q_Value, state):

# Selecting a random value to determine whether to explore or exploit while comparing it to the epsilon value.
    prob = np.random.rand()
    # If chosen random value is less than epsilon then explore
    if prob < eps:  
        return int(np.random.choice(actions))
    # And if chosen random value is less than epsilon then exploit
    else:  
        values = Q_Value[:, state[0], state[1]]
        return int(np.random.choice([action for action, value in enumerate(values) if value == np.max(values)]))

#This function takes the current state and action and returns the next state and reward based on it.
def step(state, action):
 
# max and min function is used in below if statements, so that agent doesn't fall off the grid
    i, j = state
    if action == up:
        nxt_state= [max(0, i - 1), j]
    elif action == down:
        nxt_state= [min(Grid_Height - 1, i + 1), j]
    elif action == left:
        nxt_state= [i, max(0, (j - 1))]
    elif action == right:
        nxt_state= [i, min(Grid_width- 1, (j + 1))]

    reward = -1
# This if statement takes care of penalizing the agent with -100 if it falls off the grid
    if (action == down and i == 2 and 1 <= j and j <= 10) or (action == right and i == 3 and j == 0):
        reward = -100
        nxt_state= start

    return nxt_state, reward




def q_learn(Q_QLearnig, ep, rewardsQ_learn):
    #This function directs the agent to travel across the grid and then updates the Q Values based on the Q-Learning technique for
#the specified number of cycles. It also keeps track of the cumulative reward that the agent receives for each episode.
   
    for i in range(ep):
        state = start
# This loop  end  when agent has completed the episode and has reached the goal state
        while state != end:
            action = dec_action(Q_QLearnig, state)
            nxt_state, reward = step(state, action)
            rewardsQ_learn[i] += reward
# Computing the Q values using QQ learning approach
            Q_QLearnig[action, state[0], state[1]] += alp * (
                        reward + (gam * np.max(Q_QLearnig[:, nxt_state[0], nxt_state[1]])) - Q_QLearnig[
                    action, state[0], state[1],])
            state = nxt_state

def sarsa(Q_Sarsa, ep, Reward_sarsa):

#This function directs the agent's movement in the grid and then updates the Q Values based on the SARSA method for a
#certain number of ep. It also keeps track of the cumulative reward that the agent receives for each episode.
    for i in range(ep):
        state = start
# This loop  end  when agent has completed the episode and has reached the goal state
        while state != end:
            action = dec_action(Q_Sarsa, state)
            nxt_state, reward = step(state, action)
            Reward_sarsa[i] += reward
            nextAction = dec_action(Q_Sarsa, nxt_state)
# calculating Q values
            Q_Sarsa[action, state[0], state[1]] += alp * (
                        reward + gam * Q_Sarsa[nextAction, nxt_state[0], nxt_state[1],] - Q_Sarsa[
                    action, state[0], state[1]])
            state = nxt_state

itr = 100
ep = 500
# Initialising of rewards arrays
Reward_sarsa = np.zeros(ep)
rewardsQ_learn = np.zeros(ep)
for i in tqdm(range(itr)):
    # Initializing of Q Values for each iteration
    Q_Sarsa = np.zeros((len(actions), Grid_Height, Grid_width))
    Q_QLearnig = np.zeros((len(actions), Grid_Height, Grid_width))
    # Calling of SARSA and Q_learn function
    sarsa(Q_Sarsa, ep, Reward_sarsa)
    q_learn(Q_QLearnig, ep, rewardsQ_learn)

# summing the rewards on number of itr
Reward_sarsa /= itr
rewardsQ_learn /= itr
sarsa_smooth = []
smooth_Q_learn = []

# smoothing the rewards on moving average of 10
for i in range(0, 500, 10):
    sarsa_smooth.append(sum(Reward_sarsa[i:i + 10]) / 10)
    smooth_Q_learn.append(sum(rewardsQ_learn[i:i + 10]) / 10)

#This function accepts the Q Values and prints the opt policy based on the Q Values.
def opt_policy(QValue):
  
    policy = []
    for i in range(QValue.shape[1]):
        row = []
        for j in range(QValue.shape[2]):
            if (i == 3 and j == 11):
                row.append("G")
            elif (i == 3 and 0 < j and j < 11):
                row.append("-")
            else:
                max = argmax(QValue[:, i, j])
                if max == 0:
                    row.append("U")
                elif max == 1:
                    row.append("D")
                elif max == 2:
                    row.append("L")
                elif max == 3:
                    row.append("R")

        policy.append(row)

    for row in policy:
        print(row)


# Plotting the graph
x = range(0, 500, 10)
plt.plot(x, sarsa_smooth, color="r", label="Sarsa")
plt.plot(x, smooth_Q_learn, color="b", label="Q learning")
plt.yticks([-25, -50, -75, -100])
plt.legend()
plt.xlabel('Ep')
plt.ylabel('Sum of rewards during episode')
plt.xlim([0, 500])
plt.ylim([-100, -20])
plt.show()
plt.close()


'''
Submitted By
1. Arpit Raghav(201584085)
2. Ramit Magan(201597470)
3. Shivam Nareshkumar Sharma(201598809)
4. Keerthi Raj Shashikala Rajanna(201598719)
'''
