'''
Please make sure that you have below listed libraries pre installed on your system
    * Matplotlib
    * Gym
        * Box2D
        * Swig (If Prompted)
    * NumPy
    * Time
    * Torch
    * Tensor

'''


from matplotlib import colors
import gym
import numpy as np
import time

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

import matplotlib.pyplot as plt


def plotLearning(x, scores, epsilons, figName):
    """
    This Method takes the scores and epsilon history as parameters and plot the their
    plots vs epsisode
    """

    # Computing the moving average
    movingAverage = np.empty(len(scores))
    for t in range(len(scores)):
        movingAverage[t] = np.mean(scores[max(0, t - 20):(t + 1)])

    # Plotting the Graphs
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax2.plot(x, movingAverage, label="Scores", color="g")
    ax1.plot(x, epsilons, label="Epsilon", color="r")
    ax1.set_ylabel("Epsilon", color="r")
    ax1.set_xlabel("Episode", color="r")
    ax2.set_ylabel('Score', color="g")
    ax1.tick_params(axis='x', colors="r")
    ax2.tick_params(axis='y', colors="g")
    ax1.tick_params(axis='y', colors="r")
    ax2.yaxis.set_label_position('right')
    ax2.yaxis.tick_right()
    ax2.axes.get_xaxis().set_visible(False)
    plt.legend()
    plt.savefig(figName)


class DeepQNetwork(nn.Module):
    """
    This Class takes the architecture of network as the parameter and trains it

    Parameters:
        inputDimensions : Number of dimensions in the input layer
        fc1Dims : Number of dimensions in first hiddden layer
        fc2Dims : Number of dimensions in second hiddden layer
        numberOfActions : Number of dimensions of output layer
        type: whether to train the network or use only forward pass
    """

    def __init__(self, inputDimensions, fc1Dims, fc2Dims,
                 numberOfActions, type="training", learningRate=0.001):
        super(DeepQNetwork, self).__init__()
        # Initialisation of parameters
        self.fc1Dims = fc1Dims
        self.fc2Dims = fc2Dims
        self.inputDimensions = inputDimensions
        self.numberOfActions = numberOfActions
        self.fc1 = nn.Linear(self.inputDimensions, self.fc1Dims)
        self.fc3 = nn.Linear(self.fc2Dims, self.numberOfActions)
        self.fc2 = nn.Linear(self.fc1Dims, self.fc2Dims)

        # Checking if the agent is in training mode or online mode
        if type == "training":
            self.loss = nn.MSELoss()
            self.optimizer = optim.Adam(self.parameters(), lr=learningRate)

    def forward(self, state):
        """
        This method takes an in for neural network and  passes it through the forward pass
        and at the end returns the actions from output layer
        """
        x = F.relu(self.fc2(F.relu(self.fc1(state))))
        return self.fc3(x)


class Agent():
    """
    This class takes the hyperparameter for the agent and trains it
    Parameters:
        gamma : discount factor
        epsilon : proportion of exploration
        epsMin : minimum proportion of exploration
        epsDecay : decaying rate of proportion of exploration
        learningRate : learning rate
        inputDimensions : Number of dimensions in the input layer
        fc1Dims : Number of dimensions in first hiddden layer
        fc2Dims : Number of dimensions in second hiddden layer
        numberOfActions : Number of dimensions of output layer
        batchSize : size of the batch taken from the memory
        maxMemorySize : maximum size of the memory
        stateMemory : memory for states
        newStateMemory : memory for next states
        actionMemory : memory of actions
        rewardMemory : memory of rewards
        terminalMemory : memory of terminal states


    """

    def __init__(self, gamma, epsilon, learningRate, inputDimensions, batchSize, numberOfactions, fc1Dims=512,
                 fc2Dims=512,
                 maxMemorySize=100000, epsMin=0.05, epsDecay=0.001):
        # Initialisation of parameters
        self.epsilon = epsilon
        self.epsDecay = epsDecay
        self.actionSpace = [i for i in range(numberOfactions)]
        self.iterationCounter = 0
        self.memorySize = maxMemorySize
        self.gamma = gamma
        self.epsMin = epsMin
        self.memoryCounter = 0
        self.leaarningRate = learningRate
        self.batchSize = batchSize
        
        self.qEvaluation = DeepQNetwork(learningRate=learningRate, numberOfActions=numberOfactions,
                                        inputDimensions=inputDimensions,
                                        fc1Dims=fc1Dims, fc2Dims=fc2Dims)

        self.newStateMemory = np.zeros((self.memorySize, inputDimensions), dtype=np.float32)
        self.actionMemory = np.zeros(self.memorySize, dtype=np.int32)
        self.stateMemory = np.zeros((self.memorySize, inputDimensions), dtype=np.float32)
        self.terminalMemory = np.zeros(self.memorySize, dtype=np.bool)
        self.rewardMemory = np.zeros(self.memorySize, dtype=np.float32)

    def storeTransition(self, state, action, reward, state_, terminal):
        """
        This method stores the transitions in memory after each step
        parameters :
            state : current state
            action : action taken
            state_ : next state
            reward : reward collected
            terminal : whether the episode terminated or not (bool)
        """
        index = self.memoryCounter % self.memorySize
        self.terminalMemory[index] = terminal
        self.actionMemory[index] = action
        self.stateMemory[index] = state
        self.rewardMemory[index] = reward
        self.newStateMemory[index] = state_

        self.memoryCounter += 1

    def chooseAction(self, observation):
        """
        This method takes the observation of the environment and returns the action to be taken
        """
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.actionSpace)
        else:
            action = T.argmax(self.qEvaluation.forward(T.tensor([observation]))).item()

        return action

    def learn(self):
        """
        This method holds the whole lerning process of the agent
        """
        if not(self.memoryCounter > self.batchSize):
            return

        # Getting a batch from the memory
        maxMemory = min(self.memoryCounter, self.memorySize)


        batch = np.random.choice(maxMemory, self.batchSize, replace=False)
        
        # making the gradients equal to zero
        self.qEvaluation.optimizer.zero_grad()

        batchIndex = np.arange(self.batchSize, dtype=np.int32)

        # Getting the batch of state, new state, reward, action, and termination information from memory
        stateBatch = T.tensor(self.stateMemory[batch])
        terminalBatch = T.tensor(self.terminalMemory[batch])
        rewardBatch = T.tensor(self.rewardMemory[batch])
        actionBatch = self.actionMemory[batch]
        newStateBatch = T.tensor(self.newStateMemory[batch])

        # Getting the actions from the network
        qEvaluation = self.qEvaluation.forward(stateBatch)[batchIndex, actionBatch]
        qNext = self.qEvaluation.forward(newStateBatch)
        qNext[terminalBatch] = 0.0

        # Calculating the target Q value through Q learning
        qTarget = rewardBatch + self.gamma * T.max(qNext, dim=1)[0]

        # optimising the network
        loss = self.qEvaluation.loss(qTarget, qEvaluation)
        loss.backward()
        self.qEvaluation.optimizer.step()

        self.iterationCounter += 1
        # Decaying the exploration rate
        self.epsilon = self.epsilon - self.epsDecay if self.epsilon > self.epsMin \
            else self.epsMin


def trainTheAgent():
    """
    This method accepts the user's input parameters, initialises the agent and environment, and guides the agent through the environment. Finally, it creates a graph to depict the agent's learning process.
    """

    # Taking hyperparameters from the user
    print("\nEnter the following hyperparameters\n")
    learningRate = float(input("Enter the Learning Rate : "))
    episodes = int(input("Enter the number of episodes : "))
    gamma = float(input("Enter the value of gamma : "))
    epsDecay = float(input("Enter the value of Epsilon Decay : "))
    fc1Dims = int(input("Enter the dimensions of first fully connected hidden layer : "))
    fc2Dims = int(input("Enter the dimensions of second fully connected hidden layer : "))

    # Initalising the environment
    env = gym.make('LunarLander-v2')
    stateSize = env.observation_space.shape[0]
    actionSize = env.action_space.n

    # initialising the agent
    agent = Agent(gamma=gamma, epsilon=1.0, batchSize=64, numberOfactions=actionSize, epsMin=0.01,
                  inputDimensions=stateSize, learningRate=learningRate, epsDecay=epsDecay, fc1Dims=fc1Dims,
                  fc2Dims=fc2Dims)
    scores, epsHistory = [], []

    # Running through each episode
    for i in range(episodes):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.chooseAction(observation)  # chosing the action
            observation_, reward, done, info = env.step(action)  # step into next state
            score += reward
            agent.storeTransition(observation, action, reward,  # storing transition in the memory
                                  observation_, done)
            agent.learn()  # learning through experience
            observation = observation_

        # taking account of learning
        scores.append(score)
        epsHistory.append(agent.epsilon)

        averageScore = np.mean(scores[-100:])

        print('Episode ', i, 'score %.2f' % score, 'average score %.2f' % averageScore, 'epsilon %.2f' % agent.epsilon)

    # plotting the graph
    x = [i + 1 for i in range(episodes)]
    figName = 'lunar_lander.png'
    plotLearning(x, scores, epsHistory, figName)
    # Saving the agent if asked
    save = input("\nEnter 1 if you want to save the Agent otherwise press any other key : ")
    if save == "1":
        filename = input("Enter the file name : ")
        T.save(agent.qEvaluation.state_dict(), filename)


def loadAgent(stateSize, actionSize):
    """
    This method takes input and output dimension of the network, ask for the architecture of
    the network, loads the trained network and retrns it
    """
    print("Enter the architecture of the network\nThe hidden units for our trained model were 512 units each layer")
    fc1Dims = int(input("Enter the dimensions of first fully connected hidden layer : "))
    filename = input("Enter the name of the file of trained agent : ")
    fc2Dims = int(input("Enter the dimensions of second fully connected hidden layer : "))
    # initialising the network
    agent = DeepQNetwork(inputDimensions=stateSize, fc1Dims=fc1Dims, fc2Dims=fc2Dims, numberOfActions=actionSize,
                         type="trained")
    # loading the trained network
    agent.load_state_dict(T.load(filename))
    return agent


def trainedAgent():
    """
    This method loads the trained agent and environment and shows its live performance
    """
    # initialisng the environment
    env = gym.make('LunarLander-v2')
    actionSize = env.action_space.n
    stateSize = env.observation_space.shape[0]
    # loading the agent
    agent = loadAgent(stateSize, actionSize)
    state = env.reset()
    score = 0
    state = np.reshape(state, [1, stateSize])
    # running the agent
    for t in range(500):
        env.render()
        action = np.argmax(agent.forward(Variable(T.Tensor(state))).data.numpy()[0])  # deciding the action
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, stateSize])
        score += reward
        state = next_state
        time.sleep(0.01)
        if done:
            break
    env.close()
    print("\nscore: {}".format(score), "\n")


def main():
    """
    This method gives the option either to train an agent or seee the performance of trained agent,
    after taking the input it do thr desired action
    """
    while True:
        menu = input(
            "\n\nEnter from the following choices : \n1. Press 1 if you want to train an agent\n2. Press 2 if you want to run a trained agent\n3. Press q/Q to quit : ")
        if menu == "q" or menu == "Q":
            break
        elif menu == "2":
            trainedAgent()
        elif menu == "1":
            trainTheAgent()
        else:
            print("Enter Valid Input : ")


if __name__ == '__main__':
    main()
