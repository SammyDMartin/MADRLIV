import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
from policygrad import PolicyGrad
from dqn import Agent as DQNxR

def softmax(H):
    h = H - np.max(H)
    exp = np.exp(h)
    return exp / np.sum(exp)


def get_utility(rank,use_exp,options):
    #outputs utility if given rank order of option
    if use_exp == True:
        return 2**-(float(rank)) #exponential reward in the paper
    if use_exp == False:
        return len(options)-rank-1

def util_from_ranks(rank_list,use_exp=True):
    #outputs utility of entire list of rank order of options, indexed by A,B,C...
    letters = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    options = letters[:len(rank_list)]
    out = []
    for o in options:
        u = get_utility(rank_list.index(o),use_exp,options)
        out.append(u)
    return out

class Bot:
    def __init__(self, bandit, epsilon,_,ranks):
        self.Q = np.array(util_from_ranks(ranks))
        self.epsilon = epsilon
        #print(self.Q)
    def update_Q(self,action, reward):
        pass
    def get_action(self, bandit, actnum, decline, N_episodes, force_explore=False):
        rand = np.random.random()  # [0.0,1.0)
        if decline:

            if (rand < (self.epsilon * (1-(float(actnum)/N_episodes)))) or force_explore:
                action_explore = np.random.randint(bandit.N)  # explore random bandit

                return action_explore
            else:
                # action_greedy = np.argmax(self.Q)  # exploit best current bandit
                action_greedy = np.random.choice(np.flatnonzero(self.Q == self.Q.max()))
                return action_greedy
        else:
            if (rand < self.epsilon) or force_explore:
                action_explore = np.random.randint(bandit.N)  # explore random bandit
                return action_explore
            else:
                # action_greedy = np.argmax(self.Q)  # exploit best current bandit
                action_greedy = np.random.choice(np.flatnonzero(self.Q == self.Q.max()))
                return action_greedy

class Agent:
    def __init__(self, bandit, epsilon,alpha=None,ranks=None):
        self.epsilon = epsilon
        self.alpha = alpha
        self.k = np.zeros(bandit.N, dtype=np.int)  # number of times action was chosen
        self.Q = np.zeros(bandit.N, dtype=np.float)  # estimated value
        if ranks:
            for idx,_ in enumerate(self.Q):
                self.Q = self.alpha*np.array(util_from_ranks(ranks)) #preload utilities
    
    # Update Q action-value using:
    # Q(a) <- Q(a) + 1/(k+1) * (r(a) - Q(a))
    def update_Q(self, action, reward):
        self.k[action] += 1  # update action counter k -> k+1
        if self.alpha is not None:
            self.Q[action] += self.alpha * (reward - self.Q[action])
        else:
            self.Q[action] += (1./self.k[action]) * (reward - self.Q[action])

    # Choose action using an epsilon-greedy agent
    def get_action(self, bandit, actnum, decline, N_episodes, force_explore=False):
        rand = np.random.random()  # [0.0,1.0)
        if decline:

            if (rand < (self.epsilon * (1-(float(actnum)/N_episodes)))) or force_explore:
                action_explore = np.random.randint(bandit.N)  # explore random bandit

                return action_explore
            else:
                # action_greedy = np.argmax(self.Q)  # exploit best current bandit
                action_greedy = np.random.choice(np.flatnonzero(self.Q == self.Q.max()))
                return action_greedy
        else:
            if (rand < self.epsilon) or force_explore:
                action_explore = np.random.randint(bandit.N)  # explore random bandit
                return action_explore
            else:
                # action_greedy = np.argmax(self.Q)  # exploit best current bandit
                action_greedy = np.random.choice(np.flatnonzero(self.Q == self.Q.max()))
                return action_greedy

class Gradient_Agent:

    def __init__(self, bandit, epsilon, alpha):
        #self.epsilon = epsilon
        self.k = np.zeros(bandit.N, dtype=np.int)  # number of times action was chosen
        self.H = np.zeros(bandit.N, dtype=np.float)  # estimated value
        self.alpha = alpha
        self.epsilon = epsilon
        self.reward_history=[]

    def update_Q(self, action, reward):
        self.reward_history.append(float(reward))
        self.k[action] += 1  # update action counter k -> k+1
        R_ = np.average(self.reward_history)

        policy = softmax(self.H)

        self.H[action] += self.alpha*(reward-R_)*(1-policy[action])

        self.H[:action] += self.alpha*(reward-R_)*(policy[:action])
        self.H[action+1:] += self.alpha*(reward-R_)*(policy[action+1:])

    # Choose action using an epsilon-greedy agent
    def get_action(self, bandit, actnum, decline, N_episodes):
        policy = softmax(self.H)
        #rand = np.random.random()
        """
        if decline:

            if (rand < (self.epsilon * (1-(float(actnum)/N_episodes)))) or force_explore:
                action_explore = np.random.randint(bandit.N)  # explore random bandit

                return action_explore
            else:
                # action_greedy = np.argmax(self.Q)  # exploit best current bandit
                action_greedy = np.random.choice(bandit.N, p=policy)
                return action_greedy
        else:
        """
        action = np.random.choice(bandit.N, p=policy)
        return action


class Neural_Agent:
    def __init__(self,bandit,epsilon,alpha,layersize=128,UI=1000,gm=0.99,remember=False,algorithm='DQNxR'):
        self.size = bandit.nvot
        if algorithm == 'DQNxR':
            seed = np.random.rand() #DOESNT DO ANYTHING

            self.DQN = DQNxR(state_size=self.size,action_size=bandit.N,seed=seed,alpha=alpha,UI=UI,batch_size=10,gamma=gm,tau=1e-3,buffer_size=int(1e5))
            #print(vars(self.DQN))
            self.epsilon = epsilon
            self.last_state = None
            self.remember = remember
        elif algorithm == 'policygrad':
            self.DQN = None
            self.policy = PolicyGrad(state_space=self.size,action_space=bandit.N,hidden_layer_size=layersize,gamma=gm)
            self.optimizer = optim.Adam(self.policy.parameters(), lr=alpha)
            self.update_interval = UI
            self.remember = remember

#POLICY GRADIENT

    def select_action(self, state):
        #Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
        state = torch.from_numpy(state).type(torch.FloatTensor)
        state = self.policy(Variable(state))
        c = Categorical(state)
        action = c.sample()
        
        # Add log probability of our chosen action to our history    
        if self.policy.policy_history.dim() != 0:
            #print(policy.policy_history)
            #print(c.log_prob(action))
            self.policy.policy_history = torch.cat([self.policy.policy_history, c.log_prob(action).unsqueeze(0)])
            #print("DID!")
        else:
            self.policy.policy_history = (c.log_prob(action))
        return action

    def update_policy(self):
        R = 0
        rewards = []

        #print(self.policy.reward_episode)
        
        # Discount future rewards back to the present using gamma
        for r in self.policy.reward_episode[::-1]:
            R = r + self.policy.gamma * R
            rewards.insert(0,R)
            
        # Scale rewards
        rewards = torch.FloatTensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
        
        # Calculate loss
        loss = (torch.sum(torch.mul(self.policy.policy_history, Variable(rewards)).mul(-1), -1))
        
        # Update network weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #self.policy.loss_history.append(loss.data.item())
        #self.policy.reward_history.append(np.sum(policy.reward_episode))
        self.policy.policy_history = Variable(torch.Tensor())
        self.policy.reward_episode= []


#UNIVERSAL


    def update_Q(self, action, reward):
        if self.DQN is not None:
            self.AR = (action,reward)
        else:
            if len(self.policy.reward_episode) == self.update_interval:
                self.policy.reward_episode.append(reward)
                self.update_policy()
            else:
                self.policy.reward_episode.append(reward)
        
    
    def get_action(self, bandit, actnum, decline, N_episodes):
        if self.remember == False:
            state = np.ones(self.size)/100
        elif self.remember == "Rewards":
            state_info = bandit.last_rewards
            state = np.array(state_info)
            #print(actnum, state)
        elif self.remember == "Actions":
            state_info = bandit.last_actions
            state = np.array(state_info)
        elif self.remember == "Actions_now":
            state = bandit.partial_result

        if self.DQN is not None:
            if self.last_state is not None:
                #print(actnum,self.last_state,self.AR[0],self.AR[1],state)
                self.DQN.step(self.last_state,self.AR[0],self.AR[1],state, done=False)
                #print(self.last_state,self.AR[0],self.AR[1],state)

            actnum = self.DQN.act(state,self.epsilon).item()
            self.last_state = state
        else:            
            actnum = self.select_action(state).item()

            #print(state, actnum)

        return actnum