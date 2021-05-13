from typing_extensions import final

from numpy.core.fromnumeric import shape
from MADRLIV import Bandit, generate_agent,get_utility,util_from_ranks,utility_of_result, experiment, unique_nontrival_winner
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from voting_agents import Agent, Gradient_Agent,Neural_Agent,Bot

def check_convergence(action_hist,rounds=2):
    round_len = len(action_hist[-1])

    lookback = (rounds * round_len) - 1

    if len(action_hist)<lookback:
        return False

    current = action_hist[-1]

    for step in range(lookback):
        step = int(-1*step)
        if action_hist[step] != current:
            return False
    
    print("Converged!")


    return True


def generate_partial_action_input(partial):
    "Needed for sequential voting - creates list of actions already taken and pads to full length"
    out = []
    for p in partial:
        if p is None:
            out.append(0)
        else:
            out.append(p+1)
    
    return out

def generate_votes_input(partial):
    return [x for x in partial if x is not None]

seq_debug = False

def experiment_sequential(agents, bandit, N_episodes,use_exp,options):
    voting_method = 'plurality'
    decline = False

    episode = 0
    games = {'rewards':[],'winners':[],'round_lengths':[],'converged':[],'final_votes':[]}

    pbar = tqdm(total=N_episodes,disable=True)

    bandit.poll = []
    bandit.moves = []
    
    while episode < N_episodes:

        converged = False
        the_end = False

        actions = [None for x in range(len(agents))]
        step = 0
       

        while the_end == False:
            end =  int(np.random.uniform(5,10))
            for agent_idx, agent in enumerate(agents):
                #pad out current voting state

                poll = actions.copy()
                poll[agent_idx] = None
                bandit.poll.append(poll)
                bandit.moves.append(agent_idx)

                input = generate_partial_action_input(poll)

                bandit.partial_result = np.array([float(x) for x in input])

                


                #Obtain agent actions
                action = agent.get_action(bandit, episode, decline, N_episodes)
                actions[agent_idx]=action
                step += 1
        

                #calculate winner of election
                
                
                
                if None not in actions:
                    rewards = bandit.get_rewards(actions,voting_method,use_exp,options)
                else:
                    bandit.vote_winners.append(None)


                bandit.vote_history.append(actions.copy())

                converged = check_convergence(bandit.vote_history)

                if int(step/len(agents))>=end:
                    the_end = True
                if converged==True:
                    the_end=True

                if seq_debug == True:
                    print(episode,agent_idx,step,"----",action,actions,bandit.partial_result,bandit.ranks[agent_idx])

                #update each agent with their rewards
                if the_end == True:
                    for idx,agent in enumerate(agents):
                        #imperfect fix as double updates
                        if actions[idx] is None:
                            print("Update Error!")
                            print(episode,step,agent_idx,"----",action,actions,options)
                            raise InterruptedError
                        agent.update_Q(actions[idx], rewards[idx])
                        if seq_debug == True:
                            print("Q of",idx,"+",str((actions[idx], rewards[idx])))

                    #print(np.mean(rewards),converged)
                    break

                else:
                    agent.update_Q(action, 0.0)
                    if seq_debug == True:
                        print("Q of",agent_idx,"+",str((action, 0.0)))
                
                
              
        
        episode += 1
        pbar.update(1)
        games['rewards'].append(rewards)
        games['winners'].append(bandit.vote_winners[-1])
        games['final_votes'].append(bandit.vote_winners[-1])
        games['converged'].append(int(converged))
        games['round_lengths'].append(step)
    return bandit,games


def agent_experiment(ag_type,to_remember,pref_profile,N_episodes=1000,epsilon=0.1,alpha=0.1,use_exp=True,neural_update_interval=2,sequential=False,pbar=None):
    """
    Wrapper around experiment that tests specific comparison metric (borda score/max borda score or condorcet efficiency)
    """
    voting_method='plurality' #has to be

    options = pref_profile[0]
    voting_ranks = pref_profile[1]

    N_agents = len(voting_ranks)

    bandit = Bandit(voting_ranks,options,N_agents)  # initialize bandits
    agents = []
    
    for n in range(N_agents):
        if ag_type == "DQN_truthful":
            new_agent = pretrain_agent('DQN',to_remember,N_agents,agent_pref_profile=(options,voting_ranks[n]),pretrain_steps=1000,epsilon=epsilon,alpha=alpha,use_exp=use_exp,neural_update_interval=neural_update_interval)
        else:
            new_agent = generate_agent(ag_type,bandit,epsilon,alpha,neural_update_interval,memory=to_remember,bot_prefs=voting_ranks[n])
        agents.append(new_agent)
           
    if sequential == False:
        _,results = experiment(agents, bandit, N_episodes, decline=False,voting_method=voting_method,use_exp=use_exp,options=options,winner_to_compare='NULL',pbar=pbar,agent_reward_N=None)
    else:
        bandit, games = experiment_sequential(agents, bandit, N_episodes,use_exp,options)  # perform experiment
        results = (bandit,games)

    return results



class DummyBandit:
    "Runs each round of voting and stores information needed to feed as state inputs to agents"
    def __init__(self, ranks,options,n_vot):
        self.candidates = options

        self.nvot = n_vot
        self.N = len(options)  # number of bandits


        self.ranks = ranks  # utilities for each voting rule
        self.partial_result = None

    def fake_state(self):
        self.partial_result = np.random.rand(self.nvot)

    def get_reward(self, action,use_exp,options):
        """
        Inputs list of agent actions into the voting environment, makes an aggregator for that environment and returns rewards as list in agent order

        """
       
        act_letter = self.candidates[action]

        rank = self.ranks.index(act_letter)

        reward = get_utility(rank,use_exp,options)
        
        return reward



def pretrain_agent(ag_type,to_remember,N_agents,agent_pref_profile,pretrain_steps=1000,epsilon=0.1,alpha=0.1,use_exp=True,neural_update_interval=2):
    """
    Wrapper around experiment that tests specific comparison metric (borda score/max borda score or condorcet efficiency)
    """
    voting_method='plurality' #has to be

    options = agent_pref_profile[0]
    voting_ranks = agent_pref_profile[1]

    bandit = DummyBandit(voting_ranks,options,N_agents)  # initialize bandits

    new_agent = generate_agent(ag_type,bandit,epsilon,alpha,neural_update_interval,memory='Actions_now')


    for count in range(pretrain_steps):
        bandit.fake_state()
        action = new_agent.get_action(bandit=bandit, actnum=0, decline=False, N_episodes=10)
        
        reward = bandit.get_reward(action = action,use_exp=use_exp,options=options)

        new_agent.update_Q(action, reward)
        #print(action,reward)
    
    new_agent.remember = to_remember

    return new_agent


#pretrain_agent('DQN','Actions_now',7,[['A','B','C'],['B','C','A']])