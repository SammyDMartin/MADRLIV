import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

from voting_tools import Aggregator
from voting_tools import autogenerate_votes
from voting_tools import find_winner
import matplotlib.animation as animation

import os, sys

import warnings
import time
import pandas as pd

from voting_agents import Agent, Gradient_Agent,Neural_Agent,Bot

print_vote_diagnostics = False
print_final_diag=False
last_rewards_instead = False #plots last rewards instead of anything else
warnings.filterwarnings("ignore")


# =========================
# General Utility
# =========================

def producetimefolder():
    savestr = time.strftime("%m %d %Y, %H %M %S")
    os.mkdir(savestr)
    savestr = savestr + "/"
    return savestr


def moving_average(a, n=1):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def moving_std(a, n=3):
    s = pd.Series(a)
    out = s.rolling(n).std().to_numpy()
    out = out[np.logical_not(np.isnan(out))]
    return out

def random_color():
    r = random.random()
    b = random.random()
    g = random.random()
    colour1 = (r, g, b)
    colour2 = (0.5*r,0.5*g,0.5*b)
    return colour1, colour2

# =========================
# Calculate Utilities from pref ranks, either exponential or linear
# =========================

def get_utility(rank,use_exp,options):
    #outputs utility if given rank order of option
    if use_exp == True:
        return 2**-(float(rank)) #exponential reward in the paper
    if use_exp == False:
        return len(options)-rank-1

def util_from_ranks(rank_list,use_exp,options):
    #outputs utility of entire list of rank order of options, indexed by A,B,C...
    out = []
    for o in options:
        u = get_utility(rank_list.index(o),use_exp,options)
        out.append(u)
    return out


def utility_of_result(answer,options,voting_ranks,use_exp):
    index = options.index(answer)
    final_util = []
    for r in voting_ranks:
        utils = util_from_ranks(r,use_exp,options)
        util = utils[index]
        final_util.append(util)
    
    return (float(np.min(np.array(final_util))), float(np.mean(np.array(final_util))))


# =========================
# Define Bandit and Agent class
# =========================
class Bandit:
    "Runs each round of voting and stores information needed to feed as state inputs to agents"
    def __init__(self, ranks,options,n_vot):
        self.candidates = options
        self.nvot = n_vot
        self.N = len(options)  # number of bandits
        self.ranks = ranks  # utilities for each voting rule
        self.votes = None
        self.vote_winners = []
        self.last_rewards = [0 for x in range(self.nvot)]
        self.last_actions = [0 for x in range(self.nvot)]
        self.partial_result = None

    def voting_environment(self):
        "Generates random voting environment"
        return autogenerate_votes(n_candidates=self.N,n_voters=self.nvot) #this needs to be replaced obviously, currently he's playing flailing idiots
    
    def add_to_environment(self,action,voter_id):
        "Adds voter to environment - single first choice vote only!"
        vote_letter = self.candidates[action]

        agents_votes = self.votes[1][voter_id]
        agents_votes = [vote_letter]

        c2 = self.candidates.copy()
        random.shuffle(c2) #so that intermediate results are randomized

        for c in c2:
            if c not in agents_votes:
                agents_votes.append(c)

        self.votes[1][voter_id] = agents_votes


    def get_rewards(self, actions,voting_method,use_exp,options):
        """
        Inputs list of agent actions into the voting environment, makes an aggregator for that environment and returns rewards as list in agent order

        """
        random.seed(time.clock())

        self.votes = self.voting_environment()

        for idx,action in enumerate(actions):
            self.add_to_environment(action,idx) #pick different voter id to give different voter

        contest = Aggregator(inputs=self.votes)

        result = getattr(contest,voting_method)()
        result = [(value, key) for key, value in result.items()]

        largest = max(result)[0]
        #random tiebreak

        winners = [i for i in result if i[0] == largest]
        winner = random.choice(winners)[1]

        self.vote_winners.append(winner)

        final = self.candidates.index(winner)

        rewards = []
        for idx,action in enumerate(actions):
            agent_x = self.ranks[idx]
            rank = agent_x.index(winner)
            reward = get_utility(rank,use_exp,options)
            rewards.append(reward)

        self.last_rewards = rewards
        self.last_actions = actions
        return rewards


def generate_partial_action_input(N, partial):
    "Needed for sequential voting - creates list of actions already taken and pads to full length"
    to_pad = N - len(partial)
    partial = np.array(partial) + 1
    
    state_output = np.pad(partial,(0,to_pad))
    return state_output


def experiment(agents, bandit, N_episodes, decline,voting_method,use_exp,options,winner_to_compare=None,pbar=None,agent_reward_N=None):
    """Runs single iterative voting experiment and outputs the relevant variable being tested.
    if agent_reward_N is not None, will output the reward given to agent N.
    Otherwise outputs condorcet efficiency (if winner_to_compare is str - the condorcet winner)
    or borda score/max borda score (if winner_to_compare is dict and borda scores for each winner)

    Args:
        agents (list): list of agents with classes given in voting_agents
        bandit (Bandit): the bandit object
        N_episodes (int): how many steps of IV
        decline (bool): whether to decline epsilon
        voting_method (str): voting method to use for voting round. should be 'plurality'
        use_exp (bool): True for exponentially declining reward, False for linearly declining
        options (list): list of candidates (should be of form ['A','B','C'...])
        winner_to_compare (str or dict, optional): Either a str that is the unique condorcet winner or dict of borda scores. Defaults to None.
        pbar (tqdm, optional): tqdm object. Defaults to None.

    Returns:
        voting metric: np.array object of length N_episodes with borda/condorcet scores or reward of agent N
    """
    reward_history = []
    for episode in range(N_episodes+1):
        actions = []
        for agent in agents:
            #for if sequential voting is being used
            input = generate_partial_action_input(len(agents),actions)
            bandit.partial_result = input

            #Obtain agent actions
            action = agent.get_action(bandit, episode, decline, N_episodes)
            actions.append(action)

        #calculate winner of election
        rewards = bandit.get_rewards(actions,voting_method,use_exp,options)

        #update each agent with their rewards
        for idx,agent in enumerate(agents):
            agent.update_Q(actions[idx], rewards[idx])

        #DEBUG
        if print_vote_diagnostics == True:
            arw = []
            for idx in range(len(actions)):
                arw.append((actions[idx],rewards[idx]))
            print(episode, arw)

        reward_history.append(np.array(rewards))
    
        if episode % 10 == 9:
            done = str(round(100*float(episode)/float(N_episodes), 1))
            if type(winner_to_compare) is str:
                #Measuring condorcet efficiency
                ret = 'cw'
                out = []
                winners = bandit.vote_winners
                for w in winners:
                    if w == winner_to_compare:
                        out.append(1)
                    else:
                        out.append(0)
                out = np.array(out)

            elif type(winner_to_compare) is dict:
                #Measuring borda score
                ret = 'borda'
                singlewin = lambda x : x[0]
                winner = singlewin(find_winner(winner_to_compare))
                borda_score_winner = winner_to_compare[winner]
                out = []
                winners = bandit.vote_winners
                for w in winners:
                    out.append(winner_to_compare[w])
                out = np.array(out)/borda_score_winner
            
            done = ret+'='.ljust(3)+str(round(float(np.mean(out[-1*int(N_episodes/10):])),3)).ljust(10)+done.ljust(5)
            if pbar is not None:
                pbar.set_description(done)
        
    if print_final_diag is True:
        #fpr diagnostics
        print("\n\n\nDUMP:")
        for a in agents:
            print(vars(a))
        banditv = vars(bandit)
        for k in banditv.keys():
            print(k, banditv[k])
        print("DUMP over\n\n\n")
        print(out[::50])
    
    if agent_reward_N is not None:
        #outputs rewards instead of borda/cw
        result = []
        for r in reward_history:
            result.append(r[agent_reward_N])
        return np.array(result)
    else:
        return out
    
    if print_vote_diagnostics is True:
        for idx, r in enumerate(reward_history[:10]):
            print(str(idx).ljust(5), str(bandit.vote_winners[idx]).ljust(10))
            print("Rewards to agents: ", list(r))
            print(ret, "score returned:", out[idx])
            print("\n\n")
    return out





# =========================
# Filter preference profiles
# =========================


def unique_nontrival_winner(NC,NV,measure,restrict,correlation_const=None):
    """ Used to produce preference profiles for IV. If measure is condorcet winner (pairwise_comparison), ensures unique condorcet winner
    otherwise, if restrict is True, ensures there is only a single winner by measure and that it is not the plurality winner(s)
    if restrict is False, just ensures output makes sense according to measure
    if restrict is 'partial', ensures plurality winner(s) != measure winner(s)

    Args:
        NC (int): number of candidates
        NV (int): number of voters
        measure (str): borda or pairwise_comparison - metric used to filter profiles
        restrict (bool or str): True, False or partial
        correlation_const (float): correlation constant for correlating similar profiles. Defaults to None.

    Returns:
        opt,vr, metric_results,plurality_results
    """

    #print("\nSearching...",NC, NV)
    single_winner = False
    while single_winner is False:
        out = autogenerate_votes(NC,NV,correlation_const)
        opt,vr = out[0],out[1]

        scenario = [opt,vr]
        comparison = Aggregator(inputs=scenario)

        metric_results = getattr(comparison,measure)()
        plurality_results = comparison.plurality()

        metric_winners = find_winner(metric_results)
        plurality_winners = find_winner(plurality_results)

        if restrict is False:
            if measure is 'borda':
                #we don't care if there are multiple plurality or borda winners in this case, any config will do
                single_winner = True
            elif measure is 'pairwise_comparison':
                if len(metric_winners) is 1:
                    single_winner = True
        elif restrict is True:
            if bool(set(metric_winners) & set(plurality_winners)) is False:
                if measure is 'borda':
                    single_winner = True
                elif measure is 'pairwise_comparison':
                    if len(metric_winners) is 1:
                        single_winner = True
        elif restrict is 'partial':
            if bool(set(plurality_winners) - set(metric_winners)) is True:
                if measure is 'borda':
                    single_winner = True
                elif measure is 'pairwise_comparison':
                    if len(metric_winners) is 1:
                        single_winner = True

    return opt,vr, metric_results,plurality_results

def generate_agent(ag_type,bandit,epsilon,alpha,update_interval,memory=False,bot_prefs=None):
    """
    Produces agent from voting_agents with the given parameters, including memory types, update interval, exploration rate, learning rate
    """

    if memory not in [False,'Rewards','Actions','Actions_now']:
        print(memory, ag_type,"No such agent exists")
        return

    if ag_type is 'grad':
        return Gradient_Agent(bandit, epsilon, alpha)  # initialize agent
    elif ag_type is 'tabular':
        return Agent(bandit, epsilon, alpha)  # initialize agent
    elif ag_type is 'neural':
        return Neural_Agent(bandit,epsilon,alpha,remember=memory,UI=update_interval,algorithm='policygrad')
    elif ag_type is 'DQN':
        return Neural_Agent(bandit,epsilon,alpha,remember=memory,UI=update_interval,algorithm='DQNxR')
    elif ag_type is 'bot':
        if bot_prefs is not None:
            return Bot(bandit,epsilon,None,bot_prefs)
        else:
            print("No bot Prefs")
    else:
        print(ag_type, "No such agent exists")





# =========================
# Used for keeping agents and their preference profiles together
# =========================


def pack_agentdict(voting_ranks,agent_list):
    out = {}
    for idx,a in enumerate(agent_list):
        agent = a
        agent.bandit = None
        tvr = tuple(voting_ranks[idx])
        out[tvr] = a
    return out

def unpack_agentdict(bandit,voting_ranks,agentlist,agentdict):
    for idx,voter in enumerate(voting_ranks):
        if tuple(voter) in agentdict.keys():
            old_params = agentdict[tuple(voter)]
            old_params.bandit = bandit
            agentlist[idx] = old_params
            #print("Uploaded",idx,voter)
    return agentlist




def test_agents(ag_type,to_remember,pref_profile,agent_dict = {}, N_episodes=1000,epsilon=0.1,alpha=0.1,use_exp=True,use_cw=False,DEC=False,restrict=False,neural_update_interval=2,pbar=None):
    """
    Generates a test sequence for 

    """
    voting_method='plurality' #has to be
    
    singlewin = lambda x : x[0] if len(x) == 1 else None #gets THE single winner or returns None, causing later errors if this isn't true. needed to detect errors

    if use_cw is True:
        metric = 'pairwise_comparison'
    elif use_cw is False:
        metric = 'borda'

    options = pref_profile[0]
    voting_ranks = pref_profile[1]

    comparison = Aggregator(inputs=pref_profile)
    metric_results = getattr(comparison,metric)()
    plurality_results = comparison.plurality()


    if use_cw is True:
        metric = 'pairwise_comparison'
    elif use_cw is False:
        metric = 'borda'
        if restrict == False:
            singlewin = lambda x : x[0] #if
    
    """
    if print_vote_diagnostics is True:
        #print("Random cycle #", count)
        print("Voting Ranks:")
        for r in voting_ranks:
            print(r)
        print(metric, "Election results:")
        print(metric_results)
        print("Plurality results:")
        print(plurality_results)
    """
 
    score_maximised = singlewin(find_winner(metric_results))

    N_agents = len(voting_ranks)

    bandit = Bandit(voting_ranks,options,N_agents)  # initialize bandits
    agents = []
    
    for n in range(N_agents):
        new_agent = generate_agent(ag_type,bandit,epsilon,alpha,neural_update_interval,memory=to_remember,bot_prefs=voting_ranks[n])
        agents.append(new_agent)

    agents = unpack_agentdict(bandit,voting_ranks,agents,agent_dict)
            
    """
    for idx,a in enumerate(agents):
        print("Agent #:",str(idx),type(a))
        print(vars(a))
    """
    if use_cw is True:
        if type(score_maximised) is not str:
            print("\nC error!")
            print("Input Winner:", score_maximised)
            print("Winner(s): ", find_winner(metric_results))
            print(metric, metric_results)
            print("CW error!")
            raise ValueError
        reward_history = experiment(agents, bandit, N_episodes, DEC,voting_method,use_exp,options,score_maximised,pbar=pbar)  # perform experiment
    else:
        if type(metric_results) is not dict:
            print(type(metric_results))
            print("\nB Error!")
            print(score_maximised)
            print(metric,metric_results)
            print("Borda error!")
            raise ValueError
        reward_history = experiment(agents, bandit, N_episodes, DEC,voting_method,use_exp,options,metric_results,pbar=pbar)
    #print(i, np.mean(reward_history))
    reward_history_avg = np.array(reward_history)

    if print_vote_diagnostics is True or print_final_diag is True:
        print("Voting Ranks:")
        for r in voting_ranks:
            print(r)
        print(metric, "Election results:")
        print(metric_results)
        print("Plurality results:")
        print(plurality_results)
        print("Score Maximised")
        print(score_maximised)

    if pbar is not None:
        pbar.update(1)

    
    return reward_history_avg, pack_agentdict(voting_ranks,agents)
















# =========================
# Two Plotting Methods
# =========================



def plot_singlepref(fold,mems,agent_types,pref_profile,agent_alpha,N_tests,percent,metric='borda_score',eps_len=1200,updateinterval=2):
    """[summary]

    Args:
        fold ([type]): [description]
        mems ([type]): [description]
        agent_types ([type]): [description]
        pref_profile ([type]): [description]
        agent_alpha ([type]): [description]
        N_tests ([type]): [description]
        percent ([type]): [description]
        metric (str, optional): [description]. Defaults to 'borda_score'.
        eps_len (int, optional): [description]. Defaults to 1200.
        updateinterval (int, optional): [description]. Defaults to 2.
    """
    global last_rewards_instead
    last_rewards_instead = False

    comparison = Aggregator(inputs=pref_profile)
    if metric == "borda_score":
        metric_results = getattr(comparison,"borda")()
    elif metric == "condorcet_eff":
        metric_results = getattr(comparison,"pairwise_comparison")()
    plurality_results = comparison.plurality()
    
    plt.figure(figsize=(20,20))

    limr = N_tests * len(mems)*len(agent_types)

    pbar = tqdm(total=limr,ncols=100)
    
    for mem in mems:
        for agent_type in agent_types:
            if agent_type == 'tabular':
                a = 0.1
            elif agent_type == 'bot':
                a = 0.0
            else:
                a = agent_alpha

            CV = (1,1)
            x = []
            y = []

            for T in range(N_tests):
                reward_seq,_ = test_agents(agent_type,mem,pref_profile,agent_dict={},N_episodes=eps_len,epsilon=0.1,alpha=a,pbar=pbar)

                interv = int((percent/100)*len(reward_seq))

                first = reward_seq[:interv]
                last = reward_seq[-1*interv:]


                x.append(np.mean(first))
                y.append(np.mean(last))


            
            plt.scatter(x,y,label = str(mem) + "  " + agent_type + " alph=" + str(a))
        
    plt.plot(np.linspace(0,1),np.linspace(0,1),'--',linewidth=1.0)


    plt.xlabel(metric+' first {}%'.format(int(percent)))
    plt.ylabel(metric+' last {}%'.format(int(percent)))

    full_results=metric + ": " + str(metric_results) + "\n" + "plurality" + ": " + str(plurality_results)
    plt.text(0.05, 0.95, full_results, fontsize=16, bbox=dict(facecolor='blue', alpha=0.2))
    single_profile = "\nPlurality Winners: {}, {} Winners: {}".format(find_winner(plurality_results),metric,find_winner(metric_results))


    title = metric + " e_l " + str(eps_len) + " neural_UI " + str(updateinterval) +"\n" + single_profile
    #print(title)

    plt.legend()

    plt.title(title)

    savestr = time.strftime("%H %M %S")
    savestr = fold+str(savestr)+"2dp_"+str(metric)

    #plt.ylim((0.0,1.0))
    #plt.xlim((0.0,1.0))

    plt.savefig(savestr+".png")


def run_general_comparison(fold,NT,NE,ELIM,params,agents,mems,average_per=None,Cb=[True,False]):
    NUI,alpha,restrict,use_exp,CV = params
    global last_rewards_instead
    last_rewards_instead = False

    if type(CV) is list:
        if NT is 1:
            C = len(CV[0])
            V = len(CV[1])
            preferences = CV
            CV = (C,V)
        else:
            print("Cannot reshuffle when preference profile is given!")
            print(type(CV))
            print(CV)
            print(NT)
            raise ValueError
    elif type(CV) is tuple:
        if NT == 1:
        #For if we're using a single agent as experiment
            if True in Cb:
                preferences = unique_nontrival_winner(CV[0],CV[1],'pairwise_comparison',restrict=restrict,correlation_const=None)
            else:
                preferences = unique_nontrival_winner(CV[0],CV[1],'borda',restrict=restrict,correlation_const=None)

    resolution_per = int(ELIM/20)
    
    params_str = 'neural update interval = ' + str(NUI) + ' restrict = ' + str(restrict) + '\nexponential reward = ' + str(use_exp) + ' N_C/N_V: ' + str(CV) + 'S/E res:' + str(resolution_per)
    if average_per is None:
        average_per = resolution_per
    

    for C in Cb:
        #BORDA and CW

        basic_cycle_interval = len(agents)*len(mems)*NT*NE
        pbar = tqdm(total=basic_cycle_interval,ncols=100)

        plt.figure(figsize=(20,10))
        if C == False:
            kind = 'borda'
        elif C == True:
            kind = 'condorcet'

        if NT == 1:
            #For if we're using a single agent as experiment
            comparison = Aggregator(inputs=preferences)
            if kind == "borda":
                metric_results = getattr(comparison,"borda")()
            elif kind == "condorcet":
                metric_results = getattr(comparison,"pairwise_comparison")()
            plurality_results = comparison.plurality()

            full_results=kind + ": " + str(metric_results) + "\n" + "plurality: " + str(plurality_results)
            plt.text(0.05, 0.95, full_results, fontsize=16, bbox=dict(facecolor='blue', alpha=0.2))
            single_profile = "\nPlurality Winners: {}, {} Winners: {}".format(find_winner(plurality_results),kind,find_winner(metric_results))
        else:
            single_profile = ""

        for agent_kind in agents:
            #Cycle between agent kinds
            for mem in mems:
                #Cycle between memory types (redundant for tabular!!)
                if agent_kind == 'tabular':
                    a = 0.1
                elif agent_kind == 'bot':
                    a = 0.0
                else:
                    a = alpha

                vote_result = np.zeros(ELIM)
                if NT > 1:
                    for reshuffles in range(NT):
                        #Reshuffle
                        if C is True:
                            preferences = unique_nontrival_winner(CV[0],CV[1],'pairwise_comparison',restrict=restrict,correlation_const=None)
                        else:
                            preferences = unique_nontrival_winner(CV[0],CV[1],'borda',restrict=restrict,correlation_const=None)
                            
                            for experiments in range(NE):
                                voteres, _ = test_agents(agent_kind,mem,preferences,{},ELIM,epsilon=0.1,alpha=a,use_exp=use_exp,use_cw=C,DEC=False,restrict=restrict,neural_update_interval=NUI,pbar=pbar)
                                vote_result += voteres
                elif NT == 1:
                    for experiments in range(NE):
                        voteres, _ = test_agents(agent_kind,mem,preferences,{},ELIM,epsilon=0.1,alpha=a,use_exp=use_exp,use_cw=C,DEC=False,restrict=restrict,neural_update_interval=NUI,pbar=pbar)
                        vote_result += voteres


                vote_result = vote_result / float(NT*NE)

                average, std = moving_average(vote_result,n=average_per), moving_std(vote_result,n=average_per)
                base = np.arange(len(average))

                start_average,start_std = np.mean(vote_result[:resolution_per]), np.std(vote_result[:resolution_per])
                final_average,final_std = np.mean(vote_result[-1*resolution_per:]), np.std(vote_result[-1*resolution_per:])

                start_r = str(round(start_average,3)) + " +/- " + str(round(start_std,3))
                end_r = str(round(final_average,3)) + " +/- " + str(round(final_std,3))

                res_string = str(mem) + " " + str(agent_kind) + " start: "  + start_r + "\n" + "end: "+end_r
                #print("\n")
                #print(agent_kind + " S:"  + start_r + "    E:"+end_r)
                #print("\n")

                plt.plot(base, average, label = res_string + '__a=' + str(a))
                plt.fill_between(base, average+std, average-std,alpha = 0.1)

        
        plt.legend()

        plt.ylabel('{} efficiency')
        plt.xlabel('Episode')

        savestr = fold+time.strftime("%H %M %S")
        savestr = savestr+"general_"+str(ELIM) + "_" + str((NT,NE))
        plt.title(params_str+single_profile)
        
        plt.savefig(savestr+"{}.png".format(kind))
        plt.xscale("log")
        plt.savefig(savestr+"log_{}.png".format(kind))




def run_situation_comparison(fold,NE,ELIM,pp,average_per=None,bot_epsilon=1.0):
    #print("Plot averaging period = ", average_per)

    NUI = 2
    alpha = 0.01
    epsilon = 0.1
    restrict = True
    use_exp = True
    CV = (len(pp[0]),len(pp[1]))

    global last_rewards_instead
    last_rewards_instead = True

    resolution_per = int(ELIM/20)

    preferences = pp

    dummy = Bandit(preferences[1],preferences[0],len(preferences[1]))

    final_agent_list = [['DQN','Actions_now'],['DQN',False]]
    rest_agent = 'bot'

    #params_str = 'neural update interval = ' + str(NUI) +' bot_eps = ' + str(epsilon) = ' alpha = ' + str(alpha) + ' nontrivial_only = ' + str(restrict) + '\nexponential reward = ' + str(use_exp) + ' N_C/N_V: ' + str(CV) + 'S/E res:' + str(resolution_per)

    if average_per is None:
        average_per = resolution_per
    
    basic_cycle_interval = len(final_agent_list)*NE
    pbar = tqdm(total=basic_cycle_interval,ncols=100)

    plt.figure(figsize=(20,10))
    
    C = False
    kind = 'borda'

    comparison = Aggregator(inputs=preferences)
    if kind == "borda":
        metric_results = getattr(comparison,"borda")()
    elif kind == "condorcet":
        metric_results = getattr(comparison,"pairwise_comparison")()
    plurality_results = comparison.plurality()
    single_profile ="\n"+ kind + ": " + str(metric_results) + "\n" + str(plurality_results)
    
    for final_agent in final_agent_list:
        situation_dict = {}

        #construct the situation dict
        for idx,pref_profile in enumerate(preferences[1]):
            if idx < CV[1]-1:
                ag = generate_agent(rest_agent,dummy,bot_epsilon,alpha,NUI,bot_prefs=pref_profile)
            elif idx == CV[1]-1:
                final_type = final_agent[0]
                final_mem = final_agent[1]
                ag = generate_agent(final_type,dummy,epsilon,alpha,NUI,memory=final_mem)
            
            situation_dict[tuple(pref_profile)] = ag
        #dict constructed

        vote_result = np.zeros(ELIM+1)
        for experiments in range(NE):
            voteres, _ = test_agents('tabular',False,preferences,situation_dict,ELIM,epsilon=0.1,alpha=alpha,use_exp=use_exp,use_cw=C,DEC=False,restrict=restrict,neural_update_interval=NUI,pbar=pbar)
            vote_result += voteres


        vote_result = vote_result / float(NE)

        average, std = moving_average(vote_result,n=average_per), moving_std(vote_result,n=average_per)
        base = np.arange(len(average))

        start_average,start_std = np.mean(vote_result[:resolution_per]), np.std(vote_result[:resolution_per])
        final_average,final_std = np.mean(vote_result[-1*resolution_per:]), np.std(vote_result[-1*resolution_per:])

        start_r = str(round(start_average,3)) + " +/- " + str(round(start_std,3))
        end_r = str(round(final_average,3)) + " +/- " + str(round(final_std,3))

        res_string = str(final_agent) + " start: "  + start_r + "\n" + "end: "+end_r
        #print("\n")
        #print(agent_kind + " S:"  + start_r + "    E:"+end_r)
        #print("\n")

        plt.plot(base, average, label = res_string + '__a=' + str(alpha))
        plt.fill_between(base, average+std, average-std,alpha = 0.1)

        
    plt.legend()

    plt.ylabel('Last agent mean reward'.format(kind))
    plt.xlabel('Episode')

    savestr = fold+time.strftime("%H %M %S")
    savestr = savestr+"sit_"+str(ELIM) + "_" + str(NE)+ " " + str(bot_epsilon)
    plt.title(single_profile)
    
    plt.savefig(savestr+"special.png")
    plt.xscale("log")
    plt.savefig(savestr+"log_special.png")

def memory_test(NE):
    pp = unique_nontrival_winner(5,5,'pairwise_comparison',restrict=True,correlation_const=None)
    
    _, agents = test_agents("DQN",'Actions_now',pp,{},alpha=0.01,N_episodes=1000)
    progressbar = tqdm(total=NE*2,ncols=100)

    for count in range(NE):

        r, _ = test_agents("DQN",'Actions_now',pp,agents,alpha=0.01,N_episodes=1000,pbar=progressbar)
        progressbar.update(1)
        r2, _ = test_agents("DQN",'Actions',pp,{},alpha=0.01,N_episodes=1000,pbar=progressbar)
        progressbar.update(1)
        r,r2 = np.array(r),np.array(r2)

        if count == 0:
            results,results2 = np.zeros_like(r),np.zeros_like(r2)
        
        results += r
        results2 += r2


    plt.plot(results/NE,label='DQN_pre_curr')
    plt.plot(results2/NE,label='DQN')
    plt.legend()
    plt.show()

def deep_background(tf,alpha,epslen,nt,agent_types,mems,CV,AP=10):
    global last_rewards_instead
    last_rewards_instead = False
    opt,vr, metric_results,plurality_results = unique_nontrival_winner(CV[0],CV[1],'borda',restrict=True)
    pp = [opt,vr]

    plot_singlepref(tf,mems=mems,agent_types=agent_types,pref_profile=pp,agent_alpha=alpha,N_tests=nt,percent=10,metric='borda_score',eps_len=epslen,updateinterval=2)
    
    #print("Situational")
    #run_situation_comparison(timefolder,10,2000)
    #memory_test(10)

    #NUI,alpha,restrict,use_exp,CV = params
    params_list = [2,alpha,True,True,pp]
    #run_general_comparison(timefolder,NT=1,NE=10,ELIM=1000,params=params_list,agents=['DQN'],mems=[False,'Actions_now'],average_per=None,Cb=[False])
    run_general_comparison(tf,NT=1,NE=nt,ELIM=epslen,params=params_list,agents=agent_types,mems=mems,average_per=AP,Cb=[False])

if __name__ == "__main__":
    
    #for testing everything
    """
    timefolder = producetimefolder()
    opt,vr, metric_results,plurality_results = unique_nontrival_winner(5,3,'borda',restrict=True)
    pp = [opt,vr]
    reruns = 100
    repeats = 10

    for count in range(repeats):
        run_situation_comparison(timefolder,reruns,2000,pp,average_per=2)
    
    for count in range(repeats):
        deep_background(timefolder,0.01,300,reruns,['DQN'],[False,'Actions_now'],(5,5))

    timefolder = producetimefolder()
    for count in range(repeats):
        deep_background(timefolder,0.02,300,reruns*10,['tabular','bot'],[False],(5,5))

    timefolder = producetimefolder()
    params_list = [2,0.01,True,True,(5,5)]
    #run_general_comparison(timefolder,NT=1,NE=10,ELIM=1000,params=params_list,agents=['DQN'],mems=[False,'Actions_now'],average_per=None,Cb=[False])
    run_general_comparison(timefolder,NT=repeats,NE=reruns,ELIM=300,params=params_list,agents=['tabular','bot'],mems=[False],average_per=2,Cb=[False])
    """
    timefolder = producetimefolder()

    reruns = 2000
    repeats = 5
    eln = 1000

    for count in range(repeats):
        opt,vr, metric_results,plurality_results = unique_nontrival_winner(5,5,'borda',restrict=True)
        pp = [opt,vr]
        #run_situation_comparison(timefolder,reruns,eln,pp,average_per=100,bot_epsilon=0.5)
        #run_situation_comparison(timefolder,reruns,eln,pp,average_per=10,bot_epsilon=0.1)
        #run_situation_comparison(timefolder,reruns,eln,pp,average_per=10,bot_epsilon=1.0)

    params_list = [4,0.005,True,True,(5,5)]
    for count in range(10):
        run_general_comparison(timefolder,NT=1000,NE=1,ELIM=1000,params=params_list,agents=['DQN'],mems=['Actions_now',False],average_per=50,Cb=[False])

    timefolder = producetimefolder()

    #testing specifics of actions_now vs none with short update and long time horizon
    for count in range(5):
        deep_background(timefolder,0.01,500,1000,['DQN'],[False,'Actions_now'],(5,5))

    #testing random reshuffles of actions_now vs none with short update and long time horizon
    timefolder = producetimefolder()
    params_list = [1,0.005,True,True,(5,5)]
    run_general_comparison(timefolder,NT=1000,NE=1,ELIM=1000,params=params_list,agents=['DQN'],mems=['Actions_now',False],average_per=100,Cb=[False])


    #do something about eps declining!!
    #testing the bot environment with short update and long time horizon and possibly also 
