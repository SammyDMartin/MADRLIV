from MADRLIV import plot_singlepref
from MADRLIV import unique_nontrival_winner
from MADRLIV_sequential import agent_experiment

from MADRLIV import util_from_ranks
import numpy as np
from tqdm import tqdm
import pickle

mems = [False,'Actions_now'] #types of agent memory to record

agent_types = ['DQN','DQN_truthful','tabular','tabular_truthful'] #types of agent

#mems = [False,'Actions_now']
#agent_types = ['DQN','DQN_truthful']
#agent_types = ['tabular_truthful','tabular']

N_pref=2000

SEQ = False
#use sequential or simultaneous

RES = True
#restrict preference profiles?

DP=False
if N_pref>1:
    DP = True

#produce the preference profile
C = 3 #must be 3
V = 7
CV = (C,V)


alpha = 0.1
alpha_dqn = 0.01
nt = 1
epslen = 500

filename = None
savename = 'simultaneous_1000_0.1_0.01'

A_test = False

if filename is not None:
    if SEQ == False:
        vote_histories = pickle.load(open(filename,"rb"))
    elif SEQ == True:
        vote_histories,vote_histories_seq = pickle.load(open(filename,"rb"))
else:
    total = N_pref*len(agent_types)*len(mems)*nt
    vote_histories = {}
    vote_histories_seq = {}
    pbar = tqdm(total=total, ncols=200)

    for tests in range(N_pref):
        opt,vr, metric_results,plurality_results = unique_nontrival_winner(CV[0],CV[1],'borda',restrict=RES)
        pp = [opt,vr]

        for ag_type in agent_types:
            if 'tabular' in ag_type:
                mems_here = [False]
                correction = int((len(mems)-1)*nt)
                pbar.update(correction)
            else:
                mems_here = mems
            for mem in mems_here:
                
                #Perform repeats for a given agent type

                MA = (mem,ag_type)
                reslist = vote_histories.get(MA,[])
                reslist_seq = vote_histories_seq.get(MA,[])
                for repeat in range(nt):
                    pbar.update(1)
                    if ag_type == 'DQN':
                        a = alpha_dqn
                    elif ag_type == 'DQN_truthful':
                        a = alpha_dqn
                    else:
                        a = alpha
                    bd = agent_experiment(ag_type=ag_type,to_remember=mem,pref_profile=pp,N_episodes=epslen,epsilon=0.1,alpha=a,use_exp=True,neural_update_interval=2,sequential=SEQ)


                    
                    if SEQ == True:
                        final_winners = bd[1]['winners']
                        rewards = bd[1]['rewards']
                        #length = bd[1]['round_lengths']
                        final_vote_history = bd[1]['final_votes']               
                        result = (final_vote_history, final_winners,opt,vr)
                        result_seq = (bd[0].vote_history, bd[0].vote_winners,(bd[0].poll,bd[0].moves),opt,vr)
                        reslist.append(result)
                        reslist_seq.append(result_seq)
                        desc = "{}-{}, R:{}".format(ag_type,mem,round(np.mean(bd[1]['rewards'][-100:]),3))
                    elif SEQ == False:
                        result = (bd.vote_history, bd.vote_winners,opt,vr)
                        desc = "{}-{}".format(ag_type,mem)
                        reslist.append(result)
                    #pbar.set_description(desc)
                        

                vote_histories[MA]=reslist
                vote_histories_seq[MA]=reslist_seq
if savename is not None:
    if SEQ == True:
        file = open(savename, 'wb')
        pickle.dump((vote_histories,vote_histories_seq),file)
        file.close()
    elif SEQ == False:
        file = open(savename, 'wb')
        pickle.dump(vote_histories,file)
        file.close()