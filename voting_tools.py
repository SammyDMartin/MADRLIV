#!/usr/bin/env python3

import argparse
import copy
import csv
import os
import sys
from random import sample
from random import random
import random as rand
import sys, os
from tqdm import tqdm
import string
import time
import numpy as np
import matplotlib.pyplot as plt

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


'''
candidates = ['A', 'B', 'C', 'D']

prefs = [
    ['A', 'B', 'C', 'D'],
    ['D', 'C', 'B', 'A'],
    ['B', 'C', 'A', 'D'],
    ['A', 'B', 'C', 'D'],
    ['A', 'B', 'C', 'D'],

    ['A', 'B', 'C', 'D'],
    ['B', 'C', 'A', 'D'],
    ['A', 'B', 'C', 'D'],
    ['B', 'C', 'A', 'D'],
    ['B', 'C', 'A', 'D'],

    ['B', 'C', 'A', 'D'],
    ['C', 'B', 'A', 'D'],
    ['C', 'B', 'A', 'D'],
    ['C', 'B', 'A', 'D'],
    ['D', 'C', 'B', 'A'],

    ['C', 'B', 'A', 'D'],
    ['D', 'C', 'B', 'A'],
    ['A', 'B', 'C', 'D']
]
'''


class InputError(Exception):

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return repr(self.msg)


class PreferenceSchedule():

    def __init__(self, candidates, prefs):
        # check whether the candidates list consists of only strings
        if not all(map(lambda x: type(x) == str, candidates)):
            raise InputError('Candidate must be a string')

        # check the validity of the preferences
        for pref in prefs:
            # check whether the number of candidates in the preference schedule
            # is valid
            if len(pref) != len(candidates):
                raise InputError('Invalid preference schedule')

            # check whether the candidates in the preference schedule are unique
            if len(pref) != len(candidates):
                raise InputError('Invalid preference schedule')

            # check whether the candidates in the preference schedule are also
            # in the candidates list
            for candidate in pref:
                if candidate not in candidates:
                    raise InputError('Invalid preference schedule')

        self.prefs = prefs

    def original(self):
        '''Returns the original preference schedule as a printable string'''

        res = ''
        for i in range(len(self.prefs)):
            res += 'Voter {}: '.format(i+1) + ', '.join(self.prefs[i]) + '\n'

        return res[:-1]

    def detailed(self):
        '''Returns the detailed preference schedule as a printable string'''

        # count the number of occurences of each preference
        prefs = self.prefs[:]
        prefs = [tuple(p) for p in self.prefs]
        counts = {}
        while prefs:
            pref = prefs.pop(0)
            count = 1
            while pref in prefs:
                prefs.remove(pref)
                count += 1
            counts[pref] = count

        res = ''
        for pref in counts:
            res += str(counts[pref]) + ' Voters: ' + ', '.join(pref) + '\n'

        return res[:-1]

class Aggregator():

    def __init__(self, file=None,inputs=None):
        try:
            if inputs is not None:
                self.candidates = inputs[0]
                self.pref_schedule = PreferenceSchedule(inputs[0], inputs[1])
            else:
                candidates, prefs = csv_to_preference_schedule(file)
                self.candidates = candidates
                self.pref_schedule = PreferenceSchedule(candidates, prefs)
        except InputError as e:
            print(e)
    def __str__(self):
        res = ''
        res += 'Preference Schedule:\n'
        res += self.pref_schedule.original() + '\n\n'
        #res += 'Detailed Preference Schedule:\n'
        #res += self.pref_schedule.detailed() + '\n'

        return res
    
    def one_man_one_vote(self):
        "Dictatorship of first voter"

        winner = []
        the_man = self.pref_schedule.prefs[0]
        print("The vote:", the_man)
        winner.append(the_man[0])
        print("The winner(s) is(are):", winner)
        counts={}
        for c in self.candidates:
            if c is winner[0]:
                counts[c]=1
            else:
                counts[c]=0
        return counts
    
    def condorcet(self):
        "Returns the condorcet winner if it exists"
        prefs = self.pref_schedule.prefs
        pairwise_wins = []
        for c1 in self.candidates:
            for c2 in self.candidates:
                c1_votes = 0
                c2_votes = 0
                if c1 is not c2:
                    for p in prefs:
                        if p.index(c1) < p.index(c2):
                            c1_votes += 1    
                        else:
                            c2_votes += 1
                            
                    if c1_votes > c2_votes:
                        winpair = (c1,c2)
                    elif c1_votes == c2_votes:
                        winpair = None
                    else:
                        winpair = (c2,c1)
                    if winpair not in pairwise_wins:
                        pairwise_wins.append(winpair)
        print(pairwise_wins)
        winners = [w[0] for w in pairwise_wins]
        print(winners)
        for c in self.candidates:
            if winners.count(c) is len(self.candidates)-1:
                print("Condorcet winner is:", c)
                return
        print("No Condorcet winner")            


    def plurality(self):
        '''Prints who wins by the plurality method'''

        counts = {}
        for pref in self.pref_schedule.prefs:
            highest = pref[0]
            if highest in counts:
                counts[highest] += 1
            else:
                counts[highest] = 1

        winner = []
        highest_votes = max(counts.values())
        for candidate in counts:
            if counts[candidate] == highest_votes:
                winner.append(candidate)

        for c in self.candidates:
            if c not in counts.keys():
                counts[c] = 0

        #print('The numbers of votes for each candidate:', counts)
        #print('The winner(s) is(are)', find_winner(counts))
        return counts

    def runoff(self):
        '''Prints who wins by the runoff method'''

        # first round
        counts = {}
        for pref in self.pref_schedule.prefs:
            highest = pref[0]
            if highest in counts:
                counts[highest] += 1
            else:
                counts[highest] = 1

        first_round_winners = []
        scores = list(counts.values())
        highest_votes = max(scores)
        while highest_votes in scores:
            scores.remove(highest_votes)
        second_highest_votes = max(scores)
        for candidate in counts:
            if counts[candidate] == highest_votes:
                first_round_winners.append(candidate)
        if len(first_round_winners) == 1:
            for candidate in counts:
                if counts[candidate] == second_highest_votes:
                    first_round_winners.append(candidate)

        print('The numbers of votes for each candidate in the first round:', counts)
        print('The first round winners are', first_round_winners)

        # second round
        counts = {c: 0 for c in first_round_winners}
        for candidate in first_round_winners:
            for pref in self.pref_schedule.prefs:
                ranks = [pref.index(c) for c in first_round_winners]
                if pref.index(candidate) == min(ranks):
                    counts[candidate] += 1

        print('The numbers of votes for each candidate in the second round:', counts)
        print('The winner(s) is(are)', find_winner(counts))
        return counts

    def elimination(self):
        '''Prints who wins by the elimination method'''

        num_round = 1
        candidates = self.candidates[:]
        prefs = copy.deepcopy(self.pref_schedule.prefs)

        while len(candidates) >= 2:
            counts = {}
            for pref in prefs:
                highest = pref[0]
                if highest in counts:
                    counts[highest] += 1
                else:
                    counts[highest] = 1
            print('The numbers of votes for each candidate (round {}):'.format(num_round), counts)

            lowest_votes = min(counts.values())
            for candidate in counts:
                if counts[candidate] == lowest_votes:
                    candidates.remove(candidate)
                    for pref in prefs:
                        pref.remove(candidate)

            num_round += 1

        print('The winner(s) is(are)', find_winner(counts))
        return counts

    def borda(self):
        '''Prints who wins by the Borda count'''
        counts = {}
        candidates = list(self.pref_schedule.prefs[0])
        for candidate in candidates:
            counts[candidate] = 0

        max_point = len(candidates)-1
        #print("\n\n\nCandidates:", candidates)
        #print("Prefs", self.pref_schedule.prefs)
        for pref in self.pref_schedule.prefs:
            #print(pref)
            for i in range(len(pref)):
                #print(pref[i], (max_point-i))
                counts[pref[i]] += max_point - i
        #!!! REMOVE THIS TO PRINT
        #print('Borda scores:', counts)
        #print('The winner(s) is(are)', find_winner(counts))
        return counts
    
    def pairwise_comparison(self):
        '''Prints who wins by the pairwise comparison method'''

        points = {candidate: 0 for candidate in self.candidates}
        candidates = list(self.candidates)
        for candidate in candidates[:]:
            candidates.remove(candidate)
            for rival in candidates:
                candidate_points = 0
                for pref in self.pref_schedule.prefs:
                    if pref.index(candidate) < pref.index(rival):
                        candidate_points += 1
                    else:
                        candidate_points -= 1
                if candidate_points > 0:
                    points[candidate] += 1
                else:
                    points[rival] += 1

        return points

def find_winner(aggregated_result):
    max_point = 0
    for point in aggregated_result.values():
        if point > max_point:
            max_point = point

    winner = []  # winner can be many, so use a list here
    for candidate in aggregated_result.keys():
        if aggregated_result[candidate] == max_point:
            winner.append(candidate)

    return winner


#    If A is preferred to B out of the choice set {A,B}, introducing a third option X, expanding the choice set to {A,B,X}, must not make B preferable to A.
def check_for_dependence(old_counts,new_counts,irrelevant,candidates):
    for A in candidates:
        for B in candidates:
            if A != B:
                if (A not in irrelevant) or (B not in irrelevant):
                    if (old_counts[A]>old_counts[B]) and (new_counts[B]>new_counts[A]):
                        return True
    return False


def remove_irrelevant(agg,voting_method):

    "Checks for dependence on irrelevant alternatives by randomly flipping two adjacent votes and checking for shifts in other votes"

    dependence = False
    #pbar = tqdm()
    blockPrint()
    count = 0
    while dependence == False and (count < LIM):
        count += 1
        #pbar.update()
        original_pref_schedule = agg.pref_schedule

        #counts = getattr(agg,voting_method)() #CHANGE
        
        #ranked = sorted(counts.items(),key=lambda x : x[1], reverse=True)
        #ranked = [x[0] for x in ranked]

        candidates = agg.candidates

        #relevant = ranked[2:] #NEED TO FIX
        #n = rand.randint(0,len(candidates)-2)
        #irrelevant = ranked[n:n+2]
        irrelevant = sample(candidates,2)
        relevant = [c for c in candidates if c not in irrelevant]



        #print(relevant)
        #print(irrelevant)
        new_prefs = []
        for row in original_pref_schedule.prefs:
            new_row = row.copy()
            irrelevant = [c for c in candidates if c not in relevant]
            irr_indexes = []
            for idx,item in enumerate(row):
                if item in irrelevant:
                    irr_indexes.append(idx)
            
            irr_indexes = sample(irr_indexes,2)
            if (irr_indexes[1] is irr_indexes[0]+1) or (irr_indexes[1] is irr_indexes[0]-1):
                if random() > 0.5:
                    new_row[irr_indexes[0]]=row[irr_indexes[1]]
                    new_row[irr_indexes[1]]=row[irr_indexes[0]]

            new_prefs.append(new_row)
        removed_pref_schedule = PreferenceSchedule(candidates,new_prefs)
        agg_new = copy.copy(agg)
        agg_new.pref_schedule = removed_pref_schedule

        """
        print("\n\n\n ========================== \n\n\nOriginal Pref_schedule:")
        print(agg)
        print("\n\nIrrelevant: ", irrelevant)
        print("Irrelevants-shuffled Pref_schedule:")
        print(agg_new)
        """
        #input()

        irrelevant = [c for c in candidates if c not in relevant]
        old_counts = getattr(agg,voting_method)() #CHANGE
        new_counts = getattr(agg_new,voting_method)()#CHANGE

        dependence = check_for_dependence(old_counts,new_counts,irrelevant,candidates)

        if dependence == True:
            enablePrint()
            print("Original Pref schedule:")
            print(agg)
            print("\nIrrelevant: ", irrelevant)
            print("\n\nIrrelevants-shuffled Pref schedule:")
            print(agg_new)
            print("Old Winners:", old_counts)
            print("Shuffled Winners:", new_counts)
            print("Irrelevant: ", irrelevant)

            #print("\n\nCandidate".ljust(20), "Ranked above before".ljust(30), "Ranked above or evenly after")
            #print()
            #print(c.ljust(20), str(RA).ljust(30),str(RA_even_new))
            print("DEPENDENCE ON IRRELEVANT Detected")
            return True
    return False


def csv_to_preference_schedule(file):
    '''Reads a csv file and returns candidates and preferences'''

    file_name, ext = os.path.splitext(file)
    if file not in os.listdir('.'):
        raise InputError('File does not exist')
    if ext != '.csv':
        raise InputError('File must be a csv file')

    with open(file) as f:
        candidates = None
        prefs = []
        reader = csv.reader(f)
        for row in reader:
            if candidates is None:
                candidates = list(row)
                prefs.append(list(row)) #I think???
            else:
                prefs.append(list(row))
        return candidates, prefs

def copy_prob(v_n,corr):
    n_voters = v_n #number of voters before this one. should always be zero
    beta = (1/corr) - 1
    copy_prob = 1 - (beta / (beta + n_voters))
    return copy_prob

def autogenerate_votes(n_candidates,n_voters,corr=None):
    
    options = list(string.ascii_uppercase)
    candidates = options[:n_candidates]
    votes = []

    for voter_n in range(n_voters):
        vote = candidates.copy()
        rand.shuffle(vote)

        if corr is not None:
            cp = copy_prob(voter_n,corr)
            if rand.random() < cp:
                try:
                    prev = votes[-1]
                except Exception:
                    print(votes)
                    print(cp)
                    print("Tried to copy too early!")
                votes.append(prev)
            else:
                votes.append(vote)
        else:
            votes.append(vote)

    return (candidates,votes)

def test_n(cands,voters,tries):
    dependence = False
    for _ in range(tries):
        aggr = Aggregator(inputs=autogenerate_votes(cands,voters))
        answer = remove_irrelevant(aggr,rule)
        if answer == True:
            dependence = True
            return dependence
    return dependence

