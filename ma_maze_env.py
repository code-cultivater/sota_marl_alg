#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simulate a predator prey environment.
Each agent can just observe itself (it's own identity) i.e. s_j = j and vision sqaure around it.
Design Decisions:
    - Memory cheaper than time (compute)
    - Using Vocab for class of box:
         -1 out of bound,
         indexing for predator agent (from 2?)
         ??? for prey agent (1 for fixed case, for now)
    - Action Space & Observation Space are according to an agent
    - Rewards -0.05 at each time step till the time
    - Episode never ends
    - Obs. State: Vocab of 1-hot < predator, preys & units >
"""

# core modules
import random
import math
import curses
import tkinter as tk
import copy

# 3rd party modules
import gym
import numpy as np
from gym import spaces


class MeetEnv(gym.Env):
    # metadata = {'render.modes': ['human']}

    def __init__(self,):
        self.__version__ = "0.0.1"

        # TODO: better config handling
        self.episode_over = False
        self.multi_agent_init()
        self.init_args()

        self.init_windows()
    def get_env_info(self):
        env_info={
            "n_actions":self.naction,
            "n_agents":self.n_player,
            "state_shape":2*self.n_player,
            "obs_shape":2*self.n_player,
            "episode_limit":20,
        }
        return env_info
    def get_obs(self):
        return [np.concatenate(copy.deepcopy(self.players_pos)),np.concatenate(copy.deepcopy(self.players_pos))]
    def get_state(self):
        return np.concatenate(copy.deepcopy(self.players_pos))##???
    def get_avail_agent_actions(self,agent_id):
        return np.ones(self.naction)
    def init_windows(self):
        self.windows = tk.Tk()
        self.windows.title("simple_ma_env")
        self.space=100
        self.colors = ["red", 'yellow']
        self.windows.geometry("500x500")
        self.canvas = tk.Canvas(self.windows, bg='white', height=500, width=500)
        for i in range(self.dim + 1):
            self.canvas.create_line(0, 0 + i * self.space, 500, 0 + i * self.space)
            self.canvas.create_line(0 + i * self.space, 0, 0 + i * self.space, 500)
        self.ovals = []
        for i in range(self.n_player):
            self.ovals.append(self.canvas.create_oval(self.players_pos[i][1] * self.space, self.players_pos[i][0] * self.space,
                                            self.players_pos[i][1] * self.space + self.space,self.players_pos[i][0] * self.space + self.space,
                                            fill=self.colors[i]))
        self.canvas.pack()

    def init_args(self):
        self.dim=5
        self.dims = (self.dim, self.dim)
        self.naction = 5
        self.vision=2
        self.reward=10
        self.players_pos = [[0, 0], [self.dim - 1, self.dim - 1]]
        self.prev_players_pos=copy.deepcopy(self.players_pos)
        self.grid = self._set_grid()
        for i in range(self.n_player):
            player_i_x, player_i_y = self.players_pos[i][0], self.players_pos[i][1]
            self.grid[player_i_x][player_i_y] = self.grid[player_i_x][player_i_y] + str(i)
    def multi_agent_init(self):
        self.n_player = 2


        return

    def reset(self):
        #self.windows.update()
        self.episode_over = False
        self.init_args()
        self.multi_agent_init()
        for oval in self.ovals:
            self.canvas.delete(oval)
        self.ovals=[]
        for i in range(self.n_player):
            self.ovals.append(self.canvas.create_oval(self.players_pos[i][1] * self.space, self.players_pos[i][0] * self.space,
                                            self.players_pos[i][1] * self.space + self.space,self.players_pos[i][0] * self.space + self.space,
                                            fill=self.colors[i]))

        return self._get_obs()


    def _get_obs(self):
        poses= [self.players_pos[i] for i in range(self.n_player)]
        return  poses,self.grid

    def step(self, action):
        if self.episode_over:
            raise RuntimeError("Episode is done")
        for i, a in enumerate(action):
            self._take_action(i, a)


        self.obs = self._get_obs()
        self.episode_over = False
        if(self._get_reward()==self.reward) :self.episode_over=True
        debug = {'player_pos':self.players_pos,'grid':self.grid}

        for i in range(self.n_player):
            self.canvas.move(self.ovals[i], (self.players_pos[i][1] - self.prev_players_pos[i][1]) * self.space,
                             (self.players_pos[i][0] - self.prev_players_pos[i][0]) * self.space)
        return self.obs, self._get_reward(), self.episode_over, debug

    def _set_grid(self):
        grid=np.full(self.dims,"",dtype=np.object)
        return  grid

    def _take_action(self, idx, act):
        self.prev_players_pos[idx]=[self.players_pos[idx][0],self.players_pos[idx][1]]
        #print(id(self.prev_players_pos[idx][0]))
        #print(self.players_pos[idx][0])
        player_i_x,player_i_y=self.players_pos[idx]
        if act==0:
            pass
        # UP
        if act==1 and player_i_x!=0:
            self.grid[player_i_x - 1][player_i_y] =self.grid[player_i_x - 1][player_i_y]+ str(idx)
            self.grid[player_i_x ][player_i_y] = self.grid[player_i_x ][player_i_y].replace(str(idx),"")
            self.players_pos[idx]= [player_i_x - 1,player_i_y]
        # DOWN
        elif act == 2 and player_i_x != self.dims[0] - 1 :
            self.grid[player_i_x + 1][player_i_y] = self.grid[player_i_x + 1][player_i_y] + str(idx)
            self.grid[player_i_x][player_i_y] = self.grid[player_i_x][player_i_y].replace(str(idx), "")
            self.players_pos[idx] = [player_i_x + 1, player_i_y]
        # LEFT
        elif act == 3 and player_i_y != 0:
            self.grid[player_i_x][player_i_y - 1] = self.grid[player_i_x][player_i_y - 1] + str(idx)
            self.grid[player_i_x][player_i_y] = self.grid[player_i_x][player_i_y].replace(str(idx), "")
            self.players_pos[idx] = [player_i_x, player_i_y - 1]
        # RIGHT
        elif act==4 and player_i_y != self.dims[1]-1:
            self.grid[player_i_x][player_i_y + 1] = self.grid[player_i_x][player_i_y + 1] + str(idx)
            self.grid[player_i_x][player_i_y] = self.grid[player_i_x][player_i_y].replace(str(idx), "")
            self.players_pos[idx] = [player_i_x, player_i_y + 1]



    def _get_reward(self):
        if(((np.array(self.players_pos[0])==np.array(self.players_pos[1]))).all()):
            return self.reward
        else: return  0
    def render(self, mode='human'):
        self.windows.update()


