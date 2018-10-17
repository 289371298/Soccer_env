import numpy as np
from utils import *

class Agent_naive:

    def __init__(self):

        self.pos_ball, self.pos_agent = 0,0

    def get_observation(self, obs):

        self.pos_ball =  np.array(obs[0]['ball_state']['pos_ball'])
        self.pos_agent = np.array(obs[0]['self_state']['pos_agent'])
        
    def return_action(self):
        
        return [(self.pos_ball - self.pos_agent)/10+rand()-0.5,
               True,
               rand() < 0.1,
              [(rand()-0.5) * 10,(rand()-0.5) * 10]]

    def send_message(self, id_to):

        return None

class Agent_heuristic(Agent_naive):

    def __init__(self, env_info, id):

        self.id = id
        self.env_info = env_info
        self.v_ball,self.pos_ball = [],[]
        self.id_ball_belongs_to = -1
        self.pos_goal = np.array([env_info['w']/2,env_info['l']])
        self.pos_agent = []
        self.other_agent = []
        self.n = env_info['n']
        if self.id < self.n // 2:self.pos_goal -= [0, env_info['l']]

    def go_outside(self, a):

        return not(a[0] >= 0 and a[0] <= self.env_info['w'] and a[1] >= 0 and a[1] <= self.env_info['l'])

    def get_pos_ball(self, v, t):

        temp = np.array(v, dtype = np.float)
        su = np.array([.0,.0])
        while t:
            t -= 1
            #print(temp, self.env_info['dt'])
            su += self.env_info['dt'] * temp
            temp *= self.env_info['alpha']
        return su + self.pos_ball

    def get_observation(self, obs):

        state = obs[0]
        self.id_ball_belongs_to = state['id_ball_belongs_to']
        self.v_ball = state['ball_state']['v_ball']
        self.pos_ball = state['ball_state']['pos_ball']
        self.pos_agent = np.array(state['self_state']['pos_agent'])
        self.other_agent = [None for i in range(self.n)]
        for i in range(len(state['other_state'])):
            self.other_agent[state['other_state'][i][0]] = state['other_state'][i][1]

    def xmul(self, a, b, c):

        return (c[1] - a[1]) * (b[0] - a[0]) - (c[0] - a[0]) * (b[1] - a[1])


    def return_action(self):

        v_agent = None
        
        will_get = -1

        if self.id_ball_belongs_to == -1:
            for i in range(1,100):
                pos_ball_temp = self.get_pos_ball(self.v_ball, i)
                #print('pos_ball',pos_ball_temp,i)
                if self.go_outside(pos_ball_temp):break
                dis = get_dis(self.pos_agent, pos_ball_temp)
                mi = 1e10
                for j in range(len(self.other_agent)):
                    if self.env_info['color'][j] != self.env_info['color'][self.id]:
                        pos = self.other_agent[j]['pos_agent']
                        dis_enemy = get_dis(pos, pos_ball_temp)
                        if dis_enemy < mi:
                            mi = dis_enemy
                            will_get = j
                time_of_enemy = mi / self.env_info['Mv']
                time_of_mine = dis / self.env_info['Mv']
                #print('dis',i,pos_ball_temp,time_of_enemy,time_of_mine)
                if time_of_mine > i:continue
                if time_of_mine < time_of_enemy:
                    v_agent = pos_ball_temp - self.pos_agent
                    print('rob!!!')
                    break
                else:break

        if v_agent is None:

            if self.id_ball_belongs_to != -1:
                will_get = self.id_ball_belongs_to

            targetA = self.other_agent[will_get]['pos_agent']
            other_enemy = will_get ^ 1
            other_ally = self.id ^ 1
            targetB = get_fd(self.other_agent[other_enemy]['pos_agent'],
                             self.other_agent[will_get]['pos_agent'],
                             0.9)
            orth_self = get_orth_point(self.pos_goal, self.pos_ball,self.pos_agent)
            orth_ally = get_orth_point(self.pos_goal, self.pos_ball,self.other_agent[other_ally]['pos_agent'])
            dis_self = get_dis(get_fd(orth_self,targetA,0.5), self.other_agent[self.id]['pos_agent'])
            dis_ally = get_dis(get_fd(orth_ally,targetA,0.5), self.other_agent[other_ally]['pos_agent'])
            print('orth',self.id, orth_self, orth_ally,self.pos_agent, dis_self, dis_ally)
            if dis_self < dis_ally:
                a = get_dis(self.pos_goal, targetA)
                b = get_dis(targetA, self.pos_agent)
                c = get_dis(self.pos_agent, self.pos_goal)
                if b < 0.5:
                    v_agent = np.array(self.other_agent[will_get]['pos_agent'])+self.other_agent[will_get]['v_agent']-self.pos_agent
                else:
                    cos_theta = ((a ** 2 + b ** 2 - c ** 2)/(2*a*b))
                    #print('cos ',a,b,c,cos_theta)
                    len_orth = b * np.sqrt(1 - cos_theta ** 2)
                    v_agent = np.array(self.pos_ball) - self.pos_goal
                    normalize(v_agent, 1)
                    orth_point = self.pos_agent + np.array([-v_agent[1], v_agent[0]]) * len_orth
                    d = self.pos_goal - orth_point
                    v1 = (orth_point - self.pos_agent) * 5
                    if self.xmul(self.pos_ball, self.pos_agent,self.pos_goal) > 0:
                        v1 = - v1
                    #if get_dis(v1) > self.env_info['Mv']:
                    #    normalize(v1, self.env_info['Mv'])
                    v2 = v_agent * b * cos_theta
                    if self.id != -1:
                        print('     before',self.id, v_agent, len_orth)
                        print('        aha',v1,v2,orth_point,self.pos_agent)
                    v_agent = v1 + v2
            else:
                v_agent = np.array(targetB) - self.pos_agent

        print('debug', self.id,will_get,self.pos_agent,v_agent)

        v_ball = self.pos_goal - self.pos_agent
        return [v_agent,
                True,
                True,
                v_ball]

        