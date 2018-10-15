import numpy as np
from copy import deepcopy
from animation import Animation
from Agent import *
from utils import *


class simple_soccer_env():

    def get_dis_with_ball(self, id_a):
        
        #print('A',id_a)
        #print(self.pos_agent[id_a][0],  self.pos_ball[0])
        #print(self.pos_agent[id_a][0],  self.pos_ball[0])
        return np.sqrt((self.pos_agent[id_a][0] - self.pos_ball[0]) ** 2 + (self.pos_agent[id_a][1] - self.pos_ball[1]) ** 2)

    def calc_p_of_steal(self, id):

        r = self.get_dis_with_ball(id)
        if r > self.r_touch:return 0
        return 0.1
        if self.id_ball_belongs_to == -1 and r <= self.r_touch:
            return 1
        if r <= self.r_touch:
            return 0.5 + (self.r_touch-r)*0.5 # 现在很大
        else: return 0

    def outside(self, a):
        return a[0]<0 or a[0]>self.w or a[1] < 0 or a[1] > self.l

    def limit(self, a):
        a[0] = max(0,a[0])
        a[0] = min(self.w, a[0])
        a[1] = max(0,a[1])
        a[1] = min(self.l, a[1])
    
    def generate_position(self):

        self.pos_agent = [[rand() * self.w, rand() * self.l] for i in range(self.n)]
        #self.pos_ball = [rand() * self.w, rand() * self.l]
        self.pos_ball = [0.5 * self.w, 0.5 * self.l]             # change
        self.c = [0 for i in range(self.n)]

    def get_info(self):

        dict = {'w':self.w,
                'l':self.l,
                'w_goal':self.w_goal,
                'dt':self.dt,
                'alpha':self.alpha,
                'limit_r':self.r_touch,
                'Max_v_agent':self.Mv,
                'Max_v_ball':self.Mvb,
                'Mv':self.Mv,
                'color':self.color,
                'n':self.n
                }

        return dict

    def __init__(self, color):

        self.w, self.l = 68, 105  #width and length of court
        self.w_goal = 7.32 # width of the goal
        
        self.n = len(color)
        self.color = color #array of teams' colors agents belong to

        self.dt = 1.
        self.alpha = 0.1 # 阻尼系数
        self.Mv = 3.0
        self.Mv_wb = 2.9
        self.Mvb = 20.0

        self.panalty_agent_go_outside = 0.0
        self.panalty_ball_go_outside = 0.0
        self.reward_control = 1.0
        self.reward_goal = 10.0
        self.r_touch = 5. #radius of the range that the ball can be touched

        self.pos_agent = []
        self.v_agent = []
        self.c = [] #是否控球

        self.vis_agent = []
        self.n_vis = 0

        self.action = []
        self.reward = []
        self.done = False
        self.step_ = 0

        self.pos_ball = []
        self.v_ball = []
        self.id_ball_belongs_to = -1
        self.last_id = -1

        self.message_pool = []

    def return_env_state(self):

        return [self.pos_agent[j] for j in range(self.n) if self.color[j] == 0], [self.pos_agent[j] for j in range(self.n) if self.color[j] == 1], self.pos_ball, self.v_ball

    def reset(self):

        self.generate_position()
        
        self.v_agent = [[0,0] for i in range(self.n)]
        self.v_ball = [0,0]                            # change

        self.vis_agent = [0 for i in range(self.n)]
        self.n_vis = 0

        self.id_ball_belongs_to = -1
        self.last_id = -1
        self.action = [[] for i in range(self.n)]
        self.message_pool = [[]for i in range(self.n)]

        self.reward = [0 for i in range(self.n)]
        self.done = False
        self.step_ = 0

    def get_message(self, id_a, id_b, info):

        self.message_pool[id_b].append([id_a, info])  # message_pool[id_a]收录所有发给id_a的信息，每条信息格式为[id_b, info]
    
    def step(self): #演算一步
        
        self.step_ += 1
        #判断足球是否要被踢出
        flag = 0
        if self.id_ball_belongs_to != -1:
            self.pos_ball = deepcopy(self.pos_agent[self.id_ball_belongs_to])
            self.v_ball = deepcopy(self.v_agent[self.id_ball_belongs_to])
            if self.action[self.id_ball_belongs_to][2] == True:
                self.v_ball = deepcopy(self.action[self.id_ball_belongs_to][3])
                self.id_ball_belongs_to = -1
                print('kicked out')
                flag = 1
                
        if flag == 0:
            #先判断有谁会抢断
            id_steal = []
            su = []
            p = 1
            for i in range(self.n):
                if self.action[i][1] == True and self.c[i] == 0 and ((self.id_ball_belongs_to == -1) or self.color[i] != self.color[self.id_ball_belongs_to]):
                    #print('steal_get', self.id_ball_belongs_to, i)
                    id_steal.append(i)
                    su.append(self.calc_p_of_steal(i))
                    #print('B', su)
                    p *= (1-su[-1])
            p = 1 - p

            #再计算谁该实现抢断
            if rand() < p:
                x = rand() * np.sum(su)
                for i in range(len(su)):
                    if su[i] >= x:
                        if self.id_ball_belongs_to != -1:
                            self.c[self.id_ball_belongs_to] = 0
                        self.id_ball_belongs_to = id_steal[i]
                        self.last_id = id_steal[i]
                        self.pos_ball = self.pos_agent[id_steal[i]]
                        self.v_ball = self.v_agent[id_steal[i]]
                        self.c[id_steal[i]] = 1
                        print('get_ball',id_steal[i])
                        break
                    x -= su[i]

        #把agent和足球的位置演算一步 并计算是否进门
        for i in range(self.n):
            self.v_agent[i] = self.action[i][0]
            add(self.pos_agent[i], self.dt, self.v_agent[i])
        new_pos_ball = deepcopy(self.pos_ball)
        add(new_pos_ball, self.dt, self.v_ball)

        #计算是否进上面的门
        if self.pos_ball[1] < self.l and new_pos_ball[1] > self.l:
            print('goal_up!!!')
            mid = self.pos_ball[0] * (new_pos_ball[1] - self.l) + new_pos_ball[0] * (self.l - self.pos_ball[1])
            mid /= new_pos_ball[1] - self.pos_ball[1];
            if np.fabs(mid - self.w * 0.5) <= self.w_goal * 0.5:
                for i in range(self.n):
                    if self.color[i] == 0:
                        self.reward[i] += self.reward_goal
                    else:self.reward[i] -= self.reward_goal
                self.done = True
                return

        #计算是否进下面的门
        if self.pos_ball[1] > 0 and new_pos_ball[1] < 0:
            print('goal_down!!!')
            mid = self.pos_ball[0] * (self.pos_ball[1]) + new_pos_ball[0] * (- new_pos_ball[1])
            mid /= self.pos_ball[1] - new_pos_ball[1];
            if np.fabs(mid - self.w * 0.5) <= self.w_goal * 0.5:
                for i in range(self.n):
                    if self.color[i] == 1:
                        self.reward[i] += self.reward_goal
                    else:self.reward[i] -= self.reward_goal
                self.done = True
                return

        self.pos_ball = new_pos_ball
        add(self.v_ball, -self.alpha, self.v_ball)
        
        #计算出界问题
        for i in range(self.n):
            if self.outside(self.pos_agent[i]):
                self.limit(self.pos_agent[i])
                if self.c[i]:
                    self.v_ball = self.v_agent[i]
                    self.pos_ball = self.pos_agent[i]
                self.reward[i] += self.panalty_agent_go_outside
        
        #球归于最近的人对手，给踢出去的人惩罚
        if self.outside(self.pos_ball):
            print('outside ball',self.pos_ball)
            self.reward[self.last_id] += self.panalty_ball_go_outside
            mi = 1e100
            for i in range(self.n):
                if self.color[i] != self.color[self.last_id] and self.get_dis_with_ball(i) > mi:
                    mi = self.get_dis_with_ball(i)
                    self.last_id = i
            self.id_ball_belongs_to = self.last_id
            self.v_ball = self.v_agent[self.last_id]
            self.pos_ball = self.pos_agent[self.last_id]


    def get_action(self, id, action): # action的格式为[v = [R,R], shoot_or_not = Bool, v_ball = [R,R]] 如果shoot_or_not = False, v_ball不存在或无效

        temp = self.Mv 
        if self.id_ball_belongs_to == id:temp = self.Mv_wb # 带球的agent运动更慢
        length_v_agent = np.sqrt(action[0][0] ** 2 + action[0][1] ** 2)
        action[0][0] /= max(temp, length_v_agent) / temp
        action[0][1] /= max(temp, length_v_agent) / temp

        length_v_ball  = np.sqrt(action[3][0] ** 2 + action[3][1] ** 2)
        action[3][0] /= max(self.Mvb, length_v_agent) / self.Mvb
        action[3][1] /= max(self.Mvb, length_v_agent) / self.Mvb

        self.action[id] = action
        if self.vis_agent[id] == 0:
            self.n_vis += 1
            self.vis_agent[id] = 1
            #print(self.n_vis)
        
        if self.n_vis == self.n:   # 目前的设定是所有人都决策之后env演算一步
            self.step()
            self.vis_agent = [0 for i in range(self.n)]
            self.n_vis = 0

    def return_observation(self, id):   #另外其他agent传输的信息要在observation之前还是之后？个人认为应该之后？因为当次observation要耗时

        self_state = {'pos_agent':self.pos_agent[id],
                      'v_agent':self.v_agent[id],
                      'control_ball':self.c[id],}
        ball_state = {'pos_ball':self.pos_ball,
                      'v_ball':self.v_ball}

        other_state = []
        for i in range(self.n):
            #if i != id: #change here
            i_state = {'pos_agent':self.pos_agent[i],
                        'v_agent':self.v_agent[i],
                        'control_ball':self.c[i],}
            other_state.append([i,i_state])

        state = {'self_state':self_state,
                 'ball_state':ball_state,
                 'other_state':other_state,
                 'id_ball_belongs_to':self.id_ball_belongs_to,
                 'message':self.message_pool[id]
                 }

        #self_state = [self.pos_agent[id], self.v_agent[id], self.c[id], self.pos_ball, self.v_ball]
        self.message_pool[id] = []
        reward = self.reward[id]
        self.reward[id] = 0
        #一个人所接收到的state是自身位置，速度，是否可控球，球的位置，速度，以及其他agent传输的信息
        return [state, reward, self.done, None]

def test():
    
    n = 4;
    env = simple_soccer_env([0,0,1,1])     #定义环境的时候传入一个n元数组表示每一个agent的阵营
    env.reset()

    agents = [Agent_heuristic(env.get_info(), i) for i in range(n)] 
    #最愚蠢的agent，向足球靠近，然后尽可能抢断，之后有10%的概率朝随便方向踢出去
    #只是用来表示agent需要做什么

    

    t0s, t1s, bs = [], [], []
    t0, t1, b, vb = env.return_env_state()# 这些变量用来记录env信息做可视化，当一局完全结束后，比赛才可变为视频
    
    t0s.append(deepcopy(t0))
    t1s.append(deepcopy(t1))
    bs.append(deepcopy(b))
#   print(t0,t1,b,vb)
    while env.done == False and env.step_ <= 100: # 默认最多演算100步，若射门(done = True)则提前退出
#       print(env.step_)
#       print('       ',env.id_ball_belongs_to)
        for i in range(n):
            agents[i].get_observation(env.return_observation(i)) # 每一个agent先获得observation,reward,done,info

        for i in range(n):
            env.get_action(i, agents[i].return_action()) # 每一个agent一起把action传给env

        for i in range(n):
            for j in range(n):
                if i != j:
                    env.get_message(i,j,agents[i].send_message(j))#i在这里决定给j什么信息
        
        t0, t1, b, vb = env.return_env_state()
        #print(t0,t1,b,vb)
        t0s.append(deepcopy(t0))
        t1s.append(deepcopy(t1))
        bs.append(deepcopy(b))

    Animation([t0s, t1s], bs, env.w_goal, env.w, env.l)

test()