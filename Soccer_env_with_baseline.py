import numpy as np
from copy import deepcopy
from animation import Animation


def rand():
    return np.random.random_sample()


def add(a, t, b):
    for i in range(len(a)):
        a[i] += t * b[i]


class simple_soccer_env():

    def get_dis_with_ball(self, id_a):

        # print('A',id_a)
        # print(self.pos_agent[id_a][0],  self.pos_ball[0])
        # print(self.pos_agent[id_a][0],  self.pos_ball[0])
        return np.sqrt(
            (self.pos_agent[id_a][0] - self.pos_ball[0]) ** 2 + (self.pos_agent[id_a][1] - self.pos_ball[1]) ** 2)

    def calc_p_of_steal(self, id):

        r = self.get_dis_with_ball(id)
        print("stealing",end=" agent number=")
        print(id,end=" current_agent_position")
        print(self.pos_agent[id],end=" ")
        print(self.pos_ball)
        print("r=",end=" ")
        print(r)
        if r > self.r_touch: return 0
        # return 0.1
        #print("steal",end=" ")
        #print(r,end=" ")
        #print(self.r_touch)
        if self.id_ball_belongs_to == -1 and r <= self.r_touch:
            return 1
        if r <= self.r_touch:
            return 0.1 + (self.r_touch - r) * 0.6  # 现在很大
        else:
            return 0

    def outside(self, a):
        return a[0] < 0 or a[0] > self.w or a[1] < 0 or a[1] > self.l

    def limit(self, a):
        a[0] = max(0, a[0])
        a[0] = min(self.w, a[0])
        a[1] = max(0, a[1])
        a[1] = min(self.l, a[1])

    def generate_position(self):

        self.pos_agent = [[rand() * self.w, rand() * self.l] for i in range(self.n)]
        #self.pos_agent=[[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
        #for i in range(14):self.pos_agent.append([0.5*self.w,0.5*self.l])
        # self.pos_ball = [0.5 * self.w, 0.5 * self.l]
        self.pos_ball = [0.5 * self.w, 0.5 * self.l]  # change
        self.c = [0 for i in range(self.n)]

    def get_info(self):

        dict = {'w': self.w,
                'l': self.l,
                'w_goal': self.w_goal,
                'dt': self.dt,
                'alpha': self.alpha,
                'limit_r': self.r_touch,
                'Max_v_agent': self.Mv,
                'Max_v_ball': self.Mvb,
                'color': self.color
                }

    def __init__(self, color):
        global globalColor
        self.w, self.l = 68, 105  # width and length of court
        self.w_goal = 7.32  # width of the goal

        self.n = len(color)
        self.color = color  # array of teams' colors agents belong to
        globalColor=color
        self.dt = 1.
        self.alpha = 0.1  # 阻尼系数
        self.Mv = 3.0
        self.Mvb = 20.0

        self.panalty_agent_go_outside = 0.0
        self.panalty_ball_go_outside = 0.0
        self.reward_control = 1.0
        self.reward_goal = 10.0
        self.r_touch = 2.5  # radius of the range that the ball can be touched(modified by experience from Teacher

        self.pos_agent = []
        self.v_agent = []
        self.c = []  # 是否控球

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
        self.race_starts = 0  # Does anyone touch the ball for the first time

    def return_env_state(self):

        return [self.pos_agent[j] for j in range(self.n) if self.color[j] == 0], [self.pos_agent[j] for j in
                                                                                  range(self.n) if self.color[
                                                                                      j] == 1], self.pos_ball, self.v_ball

    def reset(self):

        self.generate_position()

        self.v_agent = [[0, 0] for i in range(self.n)]
        self.v_ball = [0, 0]  # change

        self.vis_agent = [0 for i in range(self.n)]
        self.n_vis = 0

        self.id_ball_belongs_to = -1
        self.last_id = -1
        self.action = [[] for i in range(self.n)]
        self.message_pool = [[] for i in range(self.n)]

        self.reward = [0 for i in range(self.n)]
        self.done = False
        self.step_ = 0

    def get_message(self, id_a, id_b, info):

        self.message_pool[id_b].append([id_a, info])  # message_pool[id_a]收录所有发给id_a的信息，每条信息格式为[id_b, info]
        #print(id_a,id_b,info)

    def step(self):  # 演算一步

        self.step_ += 1
        # 判断足球是否要被踢出
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
            # 先判断有谁会抢断
            id_steal = []
            su = []
            p = 1
            for i in range(self.n):
                if self.action[i][1] == True and self.c[i] == 0 and (
                        (self.id_ball_belongs_to == -1) or self.color[i] != self.color[self.id_ball_belongs_to]):
                    # print('steal_get', self.id_ball_belongs_to, i)
                    id_steal.append(i)
                    su.append(self.calc_p_of_steal(i))
                    # print('B', su)
                    p *= (1 - su[-1])
            p = 1 - p

            # 再计算谁该实现抢断
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
                        print('get_ball', id_steal[i])
                        break
                    x -= su[i]

        # 把agent和足球的位置演算一步 并计算是否进门
        for i in range(self.n):
            self.v_agent[i] = self.action[i][0]
            add(self.pos_agent[i], self.dt, self.v_agent[i])
        new_pos_ball = deepcopy(self.pos_ball)
        add(new_pos_ball, self.dt, self.v_ball)

        # 计算是否进上面的门
        if self.pos_ball[1] < self.l and new_pos_ball[1] >= self.l:
            print('goal_up!!!')
            mid = self.pos_ball[0] * (new_pos_ball[1] - self.l) + new_pos_ball[0] * (self.l - self.pos_ball[1])
            mid /= new_pos_ball[1] - self.pos_ball[1];
            if np.fabs(mid - self.w * 0.5) <= self.w_goal * 0.5:
                for i in range(self.n):#d对每个agent结算reward
                    if self.color[i] == 1:
                        self.reward[i] += self.reward_goal
                    else:
                        self.reward[i] -= self.reward_goal
                self.done = True
                return

        # 计算是否进下面的门
        if self.pos_ball[1] > 0 and new_pos_ball[1] <= 0:
            print('goal_down!!!')
            mid = self.pos_ball[0] * (self.pos_ball[1]) + new_pos_ball[0] * (- new_pos_ball[1])
            mid /= self.pos_ball[1] - new_pos_ball[1];
            if np.fabs(mid - self.w * 0.5) <= self.w_goal * 0.5:
                for i in range(self.n):
                    if self.color[i] == 0:
                        self.reward[i] += self.reward_goal
                    else:
                        self.reward[i] -= self.reward_goal
                self.done = True
                return

        self.pos_ball = new_pos_ball
        add(self.v_ball, -self.alpha, self.v_ball)

        # 计算出界问题
        for i in range(self.n):
            if self.outside(self.pos_agent[i]):
                self.limit(self.pos_agent[i])
                if self.c[i]:
                    self.v_ball = self.v_agent[i]
                    self.pos_ball = self.pos_agent[i]
                self.reward[i] += self.panalty_agent_go_outside

        # 球归于最近的人对手，给踢出去的人惩罚
        if self.outside(self.pos_ball):
            print('outside ball', self.pos_ball)
            self.reward[self.last_id] += self.panalty_ball_go_outside
            mi = 1e100
            for i in range(self.n):
                if self.color[i] != self.color[self.last_id] and self.get_dis_with_ball(i) > mi:
                    mi = self.get_dis_with_ball(i)
                    self.last_id = i
            self.id_ball_belongs_to = self.last_id
            self.v_ball = self.v_agent[self.last_id]
            self.pos_ball = self.pos_agent[self.last_id]

    def get_action(self, id,
                   action):  # action的格式为[v = [R,R], shoot_or_not = Bool, v_ball = [R,R]] 如果shoot_or_not = False, v_ball不存在或无效

        length_v_agent = np.sqrt(action[0][0] ** 2 + action[0][1] ** 2)
        action[0][0] /= max(self.Mv, length_v_agent) / self.Mv
        action[0][1] /= max(self.Mv, length_v_agent) / self.Mv

        length_v_ball = np.sqrt(action[3][0] ** 2 + action[3][1] ** 2)
        action[3][0] /= max(self.Mvb, length_v_agent) / self.Mvb
        action[3][1] /= max(self.Mvb, length_v_agent) / self.Mvb

        self.action[id] = action
        if self.vis_agent[id] == 0:
            self.n_vis += 1
            self.vis_agent[id] = 1
            # print(self.n_vis)

        if self.n_vis == self.n:  # 目前的设定是所有人都决策之后env演算一步
            self.step()
            self.vis_agent = [0 for i in range(self.n)]
            self.n_vis = 0

    def return_observation(self, id):  # 另外其他agent传输的信息要在observation之前还是之后？个人认为应该之后？因为当次observation要耗时

        self_state = {'pos_agent': self.pos_agent[id],
                      'v_agent': self.v_agent[id],
                      'control_ball': self.c[id], }
        ball_state = {'pos_ball': self.pos_ball,
                      'v_ball': self.v_ball}

        other_state = []
        for i in range(self.n):
            if i != id:
                i_state = {'pos_agent': self.pos_agent[i],
                           'v_agent': self.v_agent[i],
                           'control_ball': self.c[i], }
                other_state.append([i, i_state])

        state = {'self_state': self_state,
                 'ball_state': ball_state,
                 'other_state': other_state,
                 'id_ball_belongs_to': self.id_ball_belongs_to,
                 'message': self.message_pool[id]
                 }

        # self_state = [self.pos_agent[id], self.v_agent[id], self.c[id], self.pos_ball, self.v_ball]
        self.message_pool[id] = []
        reward = self.reward[id]
        self.reward[id] = 0
        # 一个人所接收到的state是自身位置，速度，是否可控球，球的位置，速度，以及其他agent传输的信息
        return [state, reward, self.done, None]


class Agent_naive:

    def __init__(self,color):
        self.pos_ball, self.pos_agent, self.current_control, self.color,self.id= 0, 0, 0, color,0
        self.signal,self.signalinfo,self.passinfo="slack",None,None
    def get_observation(self, obs):
        self.pos_ball = np.array(obs[0]['ball_state']['pos_ball'])
        self.pos_agent = np.array(obs[0]['self_state']['pos_agent'])
        self.current_control = obs[0]['id_ball_belongs_to']
        self.v_ball=obs[0]['ball_state']['v_ball']
        pass

    def return_action(self):
        # print(self.current_control)
        # de=[0,1]
        if self.current_control == -1: de = (self.pos_ball - self.pos_agent)
        return [(self.pos_ball - self.pos_agent) / 10 + rand() - 0.5,  # de,
                True,  # self.current_control==-1,
                rand() < 0.1,  # False,
                [(rand() - 0.5) * 10, (rand() - 0.5) * 10]
                # [0,1]
                ]

    def send_message(self, id_to):
        return {"pos_agent": self.pos_agent,"id":self.id,"color": self.color,"signal":self.signal,"signalinfo":self.signalinfo,"passinfo":self.passinfo}

def Dist(a,b):
    return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)
class Agent_Baseline:

    def __init__(self,id,color):
        self.pos_ball, self.pos_agent, self.current_control, self.id, self.color, self.signal= 0, 0, 0, id, color,"slack"
        self.signalinfo,self.passinfo=None,-1
        self.total_player,self.l,self.w=0,105,68#记录场地长宽
        if self.color==0:self.goal=[self.w/2,0]
        else:self.goal=[self.w/2,self.l]#球门位置
    def get_observation(self, obs):
        self.pos_ball = np.array(obs[0]['ball_state']['pos_ball'])
        self.pos_agent = np.array(obs[0]['self_state']['pos_agent'])
        self.current_control = obs[0]['id_ball_belongs_to']
        self.v_ball=obs[0]['ball_state']["v_ball"]
        self.info = obs[0]['message']#现在的info里存了其他所有人目前的位置
        #print("info=",end="")
        #print(self.info)
        pass

    def memberNum(self):#得到当前我方总人数
        if self.total_player>1:return
        cnt=1
        for x in self.info:
            if x[1]["color"]!=self.color:continue
            cnt+=1
        self.total_player=cnt
        return

    def I_am_K_nearest(self):#判断自己是不是距离球最近的己方队员
        dist = Modify(self.pos_agent - self.pos_ball,self.v_ball)
        myDist = dist[0] ** 2 + dist[1] ** 2
        # print("RTA:",end="")
        nearest = 1
        for x in self.info:
            #print(x)
            if x[1]["color"] != self.color: continue
            DD = Modify(x[1]["pos_agent"] - self.pos_ball,self.v_ball)
            dist = DD[0] ** 2 + DD[1] ** 2
            #print("dist=",end=" ")
            #print(dist)
            #print(" myDist",end=" ")
            #print(myDist)
            if dist < myDist:
                nearest += 1
        return nearest

    def countDefender(self):
        cnt=0
        for x in self.info:
            if x[1]["signal"]=="Defending":
                cnt+=1
        return cnt

    def countGrabber(self):
        cnt = 0
        for x in self.info:
            if x[1]["signal"] == "Grabbing":
                cnt += 1
        return cnt
    def countSentinel(self):
        cnt=0
        for x in self.info:
            if x[1]["signal"]=="Sentinel":
                cnt+=1
        return cnt
    def nearEnemy(self,pos=None):
        cnt=0
        if pos is None:pos=self.pos_agent
        for x in self.info:
            if x[1]["color"]!=self.color: continue
            if Dist(x[1]["pos_agent"],self.pos_agent)<3.5:#如果离得太近
                cnt+=1
        return cnt
    #def defend(self):#卡位，卡住对方的射门路线。跑到的随机位置。

    def passBall(self,target,passid=None):
        print("Passball:",end=" ")
        print(self.signal,self.passinfo)
        self.signal="slack"#恢复空闲状态
        #self.signalinfo=target
        self.passinfo=passid
        print("passid=",end=" ")
        print(passid)
        tar=Modify(target-self.pos_ball,self.v_ball)
        tar[0],tar[1]=tar[0]/5,tar[1]/5
        return [rand(),rand()],False,True,tar

    def pushingForward(self,pos=None):#目前只选择一个随机位置
        self.signal="running"
        for x in self.info:
            if x[1]["passinfo"] == self.id:
                self.signal="Grabbing"
                print("Push and Grab current glocol:",end=" ")
                print(gloCol(self.current_control),end=" ")
                print(self.pos_ball)
                return [Modify(self.pos_ball - self.pos_agent,self.v_ball), gloCol(self.current_control)!=self.color, False, [0, 0]]
                #去接球
            if self.color==0:#向下 0号队守上攻下 1号队守下攻上
                pos=[self.w/2+(rand()*rand()-0.5)*self.w/2,self.pos_ball[1]*rand()]
                """a=[]
                for i in range(10):
                    a.append(self.pos_agent[1]*rand(),self.w*rand())"""
            else:
                pos=[self.w/2+(rand()*rand()-0.5)*self.w/2,(self.l-self.pos_ball[1])*rand()+self.pos_ball[1]]
                """a=[]
                for i in range(10):
                    a.append((self.l-self.pos_agent[1])*rand()+self.pos_agent[1],self.w*rand())"""
        #self.sign
            self.signalinfo=pos
        x=pos-self.pos_agent
        #print(self.id)
        #print(pos)
        #print(self.pos_agent)
        #print(x)
        #print("************")
        if x[0]**2+x[1]**2<1.0:
            self.signal="slack"#跑到地方了
            #self.signalinfo=None
        return x,True,False,[0,0]


    def carryBallHarsh(self):#应该有探测路线上敌人的功能
        self.signal="carrying"
        if self.nearEnemy()<=1:self.signalinfo=self.goal
        else:self.signalinfo=[self.goal[0]+rand()*3.5-3.5,self.goal[1]]
        print("carryHarsh",end="")
        print(self.signalinfo,end=" ")
        print(self.pos_agent,end=" ")
        print(self.signalinfo-self.pos_agent)
        if self.color==0:#向下进攻
            return [self.signalinfo-self.pos_agent,gloCol(self.current_control)!=self.color,False,self.signalinfo-self.pos_agent]
        else:
            return [self.signalinfo-self.pos_agent,gloCol(self.current_control)!=self.color,False,self.signalinfo-self.pos_agent]

    def carryBall(self):#目前还没实现
        return self.carryBallHarsh()
    def bigFoot(self):
        self.signal="slack"#恢复空闲状态
        #self.signalinfo=None
        if self.color==0:#向下
            tar=[-10,rand()*10-5]
            return [tar,False,True,[rand()-0.5,rand()-0.5]]
        else:#向上
            tar=[10,rand()*10-5]
            return [tar,False,True,[rand()-0.5,rand()-0.5]]
    def Sentinel(self):#盯人，按照球门从近到远的顺序盯
        K=self.countSentinel()
        a=[]
        for x in self.info:
            if x[1]["color"]==self.color: continue
            a.append([x[1]["pos_agent"],Dist(x[1]["pos_agent"],self.goal)])#如果离得太近
        a.sort(key=lambda p:p[1])
        if K>=len(a):return self.pushingForward()
        if a[K][1]<0.75:#如果走到了对方附近或者不需要盯防
            self.signal="slack"
            #self.signalinfo=None
        return a[K][0]-self.pos_agent,True,False,[0,0]
        #找到对方离球门第x+1近的球员，对其进行盯防

    def Shoot(self):
        print("Shoot!",end=" ")
        goal=[self.goal[0]+rand()*3-1.5-self.pos_agent[0],self.goal[1]-self.pos_agent[1]]#射门角度随机
        self.signal="slack"
        #self.signalinfo=None
        print(goal)
        return [[rand(),rand()],False,True,goal]

    def findAttacker(self):
        lst=[]
        for x in self.info:
            if x[1]["signal"] == "running" and x[1]["color"]==self.color:
                lst.append([x[1]["pos_agent"],x[1]["id"]])
        if len(lst)==0:return []
        else:
            y=int(rand()*len(lst))%len(lst)
        print("attacker=",end="")
        print(lst[y])
        return lst[y]
    def findDefender(self):
        rec=[]
        recdist=0
        for x in self.info:
            if x[1]["color"]==self.color and self.nearEnemy(x[1]["pos_agent"])<=1:
                if recdist<Dist(x[1]["pos_agent"],self.goal):
                    recdist,rec=Dist(x[1]["pos_agent"]),[x[1]["pos_agent"],x[1]["id"]]
        return rec
    def Defend(self):#暂时写成在球门内的小范围内随机游走
        if abs(self.pos_agent[0]-self.goal[0])<3.61 or abs(self.pos_agent[1]-self.goal[1])<7.62:
            return self.goal-self.pos_agent,gloCol(self.current_control)!=self.color,True,self.goal-self.pos_agent
        return [rand(),rand()],True,True,[rand(),rand()]
    def return_action(self):
        # print(self.current_control)
        # #继续做上一帧要做的事情
        #print(self.info)
        print(self.id,end=" ")
        print(self.pos_agent,end=" ")#pos_agent和posagent[id]不一致？
        print(self.signal)
        print("now control:",end=" ")
        print(self.current_control,end=" ")
        print(gloCol(self.current_control))
        self.passinfo=-1#清零
        #print(self.info)
        self.memberNum()
        if self.signal=="Grabbing":#如果我方球员已经控球，或者已经有两名球员在抢球，则重新分配别的任务
            if gloCol(self.current_control)==self.color or self.countGrabber()>1:self.signal="slack"
            else:
                print("Continue Grabbing current glocol:", end=" ")
                print(gloCol(self.current_control),end=" ")
                print(self.pos_ball)
                return [Modify(self.pos_ball - self.pos_agent,self.v_ball), True, False, [0, 0]];
        #现在的问题是：之前传球的球员一直追着球跑又抓不到球
        if self.signal=="running":#如果发现球往自己方向来了，则去接球
            if  gloCol(self.current_control)!=self.color:
                self.signal="slack"
                print("Going back")
            else:return self.pushingForward(self.signalinfo)
        if self.signal=="defending" and self.current_control>=0 and gloCol(self.current_control)==self.color:
            self.signal="slack"
        if self.signal=="slack":self.signalinfo=None#信息清空
        if self.current_control == -1:#无人控球
            nearest = self.I_am_K_nearest()#如果是我方最近队员，则去抢球，广播自己在抢球的状态, False, [0, 0]];
            print("acquiring info:",end=" ")
            print(self.id,nearest)
            if nearest == 1:
                self.signal="Grabbing"
                print("Start to Grab current glocol:", end=" ")
                print(gloCol(self.current_control),end=" ")
                print(self.pos_ball)
                return [Modify(self.pos_ball - self.pos_agent,self.v_ball),True,False,[0,0]]
            else:
                rnd=rand()
                if rnd<0.33:return self.Sentinel()
                elif rnd<0.66:return self.pushingForward(self.signalinfo)
                else:return self.Defend()
            #随机盯人、跑位创造机会或者卡位，给队友广播相应的信号
        elif gloCol(self.current_control)!=self.color:#敌方控球
            #如果自己是距离球最近的球员，则尝试抢断，同时广播自己在抢球的状态
            nearest = self.I_am_K_nearest()
            if nearest == 1 and self.countGrabber()<3:
                self.signal="Grabbing"
                return [Modify(self.pos_ball - self.pos_agent,self.v_ball), True, False, [0, 0]];
            #如果对方控球对球门构成直接威胁，且目前没有正在卡位的人，则卡位（即离球门很近，射门路线上没有己方球员）同时广播自己在卡位的状态
            print("Enemy",end=" ")
            print(self.countDefender(),end=" ")
            print(self.total_player)
            if self.countDefender() < self.total_player / 3:
                return self.Defend()
            #如果目前已经有足够的卡位人员，则盯人（按球门距离排序，没有被盯防的对方球员依次盯防），向敌方球员的位置上跑，同时广播自己在盯人，以及在盯谁
            else:return self.Sentinel()
        elif gloCol(self.current_control)==self.color:#我方控球
            if self.current_control==self.id:#自己控球
                self.signal="carrying"
                #首先尝试射门（足够近且射门路线上没有敌方球员）
                print("Shooter: ",end="")
                print(self.goal,end=" ")
                print(self.pos_agent)
                if Dist(self.goal,self.pos_agent)<20:#足球场球门区纵向半径长20.15m
                    return self.Shoot()
                if self.nearEnemy()<1:
                    return self.carryBall()
                #然后尝试进攻性传球（如果有人距对方球门比自己更近，或者目标位置更近，且传球路线上没有敌方球员）
                tmp=self.findAttacker()
                print(tmp)
                if len(tmp)>0 and len(tmp[0])==2:
                    print("pass to attacker")
                    return self.passBall(tmp[0],tmp[1])
                #然后尝试盘带（如果抢球范围内只有不超过1名敌方队员，则按照加权方式及概率决定带球方向）
                if self.nearEnemy()<=1:
                    return self.carryBall()
                #再尝试回传（如果抢球范围内有超过1名敌方队员，则传向离我方球门最远的且路线上没有敌方球员的球员）
                tmp=self.findDefender()
                if len(tmp) > 0 and len(tmp[0]) == 2:
                    print("pass to defender")
                    return self.passBall(tmp[0],tmp[1])
                #再尝试硬着头皮盘带（如果不存在这样的球员）和大脚解围（向前方90度范围内随机方向踢出，在距离己方球门非常近的情况下使用）。
                #这两个应当满足一个随机函数关系，越靠近对方半场越倾向于盘带，越靠近己方半场越倾向于解围。且显然应该在非常靠近己方半场的位置，解围的概率非常大。所以是凸函数。
                #综上采用开方乘10。这里暂时写的是线性的。
                rnd=rand()*105
                if (self.color==0 and self.pos_ball[1]<rnd) or (self.color==1 and self.pos_ball[0]>rnd):
                    return self.carryBallHarsh()
                else:return self.bigFoot()
                pass
            else:#队友控球
                #首先考虑留守卡位（如果留守人数严格小于全队人数的1/3，则留守；如果这个半场还有敌方队员，则盯人）考虑是盯人还是创造进攻机会还是原地留守
                if self.countDefender()<self.total_player/3:
                    return self.Defend()
                #随机选择盯人还是创造机会。但是注意到自己的状态一旦定下来，就不能轻易改变，不然就会变成在原地发抖。
                #应该有一个机制，只有在上一次状态维持至少10回合且期间控球者发生了变化才改变当前状态（还未实现）
                #或者，如果当前状态属于slack，也需要维持状态
                rnd=rand()
                if rnd<0.5:#rnd函数还要改
                    return self.Sentinel()
                #然后考虑创造进攻机会（在前场随机10个点，向前方离对方球员最近点最远的位置跑），广播自己的目标位置
                if rnd>0.5:
                    return self.pushingForward()
        #print("DEFAULT")
        return [1,0],True,True,[1,0]#[(rand()-0.5)*3,(rand()-0.5)*3], True, True, [(rand()-0.5)*10,(rand()-0.5)*10]#default

    def send_message(self, id_to):
        return {"pos_agent": self.pos_agent,"id":self.id,"color": self.color,"signal":self.signal,"signalinfo":self.signalinfo,"passinfo":self.passinfo}
#target="sentinel","running","blocking"
#环境本身并未规定0号和1号哪个上哪个下，这里人为规定一下0号队守上攻下，1号队守下攻上
globalColor=[]
def gloCol(x):
    if x==-1:return -1
    return globalColor[x]
def Modify(x,y):
    return [x[0]+y[0]*0.9,x[1]+y[1]*0.9]
def test():

    n = 22
    # agents = [Agent_naive() for i in range(n)]
    agents = [Agent_Baseline(0,1),Agent_Baseline(1,1),Agent_Baseline(2,1),Agent_Baseline(3,1),Agent_Baseline(4,1),Agent_Baseline(5,1),
              Agent_Baseline(6, 0), Agent_Baseline(7, 0), Agent_Baseline(8, 0), Agent_Baseline(9, 0),
              Agent_Baseline(10, 0), Agent_Baseline(11, 0),
              Agent_Baseline(12, 1), Agent_Baseline(13, 1), Agent_Baseline(14, 1), Agent_Baseline(15, 1),
              Agent_Baseline(16, 1),
              Agent_Baseline(17, 0), Agent_Baseline(18, 0), Agent_Baseline(19, 0), Agent_Baseline(20, 0),
              Agent_Baseline(21, 0),
              ]
    # 最愚蠢的agent，向足球靠近，然后尽可能抢断，之后有10%的概率朝随便方向踢出去
    # 只是用来表示agent需要做什么

    env = simple_soccer_env([1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0])  # 定义环境的时候传入一个n元数组表示每一个agent的阵营
    env.reset()

    t0s, t1s, bs = [], [], []
    t0, t1, b, vb = env.return_env_state()  # 这些变量用来记录env信息做可视化，当一局完全结束后，比赛才可变为视频

    t0s.append(deepcopy(t0))
    t1s.append(deepcopy(t1))
    bs.append(deepcopy(b))
    #   print(t0,t1,b,vb)
    while env.done == False and env.step_ <= 300:  # 默认最多演算300步，若射门(done = True)则提前退出
        print("Step=",end="")
        print(env.step_,end=" pos_ball=")
        print(env.pos_ball)
        #       print('       ',env.id_ball_belongs_to)
        for i in range(n):
            agents[i].get_observation(env.return_observation(i))  # 每一个agent先获得observation,reward,done,info
        for i in range(n):#message写在了前面，这样开场就不需要特判初始化数据了
            for j in range(n):
                if i != j:
                    env.get_message(i, j, agents[i].send_message(j))  # i在这里决定给j什么信息
        for i in range(n):
            env.get_action(i, agents[i].return_action())  # 每一个agent一起把action传给env

        t0, t1, b, vb = env.return_env_state()
        # print(t0,t1,b,vb)
        t0s.append(deepcopy(t0))
        t1s.append(deepcopy(t1))
        bs.append(deepcopy(b))
    #get action结束后算一步
    Animation([t0s, t1s], bs, env.w_goal, env.w, env.l)


test()

