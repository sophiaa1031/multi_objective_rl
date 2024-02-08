import random

import gymnasium
import numpy as np
from gymnasium import spaces


class QFLEnv(gymnasium.Env):
    def get_obj2_min(self, cof):
        d_max = [100] * (self.done_step + 1)
        rate_max = [0] * self.done_step
        obj2_min_temp = [0] * self.done_step
        for i in range(self.done_step):
            rate_max[i] = cof * self.totalbandwidth * np.log2(
                1 + 1e7 * self.pwr[0] * np.power(np.power(d_max[i] + self.t_comp_mean * self.vel[0] - self.radius, 2) + self.rsu_distance_min**2,
                                                 -1))  # action2 *10 * log2(1+1e7*action3/(rsu_dis^2))
            if cof != 0.1:
                obj2_min_temp[i] = self.qe_bit * self.modelsize / (32 * rate_max[i]) + self.t_comp_mean
            else:
                obj2_min_temp[i] = self.latency_limit
            d_max[i + 1] = d_max[i] + obj2_min_temp[i] * self.vel[0]
        return obj2_min_temp

    def __init__(self, args, weight_lambda=0.5, debug=False, obj_file=None,
                 lmax=0.35):  # done_step需要设置成2的次方，这样才能保证total_timesteps是他的整数倍
        super(QFLEnv, self).__init__()
        self.current_step = 0
        self.cars = args.num_users
        self.done_step = args.done_step  # 联邦学习全局轮数
        self.weight_lambda = weight_lambda
        self.obj_file = obj_file
        self.baseline = args.baseline
        self.qe_bit = 8
        self.qe_bit_per_car = 1 / np.power(np.power(2, self.qe_bit) - 1, 2)
        self.qe_per_car_ratio_max = np.power(np.power(2, 10) - 1, 2) / self.qe_bit_per_car
        self.qe_per_car_ratio_min = np.power(np.power(2, 7) - 1, 2) / self.qe_bit_per_car
        # 通信+计算总的时延 t_comp+latency(b=(0.05,0.1))，越小会让obj1也越小,0.27
        self.modelsize = 8
        self.totalbandwidth = 20
        self.radius = 500
        self.rsu_distance_min = 10
        self.rsu_distance_max = np.sqrt(self.radius**2 + self.rsu_distance_min**2)
        self.zeta = 1  # 控制人数在每一轮的权重
        self.t_comp_min = 0.225
        self.t_comp_max = 0.275
        self.t_comp_mean = (self.t_comp_max + self.t_comp_min) / 2
        self.t_comp = [random.uniform(self.t_comp_min, self.t_comp_max) for _ in range(self.cars)]
        # self.t_comp = [self.t_comp_mean] * self.cars
        self.latency_limit = lmax + self.t_comp_mean
        if self.obj_file:
            self.f = open(self.obj_file, 'w')
        # static 环境
        self.pwr = [1] * self.cars
        self.vel = [30] * self.cars
        self.qe_para = 1
        self.qe = self.cars * self.done_step * self.qe_para / np.power(np.power(2, self.qe_bit) - 1, 2)
        self.qe_per_car = [self.qe_bit_per_car] * self.cars
        self.obj1 = []
        self.obj2 = []
        self.cons = []
        self.debug = debug
        self.obj2_min = self.get_obj2_min(1)
        self.obj2_max = self.get_obj2_min(0.1)
        # dynamic 环境
        self.travel_dis = [random.random() * 100 for i in range(self.cars)]  # 每个人起始位置不一样, [0] * self.cars  # 每个人都在位置为0的地方开始
        self.rsu_dis = [0] * self.cars
        self.latency_itr = 0
        self.slct = [1] * self.cars
        self.cons_ratio_per_round = 1
        self.cons_ratio_total = 1
        self.cons_flag = 0

        if self.baseline == 'rb' or self.baseline == 'rq' or self.baseline == 'uq':
            action_low = np.concatenate([-np.ones(self.cars)])
            action_high = np.concatenate([np.ones(self.cars)])
        else:
            action_low = np.concatenate([-np.ones(self.cars), -np.ones(self.cars)])
            action_high = np.concatenate([np.ones(self.cars), np.ones(self.cars)])

        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)

        # static_low = np.concatenate([np.zeros(1),
        #                              np.zeros(1),
        #                              np.zeros(1)])
        # static_high = np.concatenate([np.full(1, np.inf),
        #                               np.ones(1),
        #                               np.full(1, np.inf)])

        # 动态变量的范围
        dynamic_low = np.concatenate([-np.ones(self.cars),
                                      -np.ones(self.cars),
                                      -np.ones(self.cars),
                                      -np.ones(self.cars),
                                      -np.ones(1),-np.ones(1)])
        dynamic_high = np.concatenate([np.ones(self.cars),
                                       np.ones(self.cars),
                                       np.ones(self.cars),
                                       np.ones(self.cars),
                                       np.ones(1),np.ones(1)])

        # 定义观察空间
        self.observation_space = spaces.Box(low=np.concatenate([dynamic_low, dynamic_low]),
                                            high=np.concatenate([dynamic_high, dynamic_high]),
                                            dtype=np.float32)
        print('initial completed!')

    # 训练1个episode之后reset
    def reset(self, seed=0):
        # 会append的
        self.obj1 = []
        self.obj2 = []
        self.cons = []
        # 有+=的
        self.current_step = 0
        self.cons_flag = 0
        self.travel_dis = [random.random() * 100 for i in range(self.cars)]
        self.rsu_dis =[0] * self.cars
        self.latency_itr = 0
        self.slct = [1] * self.cars
        self.last_dynamic_obs = self.get_last_dynamic_obs()
        initial_observation = self._get_obs()
        return np.array(initial_observation), {}

    def step(self, action):  #
        # print('step {}/{}'.format(self.current_step + 1, self.done_step))

        # 动作离散化/加规则
        if self.baseline == 'rb':
            bd_before = np.random.rand(self.cars)
            quant_before = action[:self.cars]/2+1/2
        elif self.baseline == 'rq':
            bd_before = action[:self.cars]/2+1/2
            quant_before = np.random.rand(self.cars)
        elif self.baseline == 'uq':
            bd_before = action[:self.cars]/2+1/2
        elif self.baseline == 'ppo':
            bd_before = action[:self.cars]/2+1/2
            quant_before = action[self.cars:self.cars * 2]/2+1/2
        bd = bd_before / np.sum(bd_before) if np.sum(bd_before) > 0 else [0]*self.cars
        if self.baseline == 'uq':
            quant =  np.array([self.qe_bit] * self.cars)
        else:
            quant = np.where(quant_before == 1, 10, np.floor(7 + (11 - 7) * quant_before).astype(int))  # 可选空间在[7,10]
        bd = np.where((np.array(bd) > 0) & (1 / (2 ** np.array(quant) - 1) ** 2 > self.qe), 0, bd)  # 约束的规则
        quant[bd == 0] = 1
        if self.debug:
            print("slct:{}".format(list(bd)))

        # 更新state
        travel_dis_after_comp = [0] * self.cars  # 加上计算时延后车辆移动距离
        rate = [0] * self.cars  # 香浓公式速率
        latency_communication = [0] * self.cars  # 通信时延
        bd_update_flag = True
        while bd_update_flag:
            bd_update_flag = False
            for i in range(self.cars):
                travel_dis_after_comp[i] = self.travel_dis[i] + self.t_comp[i] * self.vel[i]
                if bd[i] > 0:
                    self.rsu_dis[i] = np.sqrt(np.power(travel_dis_after_comp[i] - self.radius, 2) + self.rsu_distance_min**2)
                    rate[i] = bd[i] * self.totalbandwidth * np.log2(1 + 1e7 * self.pwr[i] * np.power(self.rsu_dis[i], -2))  # action2 *10 * log2(1+1e7*action3/(rsu_dis^2))
                    latency_communication[i] = quant[i] * self.modelsize / 32 / rate[i]
                    if ((latency_communication[i] + self.t_comp[i] >= self.latency_limit) or
                            (travel_dis_after_comp[i] + latency_communication[i] * self.vel[i] >= 2 * self.radius)):
                        bd[i] = 0
                        quant[i] = 1
                        bd_update_flag = True
            if bd_update_flag:
                bd = bd / np.sum(bd)
        #  这里已经做好了用户选择了
        self.slct = [1 if num > 0 else 0 for num in bd]
        self.qe_per_car = 1/ np.power(np.power(2, quant) - 1, 2)
        self.latency_itr = np.max([self.slct[i] * (self.t_comp[i] + latency_communication[i]) for i in range(self.cars)]) if np.sum(self.slct) > 0  else self.latency_limit
        for i in range(self.cars):
            self.travel_dis[i] += self.latency_itr * self.vel[i]  # 加上通信时延后车辆移动距离

        if self.debug:
            print('self.travel_dis:{}'.format(self.travel_dis))
            print("self.rsu_dis:{}".format(self.rsu_dis))
            print("rate{}".format(rate))
            print('latency_itr{}'.format( self.latency_itr))
            print("bandwidth:{}".format(list(bd)))
            print('quant{}'.format(list(quant)))
            print("slct:{}".format(self.slct))


        # 计算两个目标和约束
        cons_per_round = np.sum(self.slct / np.power(np.power(2, quant) - 1, 2))
        self.cons_ratio_total = cons_per_round / self.qe  # 为了放入观察空间里
        self.cons_ratio_per_round = self.cons_ratio_total * self.done_step  # 为了放入观察空间里
        obj1_per_round = np.log(1+sum(self.slct)) * self.zeta ** (self.current_step)
        obj2_per_round = self.latency_itr
        self.obj1.append(obj1_per_round)
        self.obj2.append(obj2_per_round)
        self.cons.append(cons_per_round)
        obs = self._get_obs()
        self.last_dynamic_obs = self.get_last_dynamic_obs()

        # normalize两个目标
        obj1_min = np.sum([np.log(1+1) * self.zeta ** i for i in range(self.current_step+1)])
        obj1_max =  np.sum([np.log(10+1) * self.zeta ** i for i in range(self.current_step+1)])
        obj2_min = np.sum([self.obj2_min[i] for i in range(self.current_step + 1)])
        obj2_max = self.latency_limit * (self.current_step+1)
        # obj2_max = np.sum([self.obj2_max[i] for i in range(self.current_step + 1)])

        # check_termination
        self.current_step += 1
        done_flag1 = False
        if self.current_step >= self.done_step:
            done_flag1 = True

        reward, done_flag2 = self.get_reward(bd, quant, obj1_min, obj1_max, obj2_min, obj2_max)
        done = done_flag1 or done_flag2
        return obs, reward, done, False, {}

    def get_reward(self, bd, quant, obj1_min, obj1_max, obj2_min, obj2_max):

        done = False
        reward = 0
        # 调参3个权重，如果满足约束，主要优化目标
        if self.cons_flag >=5:
            a = 0.01
            b = 0.01
        else:
            a = 1
            b = 1


        # if self.current_step <= self.done_step:
        current = self.current_step-1
        accumulated_obj1 = self.obj1[-1] # np.sum(np.array(self.obj1))
        accumulated_obj2 = self.obj2[-1] # np.sum(np.array(self.obj2))
        obj1 = (accumulated_obj1 - np.log(1+1) * self.zeta ** current) / (np.log(10+1) * self.zeta ** current - np.log(1+1) * self.zeta ** current) # normlize
        obj2 = (accumulated_obj2 - self.obj2_min[current]) / ( self.latency_limit - self.obj2_min[current])  # normlize
        reward = self.weight_lambda * obj1
        reward -= (1 - self.weight_lambda) * obj2

        # QE不等式约束
        qe_calculate = self.cons[-1] # np.sum(self.cons)
        qe_current = self.qe / self.done_step  # self.qe[0] * self.current_step / self.done_step
        # if qe_calculate > self.qe[0]:
        #     done = True
        if qe_calculate > qe_current:
            reward -= (qe_calculate / qe_current - 1) * a # 10
            self.cons_flag = 0
        else:
            reward += (qe_calculate/qe_current-1) * b
            self.cons_flag += 1
        # if self.current_step %50==0:
        #     print('current:{},threshold:{},constraint ratio:{},reward:{},quant:{}'.format(qe_calculate,qe_current,qe_calculate/qe_current,reward,quant))

        if self.current_step == self.done_step:
            # accumulated_obj1 = np.sum(np.array(self.obj1))
            # accumulated_obj2 = np.sum(np.array(self.obj2))
            # obj1 = (accumulated_obj1 - obj1_min) / (obj1_max - obj1_min) # normlize
            # obj2 = (accumulated_obj2 - obj2_min) / (obj2_max - obj2_min)  # normlize
            # reward = self.weight_lambda * obj1
            # reward -= (1 - self.weight_lambda) * obj2
            #
            # qe_calculate = np.sum(self.cons)
            # qe_current = self.qe # * self.current_step / self.done_step
            # if qe_calculate > qe_current:
            #     reward -= 10
            # print('constraint ratio:{},reward:{}, quant:{}'.format(qe_calculate/qe_current,reward,quant))
            # 在每个回合结束时执行，记录奖励值
            if self.obj_file:
                self.f.write(str(accumulated_obj1) + "\t" + str(accumulated_obj2) + "\t" + str(reward) + "\t" +
                             str(obj1) + "\t" + str(obj2) +
                             "\t" + str(self.slct) + "\t" + str(bd.tolist()) + "\t" + str(quant.tolist()) + "\n")
        return reward, done

    def get_last_dynamic_obs(self):
        return ([i / (self.radius) -1 for i in self.travel_dis] + [2* (i -self.rsu_distance_min) / (self.rsu_distance_max-self.rsu_distance_min) -1 for i in self.rsu_dis]+
                [2*(i/self.qe_bit_per_car-self.qe_per_car_ratio_min)/(self.qe_per_car_ratio_max-self.qe_per_car_ratio_min)-1  for i in self.qe_per_car] +[2*i-1 for i in self.slct]+
                [2*self.latency_itr/self.latency_limit-1]+ [2*self.cons_ratio_per_round-1])
    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def _get_obs(self):
        """
        观测空间
        :return:
        [self.latency_limit]+ [(self.qe_bit-7)/(10-7)]+ [self.t_comp_mean]+
        """
        observation = ([i / (self.radius) -1 for i in self.travel_dis] + [2*(i -self.rsu_distance_min) / (self.rsu_distance_max-self.rsu_distance_min) -1 for i in self.rsu_dis]+
                       [2*(i/self.qe_bit_per_car-self.qe_per_car_ratio_min)/(self.qe_per_car_ratio_max-self.qe_per_car_ratio_min)-1 for i in self.qe_per_car]+[2*i-1 for i in self.slct]+
                       [2*self.latency_itr/self.latency_limit-1]+ [2*self.cons_ratio_per_round-1]+
                       self.last_dynamic_obs)
        return observation
