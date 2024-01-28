import random

import gymnasium
import numpy as np
from gymnasium import spaces

# 配置日志记录器
"""
该环境中，共有8组设备，每组有10个设备，其中前3个设备产生时延敏感数据，后7个设备产生时延容忍数据
"""


class QFLEnv(gymnasium.Env):
    def get_obj2_min(self, cof):
        d_max = [100] * (self.done_step + 1)
        rate_max = [0] * self.done_step
        obj2_min_temp = [0] * self.done_step
        for i in range(self.done_step):
            rate_max[i] = cof * self.totalbandwidth * np.log2(
                1 + 1e7 * self.pwr[0] * np.power(np.power(d_max[i] + self.t_comp_mean * self.vel[0] - self.radius, 2) + 100,
                                                 -1))  # action2 *10 * log2(1+1e7*action3/(rsu_dis^2))
            if cof != 0.1:
                obj2_min_temp[i] = self.qe_bit * self.modelsize / (32 * rate_max[i]) + self.t_comp_mean
            else:
                obj2_min_temp[i] = self.latency_limit[0]
            d_max[i + 1] = d_max[i] + obj2_min_temp[i] * self.vel[0]
        return obj2_min_temp

    def __init__(self, cars=10, done_step=10, weight_lambda=1, debug=False, obj_file=None,
                 lmax=0.35, baseline = 'ppo'):  # done_step需要设置成2的次方，这样才能保证total_timesteps是他的整数倍
        super(QFLEnv, self).__init__()
        self.current_step = 0
        self.cars = cars
        self.done_step = done_step  # 联邦学习全局轮数
        self.weight_lambda = weight_lambda
        self.obj_file = obj_file
        self.baseline = baseline
        self.qe_bit = 9
        # 通信+计算总的时延 t_comp+latency(b=(0.05,0.1))，越小会让obj1也越小,0.27
        self.modelsize = 8
        self.totalbandwidth = 20
        self.radius = 500
        self.zeta = 1  # 控制人数在每一轮的权重
        self.t_comp_min = 0.225
        self.t_comp_max = 0.275
        self.t_comp_mean = (self.t_comp_max + self.t_comp_min) / 2
        self.t_comp = [random.uniform(self.t_comp_min, self.t_comp_max) for _ in range(self.cars)]
        # self.t_comp = [self.t_comp_mean] * self.cars
        self.latency_limit = [lmax + self.t_comp_mean]
        if self.obj_file:
            self.f = open(self.obj_file, 'w')
        # static 环境
        self.pwr = [1] * self.cars
        self.vel = [30] * self.cars
        self.qe_para = 1
        self.qe = [self.cars * self.done_step * self.qe_para / np.power(np.power(2, self.qe_bit) - 1, 2)]
        self.obj1 = []
        self.obj2 = []
        self.cons = []
        self.slct_number = []
        self.debug = debug
        self.obj2_min = self.get_obj2_min(1)
        self.obj2_max = self.get_obj2_min(0.1)
        # dynamic 环境
        self.initial_travel_dis = [random.random() * 100 for i in range(self.cars)]  # 每个人起始位置不一样, [0] * self.cars  # 每个人都在位置为0的地方开始
        self.travel_dis = [random.random() * 100 for i in range(self.cars)]
        self.rsu_dis = [0] * self.cars
        self.latency_itr = [0]
        # 动作1-slc：0或1
        # 动作2-bd：0.01到1之间的实数
        # 动作3-quant：1到32之间的整数

        action_low = np.concatenate([np.zeros(self.cars), np.ones(self.cars), np.zeros(self.cars)])
        action_high = np.concatenate([np.ones(self.cars), np.ones(self.cars), np.ones(self.cars)])
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)

        static_low = np.concatenate([np.zeros(1),
                                     np.zeros(1),
                                     np.zeros(1)])
        static_high = np.concatenate([np.full(1, np.inf),
                                      np.full(1, np.inf),
                                      np.full(1, np.inf)])

        # 动态变量的范围
        dynamic_low = np.concatenate([np.zeros(self.cars),
                                      np.zeros(self.cars),
                                      np.zeros(1)])
        dynamic_high = np.concatenate([np.full(self.cars, np.inf),
                                       np.full(self.cars, np.inf),
                                       np.full(1, np.inf)])

        # 定义观察空间
        self.observation_space = spaces.Box(low=np.concatenate([static_low, dynamic_low, dynamic_low]),
                                            high=np.concatenate([static_high, dynamic_high, dynamic_high]),
                                            dtype=np.float32)
        print('initial completed!')

    # 训练1个episode之后reset
    def reset(self, seed=0):
        # 会append的
        self.obj1 = []
        self.obj2 = []
        self.cons = []
        self.slct_number = []
        # 有+=的
        self.current_step = 0
        self.travel_dis = [random.random() * 100 for i in range(self.cars)]
        self.rsu_dis =[0] * self.cars
        self.latency_itr = [0]
        self.last_dynamic_obs = [i / self.radius for i in self.travel_dis] + [i / self.radius for i in self.rsu_dis] + self.latency_itr
        initial_observation = self._get_obs()
        return np.array(initial_observation), {}

    def step(self, action):  #
        # print('step {}/{}'.format(self.current_step + 1, self.done_step))

        # 动作离散化/加规则
        if self.baseline != 'rc':
            slct_before = action[:self.cars]
        else:
            slct_before = bd_before = np.random.rand(self.cars)
        slct = np.where(slct_before > 0.5, 1, 0)

        if self.baseline != 'rb':
            bd_before = action[self.cars:self.cars * 2]
        else:
            bd_before = np.random.rand(self.cars)
            # bd = np.array([1 / self.cars] * self.cars)  # UB method
        bd = slct * bd_before / np.sum(slct * bd_before) if np.sum(slct) > 0 else bd_before / np.sum(bd_before)

        if self.baseline != 'rq':
            quant_before = action[self.cars*2:self.cars *3]
        else:
            quant_before = np.random.rand(self.cars)
            # quant = np.array([self.qe_bit] * self.cars)
        quant = np.where(quant_before == 1, 10, np.floor(7 + (11 - 7) * quant_before).astype(int))  # 可选空间在[7,10]
        # slct[(slct == 1) & (1 / (2 ** quant - 1) ** 2 > self.qe)] = 0  # 约束的规则
        quant[slct == 0] = 1
        if self.debug:
            print("slct:{}".format(list(slct)))

        # 更新state
        travel_dis_after_comp = [0] * self.cars  # 加上计算时延后车辆移动距离
        rate = [0] * self.cars  # 香浓公式速率
        latency_communication = [0] * self.cars  # 通信时延
        for i in range(self.cars):
            travel_dis_after_comp[i] = self.travel_dis[i] + self.t_comp[i] * self.vel[i]
            if slct[i] * bd[i] > 0:
                self.rsu_dis[i] = np.sqrt(np.power(travel_dis_after_comp[i] - self.radius, 2) + 100)
                rate[i] = bd[i] * self.totalbandwidth * np.log2(1 + 1e7 * self.pwr[i] * np.power(self.rsu_dis[i], -2))  # action2 *10 * log2(1+1e7*action3/(rsu_dis^2))
                latency_communication[i] = quant[i] * self.modelsize / 32 / rate[i]
                if ((latency_communication[i] + self.t_comp[i] >= self.latency_limit[0]) or
                        (travel_dis_after_comp[i] + latency_communication[i] * self.vel[i] >= 2 * self.radius)):
                    slct[i] = 0
                    bd[i] = 0
                    quant[i] = 1

        self.latency_itr[0] = np.max([slct[i] * (self.t_comp[i] + latency_communication[i]) for i in range(self.cars)]) if np.sum(slct) > 0  else self.latency_limit[0]
        for i in range(self.cars):
            self.travel_dis[i] += self.latency_itr[0] * self.vel[i]  # 加上通信时延后车辆移动距离

        if self.debug:
            print('self.travel_dis:{}'.format(self.travel_dis))
            print("self.rsu_dis:{}".format(self.rsu_dis))
            print("rate{}".format(rate))
            print('latency_itr{}'.format( self.latency_itr[0]))
            print("bandwidth:{}".format(list(bd)))
            print('quant{}'.format(list(quant)))
            print("slct:{}".format(list(slct)))


        # 计算两个目标
        obj1_per_round = np.log(1+np.sum(slct)) * self.zeta ** (self.current_step)
        obj2_per_round = self.latency_itr[0]
        cons_per_round = np.sum(slct / np.power(np.power(2, quant) - 1, 2))
        slct_number = np.sum(slct)
        self.obj1.append(obj1_per_round)
        self.obj2.append(obj2_per_round)
        self.cons.append(cons_per_round)
        self.slct_number.append(slct_number)
        # obs = self.obj2_min + \
        #       self.latency_limit + [self.qe_bit] + [self.t_comp_mean] + \
        #       [i / self.radius for i in self.travel_dis] + [i / self.radius for i in self.rsu_dis] + \
        #       self.latency_itr + self.last_dynamic_obs + slct.tolist() + bd.tolist() + quant.tolist()
        obs = self._get_obs()
        self.last_dynamic_obs = [i / self.radius for i in self.travel_dis] + [i / self.radius for i in self.rsu_dis] + self.latency_itr

        # normalize两个目标
        obj1_min = np.sum([np.log(1+1) * self.zeta ** i for i in range(self.current_step+1)])
        obj1_max = np.sum([np.log(10+1) * self.zeta ** i for i in range(self.current_step+1)])
        obj2_min = np.sum([self.obj2_min[i] for i in range(self.current_step + 1)])
        obj2_max = self.latency_limit[0] * (self.current_step+1)
        # obj2_max = np.sum([self.obj2_max[i] for i in range(self.current_step + 1)])

        # check_termination
        self.current_step += 1
        done_flag1 = False
        if self.current_step >= self.done_step:
            done_flag1 = True

        reward, done_flag2 = self.get_reward(slct, bd, quant, obj1_min, obj1_max, obj2_min, obj2_max)
        done = done_flag1 or done_flag2
        return obs, reward, done, False, {}

    def get_reward(self, slct, bd, quant, obj1_min, obj1_max, obj2_min, obj2_max):

        done = False
        reward = 0
        # 调参3个权重
        a = 1

        # if qe_calculate > self.qe[0]:
        #     done = True

        if self.current_step <= self.done_step:
            accumulated_obj1 = np.sum(np.array(self.obj1))
            accumulated_obj2 = np.sum(np.array(self.obj2))
            obj1 = (accumulated_obj1 - obj1_min) / (obj1_max - obj1_min) # normlize
            obj2 = (accumulated_obj2 - obj2_min) / (obj2_max - obj2_min)  # normlize
            reward = self.weight_lambda * obj1
            reward -= (1 - self.weight_lambda) * obj2

            # QE不等式约束
            qe_calculate = np.sum(self.cons)
            qe_current = self.qe[0] * self.current_step / self.done_step
            if qe_calculate > qe_current:
                # reward -= (qe_calculate / qe_current - 1) * a
                reward -= 1
            # else:
            #     reward += (qe_calculate / qe_current) * a * 0.1

        if self.current_step == self.done_step:
            print('slct:{}, travel_dis:{}'.format(list(slct), self.travel_dis))

            # 在每个回合结束时执行，记录奖励值
            if self.obj_file:
                self.f.write(str(accumulated_obj1) + "\t" + str(accumulated_obj2) + "\t" + str(reward) + "\t" +
                             str(obj1) + "\t" + str(obj2) +
                             "\t" + str(slct.tolist()) + "\t" + str(bd.tolist()) + "\t" + str(quant.tolist()) + "\n")

        # print("mix {:.2f}, accumulated_obj2 {:.2f}, max {:.2f},".format(obj2_min, accumulated_obj2, obj2_max))
        return reward, done

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def _get_obs(self):
        """
        观测空间
        :return:
        """
        observation = self.latency_limit + [self.qe_bit] + [self.t_comp_mean] + \
              [i / self.radius for i in self.travel_dis] + [i / self.radius for i in self.rsu_dis]+ self.latency_itr + self.last_dynamic_obs
        return observation
