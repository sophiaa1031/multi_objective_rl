import math
import numpy as np
import IoT_Devices
import para


iotd = IoT_Devices.devices
np.random.seed(22)
# 分别生成十个 0-1的随机数
random1 = np.random.rand(10)
random2 = np.random.rand(7)
random3 = np.random.rand(10)


random_beta = np.zeros(10)
random_epsilon = np.zeros(10)
random_delta = np.zeros(10)
storage = 2048

for i in range(10):
    random_beta[i] = random1[i] / sum(random1)  # 带宽资源,比例值之和限制<=1
    random_delta[i] = random3[i] / sum(random3)  # 计算资源,比例值之和限制<=1
for i in range(7):
    random_epsilon[i + 3] = random2[i] / sum(random2)


c = np.zeros(10)  # 每个传感器的data rate
t = np.zeros(10)  # 每个传感器的传输时间
E_tr = np.zeros(10)  # 每个传感器的传输能耗
E_c = np.zeros(10)  # 每个传感器的计算能耗
q_dl = np.zeros(10)  # 该组设备每个设备时延容忍数据上传大小
one_device_reward = np.zeros(10)
social_welfare = []  # 8个组，每组设备utility之和
energy_efficiency = []
Q_ds = []
Q_dl = []
EE = []

for i in range(8):
    c = np.zeros(10)  # 每个传感器的data rate
    t = np.zeros(10)  # 每个传感器的传输时间
    E_tr = np.zeros(10)  # 每个传感器的传输能耗
    E_c = np.zeros(10)  # 每个传感器的计算能耗
    one_device_reward = np.zeros(10)
    q_dl = np.zeros(10)
    q_ds = 0
    ee = np.zeros(10)
    for _ in range(3):
        q_ds += iotd[i][_].data_size
    Q_ds.append(q_ds)
    if i == 0:
        storage = 2048 - Q_ds[0]
    else:
        storage = 2048 - Q_ds[i] - sum(Q_dl)
    if storage < 0:
        print("第",i,"组，""storage overflow")
        continue
    else:
        for j in range(10):
            if j < 3:
                # 数据传输速率 MB/s
                c[j] = random_beta[j] * 6e8 * math.log(1 + (para.IoTD_p * para.g / para.sigma)) / 8388608  # * 0.125e-6
                # 传输时间
                t[j] = iotd[i][j].data_size / c[j]
                # 传输能耗 W*s 单位为J 焦耳
                E_tr[j] = para.IoTD_p * t[j]
                # 计算能耗 数据单位为bit 所以乘以8388608
                E_c[j] = para.Omu * para.omega1 * iotd[i][j].data_size * 8388608 * ((random_delta[j] * 2.4e9) ** 2)
                ee[j] = c[j] / (E_tr[j] + E_c[j])
                di_er_xiang = para.alpha1 * math.log(1 + iotd[i][j].data_size)
                di_san_xiang = para.alpha2 * math.log(1 + ((random_delta[j] * 2.4e9) / (8388608 * para.omega1 * iotd[i][j].data_size)))
                # 一个设备的reward
                one_device_reward[j] = c[j] / (E_c[j] + E_tr[j]) + di_er_xiang + di_san_xiang
            else:
                # 数据传输速率 MB/s
                c[j] = random_beta[j] * 6e8 * math.log(1 + (para.IoTD_p * para.g / para.sigma)) * 1.25e-7
                # 单个设备时延容忍数据上传大小
                if random_epsilon[j] * storage < iotd[i][j].data_size:
                    q_dl[j] = random_epsilon[j] * storage
                else:
                    q_dl[j] = iotd[i][j].data_size
                # 上传时间
                t[j] = q_dl[j] / c[j]
                # 传输能耗
                E_tr[j] = para.IoTD_p * t[j]
                # 计算能耗
                E_c[j] = para.Omu * para.omega2 * q_dl[j] * 8388608 * ((random_delta[j] * 2.4e9) ** 2)
                ee[j] = c[j] / (E_tr[j] + E_c[j])
                di_er_xiang = para.alpha1 * math.log(1 + (q_dl[j] / iotd[i][j].data_size))
                di_san_xiang = para.alpha2 * math.log(1 + ((random_delta[j] * 2.4e9) / (para.omega2 * q_dl[j] * 8388608)))
                one_device_reward[j] = (c[j] / (E_c[j] + E_tr[j])) + di_er_xiang + di_san_xiang
        Q_dl.append(sum(q_dl))
    # print(i)
    # print("storage:",storage)
    # print("时延敏感：",Q_ds)
    # print("时延容忍",Q_dl)
        social_welfare.append(sum(one_device_reward))
        EE.append(sum(ee))
print("socail_welfare:", social_welfare)
print("ee:", EE)