import random
import numpy as np


class IoTDevice:
    def __init__(self, data_size, delay_sensitive):
        self.data_size = data_size
        self.delay_sensitive = delay_sensitive


np.random.seed(0)
devices = []
Q_ds = 0
Q_dl = 0
for _ in range(8):
    group = []  # 8组IoT 设备 每组10个
    for _ in range(3):
        data_size = np.random.randint(100, 201)  #(50,100)
        delay_sensitive = 1
        device = IoTDevice(data_size, delay_sensitive)
        group.append(device)

    for _ in range(7):
        data_size = np.random.randint(50, 101)
        delay_sensitive = 0
        device = IoTDevice(data_size, delay_sensitive)
        group.append(device)
    devices.append(group)


#打印每组每个设备数据大小
for i in range(8):
    print("第",i,"组")
    for j in  range(10):
        print(devices[i][j].data_size)
        print()

# for group in devices:
#     for device in group:
#         print(f"数据大小: {device.data_size}, 是否时延敏感: {device.delay_sensitive}")
#     print('---')
#     #打印生成的设备信息
# for i, device in enumerate(devices):
#     if devices[i].delay_sensitive == 1:
#         Q_ds += devices[i].data_size
#     else:
#         Q_dl += devices[i].data_size
#     print("Device", i+1)
#     print("delay_sensitive:", device.delay_sensitive)
#     print("Data Size:", device.data_size, "MB")
#     print()
# print("Q_ds:",Q_ds)
# print("Q_dl",Q_dl)
