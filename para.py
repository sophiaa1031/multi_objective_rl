import math
phi = 1.42e-4
g = phi / 3600
sigma = 1e-10 # 1e-16
Omu = 1e-28
omega1 = 1500   # ds = 1 计算1bit需要的cycles，应该放在IoT device里
omega2 = 1000   # ds = 0
alpha1= 1
alpha2 = 100
IoTD_p = 5
rescale = 600
#print(6e7*math.log(1+IoTD_p*g/sigma)*1.25e-7)
