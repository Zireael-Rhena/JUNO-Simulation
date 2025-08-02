# 物理常数
R_LS = 17700.0  # 液闪球半径 (mm)
R_PMT = 19500.0  # PMT到球心距离 (mm)
R_PMT_SPHERE = 254.0  # PMT球体半径 (mm)
N_PHOTONS = 10000  # 每个顶点产生的光子数
MOMENTUM = 1.0  # 顶点动量 (MeV)

# 光学常数 - 统一为 mm/ns 单位
N_LS = 1.48  # 液闪折射率
N_WATER = 1.33  # 水的折射率
C_0 = 299.792458  # 真空中光速
C_LS = C_0 / N_LS  # 液闪中光速
C_WATER = C_0 / N_WATER  # 水中光速

# 时间常数
TAU_D = 10.0  # 衰减时间常数 (ns)
TAU_R = 5.0   # 上升时间常数 (ns)

# 控制变量：设置要模拟的PMT ID，如果为None则模拟所有PMT
TARGET_PMT_ID = None  # 修改这个值：0=模拟PMT 0，None=模拟所有PMT

# 归一化系数
# 非齐次泊松过程强度函数的积分
INTEGRAL_VALUE = TAU_D**2 / (TAU_D + TAU_R)  # = 100/15 = 20/3
A = N_PHOTONS / INTEGRAL_VALUE  # = 10000 / (20/3) = 1500

# 概率密度函数的归一化系数
PDF_NORMALIZATION = A / N_PHOTONS  # = 1500 / 10000 = 0.15
