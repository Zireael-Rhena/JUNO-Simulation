import numpy as np
from sim.constants import *

def sample_uniform_sphere_vectorized(n):
    """
    完全矢量化的球内均匀采样
    
    参数:
        n: 采样点数
    
    返回:
        vertices: (n, 3) 数组，球内均匀分布的顶点
    """
    # 生成随机数
    u = np.random.random(n)
    v = np.random.random(n)
    w = np.random.random(n)
    
    # 球内均匀采样
    r = R_LS * np.power(u, 1.0/3.0)  # r^3 均匀分布
    theta = np.arccos(2*v - 1)       # cos(theta) 均匀分布
    phi = 2 * np.pi * w             # phi 均匀分布
    
    # 转换为笛卡尔坐标
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    return np.column_stack([x, y, z])

def sample_isotropic_direction_vectorized(n):
    """
    完全矢量化的各向同性方向采样
    
    参数:
        n: 采样方向数
    
    返回:
        directions: (n, 3) 数组，单位球面上的均匀分布方向
    """
    # 生成随机数
    u = np.random.random(n)
    v = np.random.random(n)
    
    # 单位球面均匀采样
    theta = np.arccos(2*u - 1)  # cos(theta) 均匀分布
    phi = 2 * np.pi * v        # phi 均匀分布
    
    # 转换为笛卡尔坐标
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    return np.column_stack([x, y, z])

def double_exponential_pdf(t):
    """
    双指数分布的概率密度函数
    f(t) = C * exp(-t/τ_d) * (1-exp(-t/τ_r))
    其中 C = (τ_d + τ_r) / τ_d² 是归一化系数
    
    参数:
        t: 时间数组
    
    返回:
        pdf_values: 对应的概率密度值
    """
    # 确保t >= 0
    t = np.maximum(t, 0)
    
    # 计算指数项
    exp_d = np.exp(-t / TAU_D)
    exp_r = np.exp(-t / TAU_R)
    
    # 归一化系数
    C = (TAU_D + TAU_R) / (TAU_D * TAU_D)
    
    return C * exp_d * (1 - exp_r)

def find_pdf_maximum():
    """
    找到双指数分布PDF的最大值点和最大值
    用于接受-拒绝采样的优化
    """
    # 解析求导，找到最大值点
    # df/dt = C * [-1/τ_d * exp(-t/τ_d) * (1-exp(-t/τ_r)) + exp(-t/τ_d) * (1/τ_r * exp(-t/τ_r))]
    # 令 df/dt = 0，可得最大值点
    
    if TAU_D == TAU_R:
        t_max = TAU_D
    else:
        t_max = TAU_D * TAU_R * np.log(TAU_D / TAU_R) / (TAU_D - TAU_R)
    
    # 确保t_max为正值
    t_max = max(t_max, 0.01)
    
    # 计算最大值
    f_max = double_exponential_pdf(t_max)
    
    return t_max, f_max

def sample_emission_time_vectorized(n):
    """
    使用接受-拒绝方法采样光子发射时间
    
    参数:
        n: 需要采样的光子数
    
    返回:
        times: 长度为n的时间数组
    """
    if n == 0:
        return np.array([])
    
    # 找到PDF的最大值点和最大值
    t_max, f_max = find_pdf_maximum()
    
    # 为了提高效率，我们使用截断的指数分布作为提议分布
    # 提议分布: g(t) = λ * exp(-λt), 其中 λ = 1/τ_d
    lambda_prop = 1.0 / TAU_D
    
    # 计算接受率的上界
    M = f_max / (lambda_prop * np.exp(-lambda_prop * 0))  # 在t=0处比较
    
    # 预分配数组
    times = []
    
    # 批量采样以提高效率
    batch_size = min(n * 3, 50000)  # 预期需要采样更多样本
    
    while len(times) < n:
        # 从提议分布采样
        u1 = np.random.random(batch_size)
        u2 = np.random.random(batch_size)
        
        # 指数分布的逆变换采样
        t_proposals = -np.log(u1) / lambda_prop
        
        # 计算接受概率
        f_vals = double_exponential_pdf(t_proposals)
        g_vals = lambda_prop * np.exp(-lambda_prop * t_proposals)
        
        # 避免除零
        g_vals = np.maximum(g_vals, 1e-15)
        
        # 接受条件
        accept_probs = f_vals / (M * g_vals)
        accept_mask = u2 < accept_probs
        
        # 收集接受的样本
        accepted_times = t_proposals[accept_mask]
        times.extend(accepted_times)
        
        # 避免无限循环
        if len(times) > n * 10:  # 如果采样效率太低
            break
    
    return np.array(times[:n])

def sample_poisson_photon_times():
    """
    使用非齐次泊松过程采样光子发射时间
    
    返回:
        times: 光子发射时间数组，长度为泊松分布的随机数
    """
    # 首先确定光子总数（泊松分布）
    n_photons = np.random.poisson(N_PHOTONS)
    
    # 确保至少有一个光子
    n_photons = max(n_photons, 1)
    
    # 从归一化的PDF采样这么多个时间点
    return sample_emission_time_vectorized(n_photons)

def sample_single_vertex_photons():
    """
    为单个顶点采样光子时间和方向
    
    返回:
        times: 光子发射时间数组
        directions: 光子发射方向数组 (n_photons, 3)
    """
    # 采样光子发射时间
    times = sample_poisson_photon_times()
    n_photons = len(times)
    
    # 采样光子发射方向
    directions = sample_isotropic_direction_vectorized(n_photons)
    
    return times, directions

def sample_vertices_and_photons(n_vertices):
    """
    采样顶点和对应的光子
    
    参数:
        n_vertices: 顶点数量
    
    返回:
        vertices: (n_vertices, 3) 顶点位置数组
        photon_data: 包含每个顶点光子信息的列表
    """
    # 采样顶点位置
    vertices = sample_uniform_sphere_vectorized(n_vertices)
    
    # 为每个顶点采样光子
    photon_data = []
    for i in range(n_vertices):
        times, directions = sample_single_vertex_photons()
        photon_data.append({
            'vertex_id': i,
            'times': times,
            'directions': directions,
            'n_photons': len(times)
        })
    
    return vertices, photon_data
