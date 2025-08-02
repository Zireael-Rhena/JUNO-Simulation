import numpy as np
from sim.constants import *

def sample_uniform_sphere_vectorized(n):
    """
    球内均匀采样。

    使用逆变换采样方法在球内生成均匀分布的随机点。采用球坐标系统，
    确保径向、极角和方位角的分布都满足均匀性要求。

    Args:
        n (int): 需要采样的点数

    Returns:
        numpy.ndarray: 形状为 (n, 3) 的数组，包含球内均匀分布的顶点坐标

    Note:
        - 使用 r³ 均匀分布确保径向均匀性
        - 使用 cos(θ) 均匀分布确保极角均匀性
        - 球半径由常量 R_LS 定义

    Mathematical Background:
        对于球内均匀分布：
        - r ~ (R_LS * U^(1/3))，其中 U ~ Uniform(0,1)
        - θ ~ arccos(2V - 1)，其中 V ~ Uniform(0,1)  
        - φ ~ 2π * W，其中 W ~ Uniform(0,1)
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
    各向同性方向采样。

    在单位球面上生成均匀分布的方向向量，用于模拟各向同性的光子发射。
    使用球面均匀分布的标准算法。

    Args:
        n (int): 需要采样的方向数

    Returns:
        numpy.ndarray: 形状为 (n, 3) 的数组，包含单位球面上均匀分布的方向向量

    Note:
        - 返回的向量已归一化为单位向量
        - 适用于各向同性发射的物理过程
        - 使用标准的球面均匀分布算法

    Mathematical Background:
        对于单位球面均匀分布：
        - θ ~ arccos(2U - 1)，其中 U ~ Uniform(0,1)
        - φ ~ 2π * V，其中 V ~ Uniform(0,1)
    """
    # 生成随机数
    u = np.random.random(n)
    v = np.random.random(n)

    # 单位球面均匀采样
    theta = np.arccos(2*u - 1)  # cos(theta) 均匀分布
    phi = 2 * np.pi * v         # phi 均匀分布

    # 转换为笛卡尔坐标
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.column_stack([x, y, z])

def double_exponential_pdf(t):
    """
    双指数分布的概率密度函数。

    计算双指数分布的概率密度值，该分布常用于模拟闪烁体的发光时间分布。
    分布形式为：f(t) = C * exp(-t/τ_d) * (1-exp(-t/τ_r))

    Args:
        t (numpy.ndarray or float): 时间值或时间数组

    Returns:
        numpy.ndarray or float: 对应的概率密度值

    Note:
        - τ_d: 衰减时间常数 (TAU_D)
        - τ_r: 上升时间常数 (TAU_R)
        - C: 归一化系数 = (τ_d + τ_r) / τ_d²
        - 对于 t < 0，返回值为 0

    Physical Background:
        这种分布描述了闪烁体发光的时间特性：
        - exp(-t/τ_r) 项描述快速上升过程
        - exp(-t/τ_d) 项描述慢速衰减过程
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
    找到双指数分布PDF的最大值点和最大值。

    通过解析方法找到双指数分布概率密度函数的最大值点，
    用于优化接受-拒绝采样算法的效率。

    Returns:
        tuple: 包含两个元素的元组
            - t_max (float): 最大值点的时间
            - f_max (float): 最大概率密度值

    Note:
        - 当 τ_d = τ_r 时，最大值点为 t_max = τ_d
        - 当 τ_d ≠ τ_r 时，通过求导数为零的条件解析求解
        - 确保返回的 t_max > 0
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
    使用接受-拒绝方法采样光子发射时间。

    采用接受-拒绝算法从双指数分布中采样，使用截断指数分布作为
    提议分布以提高采样效率。支持大批量矢量化采样。

    Args:
        n (int): 需要采样的光子数

    Returns:
        numpy.ndarray: 长度为 n 的时间数组，服从双指数分布

    Note:
        - 使用指数分布 λ*exp(-λt) 作为提议分布，其中 λ = 1/τ_d
        - 批量采样以提高效率，默认批次大小为 min(n*3, 50000)
        - 包含防止无限循环的安全机制
        - 当 n = 0 时返回空数组

    Algorithm:
        1. 计算目标分布的最大值和位置
        2. 选择合适的提议分布
        3. 批量生成提议样本
        4. 应用接受-拒绝准则
        5. 收集足够数量的有效样本
    """
    if n == 0:
        return np.array([])

    # 找到PDF的最大值点和最大值
    _, f_max = find_pdf_maximum()

    # 为了提高效率，我们使用截断的指数分布作为提议分布
    # 提议分布: g(t) = λ * exp(-λt), 其中 λ = 1/τ_d
    lambda_prop = 1.0 / TAU_D

    # 计算接受率的上界
    maximum = f_max / (lambda_prop * np.exp(-lambda_prop * 0))  # 在t=0处比较

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
        accept_probs = f_vals / (maximum * g_vals)
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
    使用非齐次泊松过程采样光子发射时间。

    首先从泊松分布中采样光子总数，然后为这些光子采样发射时间。
    这模拟了闪烁体中光子发射的随机性质。

    Returns:
        numpy.ndarray: 光子发射时间数组，长度为泊松分布的随机数

    Note:
        - 光子总数服从参数为 N_PHOTONS 的泊松分布
        - 确保至少产生一个光子（最小值为1）
        - 发射时间服从双指数分布
    """
    # 首先确定光子总数（泊松分布）
    n_photons = np.random.poisson(N_PHOTONS)

    # 确保至少有一个光子
    n_photons = max(n_photons, 1)

    # 从归一化的PDF采样这么多个时间点
    return sample_emission_time_vectorized(n_photons)

def sample_single_vertex_photons():
    """
    为单个顶点采样光子时间和方向。

    为单个粒子相互作用顶点生成光子的发射时间和方向，
    模拟各向同性发射和时间分布特性。

    Returns:
        tuple: 包含两个元素的元组
            - times (numpy.ndarray): 光子发射时间数组
            - directions (numpy.ndarray): 光子发射方向数组，形状为 (n_photons, 3)

    Note:
        - 光子数量由泊松过程决定
        - 发射方向各向同性均匀分布
        - 发射时间服从双指数分布

    Usage Example:
        >>> times, directions = sample_single_vertex_photons()
        >>> print(f"Generated {len(times)} photons")
        >>> print(f"First photon: t={times[0]:.3f}ns, dir={directions[0]}")
    """
    # 采样光子发射时间
    times = sample_poisson_photon_times()
    n_photons = len(times)

    # 采样光子发射方向
    directions = sample_isotropic_direction_vectorized(n_photons)

    return times, directions

def sample_vertices_and_photons(n_vertices):
    """
    采样顶点和对应的光子。

    生成指定数量的粒子相互作用顶点，并为每个顶点采样相应的光子。

    Args:
        n_vertices (int): 需要生成的顶点数量

    Returns:
        tuple: 包含两个元素的元组
            - vertices (numpy.ndarray): 形状为 (n_vertices, 3) 的顶点位置数组
            - photon_data (list): 包含每个顶点光子信息的字典列表

    Note:
        - 顶点在球内均匀分布
        - 每个顶点的光子数量由泊松过程决定
        - photon_data 中每个字典包含：
          - 'vertex_id': 顶点ID
          - 'times': 光子发射时间数组
          - 'directions': 光子发射方向数组
          - 'n_photons': 光子总数

    Performance:
        - 对顶点位置采用矢量化采样
        - 对光子采样采用逐顶点循环（因为每个顶点的光子数不同）

    Usage Example:
        >>> vertices, photon_data = sample_vertices_and_photons(100)
        >>> total_photons = sum(data['n_photons'] for data in photon_data)
        >>> print(f"Generated {len(vertices)} vertices with {total_photons} total photons")
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
