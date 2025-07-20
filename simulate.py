import argparse
import sys
import numpy as np
import time
import psutil
from scipy.spatial import KDTree
from functools import lru_cache
import h5py as h5
from tqdm import tqdm

from sim.compute import *
from sim.sample import *

# 处理命令行
parser = argparse.ArgumentParser(description="JUNO detector simulation")
parser.add_argument("-n", dest="n", type=int, required=True, help="Number of events to simulate")
parser.add_argument("-g", "--geo", dest="geo", type=str, required=True, help="Geometry file path")
parser.add_argument("-o", "--output", dest="opt", type=str, required=True, help="Output file path")

print(f"JUNO Detector Simulation Parameters:")
print(f"  Photons per vertex: {N_PHOTONS} (fixed)")
print(f"  τ_d = {TAU_D} ns, τ_r = {TAU_R} ns")
print(f"  Intensity function coefficient A = {A:.1f}")
print(f"  PDF normalization coefficient = {PDF_NORMALIZATION:.3f}")
print(f"  Integral verification: {INTEGRAL_VALUE:.3f}")
print(f"  Light speeds: C_LS = {C_LS:.3f} mm/ns, C_WATER = {C_WATER:.3f} mm/ns")

# 优化的PMT模拟器类
class OptimizedPMTSimulator:
    """优化的PMT模拟器类"""
    
    def __init__(self, pmt_positions, channel_ids):
        self.pmt_positions = pmt_positions
        self.channel_ids = channel_ids
        self.pmt_kdtree = None
        self.pmt_normals = None
        self._initialize_pmt_data()
    
    def _initialize_pmt_data(self):
        """初始化PMT数据结构"""
        # 创建PMT位置的KDTree索引
        self.pmt_kdtree = KDTree(self.pmt_positions)
        
        # 计算PMT法向量（指向球心）
        self.pmt_normals = -self.pmt_positions / np.linalg.norm(self.pmt_positions, axis=1, keepdims=True)
        
    
    def geometric_prefilter(self, vertex_pos, refract_origins, refract_directions, max_angle_cos=-0.3):
        """基于几何约束预筛选可能击中的PMT"""
        # 计算顶点到PMT的方向向量
        vertex_to_pmts = self.pmt_positions - vertex_pos
        vertex_to_pmts_norm = vertex_to_pmts / np.linalg.norm(vertex_to_pmts, axis=1, keepdims=True)
        
        # 对于每个折射光子，计算与PMT方向的夹角
        cos_angles = np.dot(refract_directions, vertex_to_pmts_norm.T)
        
        # 如果任何光子可能击中该PMT，则保留
        viable_pmt_mask = np.any(cos_angles > max_angle_cos, axis=0)
        
        return np.where(viable_pmt_mask)[0]
    
    def kdtree_prefilter(self, refract_origins, refract_directions, search_radius=50000):
        """使用KDTree进行PMT预筛选"""
        # 计算光线在PMT球面上的大致投影点
        sphere_center = np.array([0, 0, 0])
        estimated_distances = np.linalg.norm(refract_origins - sphere_center, axis=1)
        
        # 估计光线与PMT球面的交点
        projection_points = refract_origins + refract_directions * estimated_distances[:, np.newaxis]
        
        # 使用KDTree查找附近的PMT
        nearby_pmts = set()
        for point in projection_points:
            distances, indices = self.pmt_kdtree.query(point, k=10, distance_upper_bound=search_radius)
            valid_indices = indices[distances < np.inf]
            nearby_pmts.update(valid_indices)
        
        return list(nearby_pmts)
    
    def compute_fresnel_reflection(self, incident_dirs, normals, n1=N_LS, n2=N_WATER):
        """计算菲涅尔反射系数"""
        cos_i = np.abs(np.sum(incident_dirs * normals, axis=1))
        cos_i = np.clip(cos_i, 0, 1)
        
        sin_i = np.sqrt(1 - cos_i**2)
        sin_t = (n1/n2) * sin_i
        
        # 处理全反射情况
        total_reflection = sin_t >= 1.0
        cos_t = np.sqrt(np.maximum(0, 1 - sin_t**2))
        
        # s偏振和p偏振反射系数
        Rs = np.zeros_like(cos_i)
        Rp = np.zeros_like(cos_i)
        
        valid_mask = ~total_reflection
        if np.any(valid_mask):
            Rs[valid_mask] = ((n1*cos_i[valid_mask] - n2*cos_t[valid_mask]) / 
                             (n1*cos_i[valid_mask] + n2*cos_t[valid_mask]))**2
            Rp[valid_mask] = ((n1*cos_t[valid_mask] - n2*cos_i[valid_mask]) / 
                             (n1*cos_t[valid_mask] + n2*cos_i[valid_mask]))**2
        
        # 全反射情况
        Rs[total_reflection] = 1.0
        Rp[total_reflection] = 1.0
        
        return (Rs + Rp) / 2

def validate_physics(vertex_pos, photon_dirs, intersections=None):
    """验证物理约束"""
    # 检查顶点是否在液闪内
    r_vertex = np.linalg.norm(vertex_pos)
    if r_vertex >= R_LS:
        print(f"Warning: Vertex outside LS sphere: r={r_vertex:.1f}mm")
        return False

    # 检查光子方向是否归一化
    norms = np.linalg.norm(photon_dirs, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-6):
        print("Warning: Photon directions not normalized")
        return False

    return True

def simulate_vertex_to_pmts(vertex_pos, photon_dirs, emission_times, pmt_positions, channel_ids, target_pmt_id=None, batch_size=50000):
    """
    模拟单个顶点到PMT的光子传输
    target_pmt_id: 如果指定，只模拟该PMT；如果为None，模拟所有PMT
    """
    n_photons = len(photon_dirs)
    hit_data = []

    # 物理约束检查
    if not validate_physics(vertex_pos, photon_dirs):
        print("Warning: Physical constraints violated")

    # 创建优化的PMT模拟器（仅在模拟所有PMT时使用）
    pmt_simulator = None
    if target_pmt_id is None and len(pmt_positions) > 1000:
        pmt_simulator = OptimizedPMTSimulator(pmt_positions, channel_ids)

    # 分批处理光子以节省内存
    for batch_start in range(0, n_photons, batch_size):
        batch_end = min(batch_start + batch_size, n_photons)

        # 当前批次的光子
        batch_dirs = photon_dirs[batch_start:batch_end]
        batch_emit_times = emission_times[batch_start:batch_end]
        batch_size_actual = batch_end - batch_start

        # 起始位置（所有光子从同一顶点发射）
        batch_origins = np.tile(vertex_pos, (batch_size_actual, 1))

        # 计算与液闪边界的交点
        ls_intersections, ls_hit_mask = ray_sphere_intersection_vectorized(
            batch_origins, batch_dirs, np.array([0.0, 0.0, 0.0]), R_LS)

        if not np.any(ls_hit_mask):
            continue

        # 只处理击中液闪边界的光子
        valid_origins = batch_origins[ls_hit_mask]
        valid_dirs = batch_dirs[ls_hit_mask]
        valid_emit_times = batch_emit_times[ls_hit_mask]
        valid_intersections = ls_intersections[ls_hit_mask]

        # 计算法向量（指向外部）
        normals = valid_intersections / np.linalg.norm(valid_intersections, axis=1)[:, np.newaxis]

        # 计算液闪中的传播距离和时间
        ls_distances = np.linalg.norm(valid_intersections - valid_origins, axis=1)
        ls_times = ls_distances / C_LS

        # 计算折射
        refract_dirs, total_reflection = compute_refraction_vectorized(
            valid_dirs, normals, N_LS, N_WATER)

        # 只处理折射的光子（忽略全反射）
        refract_mask = ~total_reflection

        if not np.any(refract_mask):
            continue

        # 折射后的光子参数
        refract_origins = valid_intersections[refract_mask]
        refract_directions = refract_dirs[refract_mask]
        refract_emit_times = valid_emit_times[refract_mask]
        refract_ls_times = ls_times[refract_mask]

        # 决定要检查的PMT
        if target_pmt_id is not None:
            # 处理不同类型的target_pmt_id
            if isinstance(target_pmt_id, (int, np.integer)):
                # 单个PMT
                pmt_indices = [target_pmt_id] if target_pmt_id < len(pmt_positions) else []
            elif isinstance(target_pmt_id, (list, np.ndarray)):
                # 多个PMT，过滤掉超出范围的
                pmt_indices = [idx for idx in target_pmt_id if 0 <= idx < len(pmt_positions)]
            else:
                raise ValueError(f"target_pmt_id must be int, list, or None, got {type(target_pmt_id)}")
            # print(f"Debug: Single PMT mode, PMT indices: {pmt_indices}")
        else:
            # 模拟所有PMT
            if pmt_simulator is not None:
                # 使用几何预筛选
                viable_pmt_indices = pmt_simulator.geometric_prefilter(
                    vertex_pos, refract_origins, refract_directions
                )
                # print(f"Debug: Geometric prefilter returned {len(viable_pmt_indices)} PMTs")
                
                # 如果PMT太多，再使用KDTree筛选
                if len(viable_pmt_indices) > 1000:
                    kdtree_pmts = pmt_simulator.kdtree_prefilter(refract_origins, refract_directions)
                    # print(f"Debug: KDTree prefilter returned {len(kdtree_pmts)} PMTs")
                    viable_pmt_indices = list(set(viable_pmt_indices) & set(kdtree_pmts))
                    # print(f"Debug: Intersection returned {len(viable_pmt_indices)} PMTs")
                
                pmt_indices = viable_pmt_indices
            else:
                # 小规模PMT，使用原有策略
                pmt_indices = range(len(pmt_positions))
                # print(f"Debug: Using all {len(pmt_indices)} PMTs (no optimizer)")

        # print(f"Debug: Final PMT indices count: {len(pmt_indices)}")
        # 对选定的PMT检查交点
        for pmt_id in pmt_indices:
            pmt_pos = pmt_positions[pmt_id]

            # 计算与PMT的交点
            pmt_intersections, pmt_hit_mask = ray_sphere_intersection_vectorized(
                refract_origins, refract_directions, pmt_pos, R_PMT_SPHERE)

            if not np.any(pmt_hit_mask):
                continue

            # 使用改进的前表面判断
            pmt_intersections_valid = pmt_intersections[pmt_hit_mask]
            front_surface_mask = is_front_surface_improved(pmt_pos, pmt_intersections_valid)

            if not np.any(front_surface_mask):
                continue

            # 应用菲涅尔反射
            if pmt_simulator is not None:
                pmt_normal = pmt_simulator.pmt_normals[pmt_id]
                incident_dirs = refract_directions[pmt_hit_mask][front_surface_mask]
                normals = np.tile(pmt_normal, (len(incident_dirs), 1))
                
                reflection_probs = pmt_simulator.compute_fresnel_reflection(incident_dirs, normals)
                
                # 根据反射概率决定光子命运
                rng = np.random.default_rng()
                random_probs = rng.random(len(reflection_probs))
                transmitted_mask = random_probs > reflection_probs
                
                if not np.any(transmitted_mask):
                    continue
                
                # 只保留透射的光子
                final_pmt_intersections = pmt_intersections_valid[front_surface_mask][transmitted_mask]
                final_refract_origins = refract_origins[pmt_hit_mask][front_surface_mask][transmitted_mask]
                final_emit_times = refract_emit_times[pmt_hit_mask][front_surface_mask][transmitted_mask]
                final_ls_times = refract_ls_times[pmt_hit_mask][front_surface_mask][transmitted_mask]
            else:
                # 不考虑反射
                final_pmt_intersections = pmt_intersections_valid[front_surface_mask]
                final_refract_origins = refract_origins[pmt_hit_mask][front_surface_mask]
                final_emit_times = refract_emit_times[pmt_hit_mask][front_surface_mask]
                final_ls_times = refract_ls_times[pmt_hit_mask][front_surface_mask]

            # 计算在水中的传播距离和时间
            water_distances = np.linalg.norm(final_pmt_intersections - final_refract_origins, axis=1)
            water_times = water_distances / C_WATER

            # 计算总hit time = 发射时间 + 液闪传播时间 + 水中传播时间
            total_hit_times = final_emit_times + final_ls_times + water_times

            # 收集结果
            for hit_time in total_hit_times:
                hit_data.append((channel_ids[pmt_id], hit_time))

    return hit_data

def main():
    # 解析命令行参数
    args = parser.parse_args()

    # 读取几何文件
    try:
        with h5.File(args.geo, "r") as geo:
            geo_data = geo["Geometry"]
            pmt_theta = geo_data["theta"][:]
            pmt_phi = geo_data["phi"][:]
            channel_ids = geo_data["ChannelID"][:]

            # 转换为弧度并计算PMT位置
            pmt_theta_rad = np.radians(pmt_theta)
            pmt_phi_rad = np.radians(pmt_phi)

            # 计算PMT在球坐标系中的位置
            pmt_x = R_PMT * np.sin(pmt_theta_rad) * np.cos(pmt_phi_rad)
            pmt_y = R_PMT * np.sin(pmt_theta_rad) * np.sin(pmt_phi_rad)
            pmt_z = R_PMT * np.cos(pmt_theta_rad)

            pmt_positions = np.column_stack((pmt_x, pmt_y, pmt_z))

            # 添加调试信息
            if TARGET_PMT_ID is not None:
                if isinstance(TARGET_PMT_ID, (int, np.integer)):
                    # 单个PMT
                    if TARGET_PMT_ID < len(pmt_positions):
                        print(f"Simulating single PMT: ID={TARGET_PMT_ID}, Channel={channel_ids[TARGET_PMT_ID]}")
                        print(f"PMT position: ({pmt_x[TARGET_PMT_ID]:.1f}, {pmt_y[TARGET_PMT_ID]:.1f}, {pmt_z[TARGET_PMT_ID]:.1f}) mm")
                    else:
                        print(f"Error: TARGET_PMT_ID {TARGET_PMT_ID} is out of range (0-{len(pmt_positions)-1})")
                        sys.exit(1)
                elif isinstance(TARGET_PMT_ID, (list, np.ndarray)):
                    # 多个PMT
                    valid_ids = [idx for idx in TARGET_PMT_ID if 0 <= idx < len(pmt_positions)]
                    if len(valid_ids) != len(TARGET_PMT_ID):
                        invalid_ids = [idx for idx in TARGET_PMT_ID if not (0 <= idx < len(pmt_positions))]
                        print(f"Warning: Some PMT IDs are out of range: {invalid_ids}")
                    print(f"Simulating {len(valid_ids)} PMTs: {valid_ids}")
                else:
                    print(f"Error: TARGET_PMT_ID must be int, list, or None, got {type(TARGET_PMT_ID)}")
                    sys.exit(1)
            else:
                print(f"Simulating all {len(pmt_positions)} PMTs")

    except Exception as e:
        print(f"Error reading geometry file '{args.geo}': {e}")
        sys.exit(1)

    # 准备输出数据
    particle_data = []
    pe_data = []

    # 设置随机种子
    np.random.seed(42)

    # 存储第一个事件的统计信息
    first_event_stats = None

    # 记录开始时间
    simulation_start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

    # 模拟每个事件
    print("Starting optimized simulation...")
    for event_id in tqdm(range(args.n), desc="Simulating events"):
        # 生成顶点位置
        vertex_positions = sample_uniform_sphere_vectorized(1)
        vertex_pos = vertex_positions[0]

        # 记录粒子信息
        particle_data.append((event_id, vertex_pos[0], vertex_pos[1], vertex_pos[2], MOMENTUM))

        # 生成固定数量的光子
        photon_dirs = sample_isotropic_direction_vectorized(N_PHOTONS)
        emission_times = sample_emission_time_vectorized(N_PHOTONS)

        # 模拟光子传输（使用优化版本）
        hit_data = simulate_vertex_to_pmts(
            vertex_pos, photon_dirs, emission_times, pmt_positions, channel_ids, TARGET_PMT_ID)

        # 记录PE信息
        for channel_id, hit_time in hit_data:
            pe_data.append((event_id, int(channel_id), hit_time))

        # 保存第一个事件的统计信息
        if event_id == 0:
            hit_times = [hit_time for _, hit_time in hit_data] if hit_data else []
            first_event_stats = {
                'position': vertex_pos,
                'emission_times': emission_times,
                'hit_count': len(hit_data),
                'hit_times': hit_times
            }

    # 记录结束时间
    simulation_end_time = time.time()
    end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

    # 输出性能统计
    print(f"\nOptimized simulation completed:")
    print(f"  Total execution time: {simulation_end_time - simulation_start_time:.2f} seconds")
    print(f"  Memory usage: {end_memory - start_memory:.2f} MB")
    print(f"  Events per second: {args.n / (simulation_end_time - simulation_start_time):.2f}")

    # 输出第一个事件的统计信息（在进度条完成后）
    if first_event_stats:
        print(f"\nEvent 0 statistics:")
        print(f"  Position: ({first_event_stats['position'][0]:.1f}, {first_event_stats['position'][1]:.1f}, {first_event_stats['position'][2]:.1f}) mm")
        print(f"  Emission times: mean={np.mean(first_event_stats['emission_times']):.2f}ns, std={np.std(first_event_stats['emission_times']):.2f}ns")
        print(f"  Min/Max emission time: {np.min(first_event_stats['emission_times']):.2f}ns / {np.max(first_event_stats['emission_times']):.2f}ns")
        print(f"  Generated {first_event_stats['hit_count']} PEs")

        # 额外的物理检查
        if first_event_stats['hit_times']:
            print(f"  Hit times: mean={np.mean(first_event_stats['hit_times']):.2f}ns, std={np.std(first_event_stats['hit_times']):.2f}ns")
            print(f"  Min/Max hit time: {np.min(first_event_stats['hit_times']):.2f}ns / {np.max(first_event_stats['hit_times']):.2f}ns")

    # 输出结果
    print(f"\nSimulation results:")
    print(f"  Generated {len(pe_data)} PEs from {args.n} events")
    print(f"  Average PEs per event: {len(pe_data) / args.n:.2f}")

    # 保存结果
    try:
        with h5.File(args.opt, "w") as opt:
            # 创建ParticleTruth数据集
            particle_truth = opt.create_dataset("ParticleTruth", (len(particle_data),),
                                               dtype=[("EventID", "<i4"),
                                                      ("x", "<f8"),
                                                      ("y", "<f8"),
                                                      ("z", "<f8"),
                                                      ("p", "<f8")])

            # 填充粒子数据
            for i, (event_id, x, y, z, p) in enumerate(particle_data):
                particle_truth[i] = (event_id, x, y, z, p)

            # 创建PETruth数据集
            if pe_data:
                pe_truth = opt.create_dataset("PETruth", (len(pe_data),),
                                            dtype=[("EventID", "<i4"),
                                                   ("ChannelID", "<i4"),
                                                   ("PETime", "<f8")])

                # 填充PE数据
                for i, (event_id, channel_id, pe_time) in enumerate(pe_data):
                    pe_truth[i] = (event_id, channel_id, pe_time)
            else:
                # 创建空数据集
                pe_truth = opt.create_dataset("PETruth", (0,),
                                            dtype=[("EventID", "<i4"),
                                                   ("ChannelID", "<i4"),
                                                   ("PETime", "<f8")])

            # 添加元数据
            opt.attrs["target_pmt_id"] = TARGET_PMT_ID if TARGET_PMT_ID is not None else -1
            opt.attrs["n_events"] = args.n
            opt.attrs["n_photons_per_event"] = N_PHOTONS
            opt.attrs["tau_d"] = TAU_D
            opt.attrs["tau_r"] = TAU_R
            opt.attrs["intensity_coeff"] = A
            opt.attrs["pdf_normalization"] = PDF_NORMALIZATION
            opt.attrs["c_ls"] = C_LS
            opt.attrs["c_water"] = C_WATER
            opt.attrs["c_vacuum"] = C_0
            opt.attrs["optimization_enabled"] = True
            opt.attrs["execution_time"] = simulation_end_time - simulation_start_time

            print(f"Results saved to {args.opt}")

    except Exception as e:
        print(f"Error writing output file '{args.opt}': {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
