# 创建一个叫 diagnose.py 的文件
import h5py as h5
import numpy as np
import argparse

def diagnose_data(sim_file, geo_file):
    """诊断模拟数据"""
    
    sim_data = h5.File(sim_file, "r")
    geo_data = h5.File(geo_file, "r")
    
    # 读取数据
    particle_truth = sim_data["ParticleTruth"]
    pe_truth = sim_data["PETruth"]
    geometry = geo_data["Geometry"]
    
    # 顶点位置
    vertex_coords = np.column_stack([
        particle_truth["x"][:],
        particle_truth["y"][:],
        particle_truth["z"][:]
    ])
    
    vertex_distances = np.linalg.norm(vertex_coords, axis=1)
    
    print("=== 数据诊断 ===")
    print(f"总事件数: {len(vertex_coords)}")
    print(f"顶点位置范围:")
    print(f"  X: {np.min(vertex_coords[:,0]):.1f} 到 {np.max(vertex_coords[:,0]):.1f}")
    print(f"  Y: {np.min(vertex_coords[:,1]):.1f} 到 {np.max(vertex_coords[:,1]):.1f}")
    print(f"  Z: {np.min(vertex_coords[:,2]):.1f} 到 {np.max(vertex_coords[:,2]):.1f}")
    print(f"  径向距离: {np.min(vertex_distances):.1f} 到 {np.max(vertex_distances):.1f}")
    
    print(f"前5个顶点:")
    for i in range(min(5, len(vertex_coords))):
        x, y, z = vertex_coords[i]
        r = np.sqrt(x**2 + y**2 + z**2)
        print(f"  事件{i}: ({x:.1f}, {y:.1f}, {z:.1f}) r={r:.1f}")
    
    # PE数据
    pe_channels = pe_truth["ChannelID"][:]
    pe_events = pe_truth["EventID"][:]
    
    print(f"PE数据:")
    print(f"  总PE数: {len(pe_channels)}")
    print(f"  击中PMT数: {len(set(pe_channels))}")
    print(f"  总PMT数: {len(geometry['ChannelID'][:])}")
    print(f"  PMT击中率: {len(set(pe_channels))/len(geometry['ChannelID'][:]) * 100:.1f}%")
    
    sim_data.close()
    geo_data.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sim_file", help="Simulation file")
    parser.add_argument("geo_file", help="Geometry file")
    args = parser.parse_args()
    
    diagnose_data(args.sim_file, args.geo_file)