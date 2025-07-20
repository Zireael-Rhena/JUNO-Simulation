"""
Final Correct Probe Function R(r,θ) for JUNO detector
Calculates average response of ALL PMTs as function of vertex-PMT geometry
"""

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as colors
from scipy.interpolate import griddata
import argparse

# Physical constants
DETECTOR_RADIUS = 17700.0  # mm
PMT_SPHERE_RADIUS = 19500.0  # mm
DETECTOR_VOLUME = 4/3 * np.pi * DETECTOR_RADIUS**3

class AllPMTProbeFunction:
    """Calculate Probe Function R(r,θ) using ALL PMT average response"""
    
    def __init__(self, simulation_file, geometry_file):
        """Initialize with simulation and geometry data"""
        self.sim_data = h5.File(simulation_file, "r")
        self.geo_data = h5.File(geometry_file, "r")
        
        # Extract datasets
        self.particle_truth = self.sim_data["ParticleTruth"]
        self.pe_truth = self.sim_data["PETruth"]
        self.geometry = self.geo_data["Geometry"]
        
        # Load and preprocess data
        self._load_data()
        
    def _load_data(self):
        """Load and preprocess all data"""
        # Vertex positions
        self.vertices = np.column_stack([
            self.particle_truth["x"][:],
            self.particle_truth["y"][:],
            self.particle_truth["z"][:]
        ])
        
        # PE data
        self.pe_events = self.pe_truth["EventID"][:]
        self.pe_channels = self.pe_truth["ChannelID"][:]
        self.pe_times = self.pe_truth["PETime"][:]
        
        # PMT geometry (convert to radians)
        self.pmt_theta = self.geometry["theta"][:] * np.pi / 180.0
        self.pmt_phi = self.geometry["phi"][:] * np.pi / 180.0
        self.pmt_channels = self.geometry["ChannelID"][:]

        target_pmt_id = self.sim_data.attrs["target_pmt_id"]
        self.hit_channel_count = len(target_pmt_id) if isinstance(target_pmt_id, np.ndarray) else 1 if target_pmt_id is not None else len(self.pmt_channels)
        
        # Calculate PMT Cartesian coordinates
        self.pmt_positions = np.column_stack([
            PMT_SPHERE_RADIUS * np.sin(self.pmt_theta) * np.cos(self.pmt_phi),
            PMT_SPHERE_RADIUS * np.sin(self.pmt_theta) * np.sin(self.pmt_phi),
            PMT_SPHERE_RADIUS * np.cos(self.pmt_theta)
        ])
        
        print(f"Loaded data: {len(self.vertices)} events, {len(self.pe_events)} PEs, {len(self.pmt_channels)} PMTs")
        
        # Create lookup tables for faster processing
        self._create_lookup_tables()

        self.diagnose_pe_vertices()

    def diagnose_pe_vertices(self):
        """诊断产生PE的顶点分布"""
        print("=== PE顶点分布诊断 ===")
    
        # 获取产生PE的唯一事件
        pe_events_unique = np.unique(self.pe_events)
        print(f"产生PE的事件数: {len(pe_events_unique)}")
        print(f"总事件数: {len(self.vertices)}")
        print(f"PE产生率: {len(pe_events_unique)/len(self.vertices)*100:.1f}%")
    
        # 获取产生PE的顶点位置
        pe_vertices = []
        for event_id in pe_events_unique:
            if event_id < len(self.vertices):
                pe_vertices.append(self.vertices[event_id])
    
        pe_vertices = np.array(pe_vertices)
        pe_distances = np.linalg.norm(pe_vertices, axis=1)
    
        print(f"产生PE的顶点径向距离:")
        print(f"  范围: {np.min(pe_distances):.1f} - {np.max(pe_distances):.1f} mm")
        print(f"  平均: {np.mean(pe_distances):.1f} mm")
        print(f"  归一化范围: {np.min(pe_distances)/DETECTOR_RADIUS:.3f} - {np.max(pe_distances)/DETECTOR_RADIUS:.3f}")
    
        # 对比所有顶点的分布
        all_distances = np.linalg.norm(self.vertices, axis=1)
        print(f"所有顶点径向距离:")
        print(f"  范围: {np.min(all_distances):.1f} - {np.max(all_distances):.1f} mm")
        print(f"  平均: {np.mean(all_distances):.1f} mm")
        print(f"  归一化范围: {np.min(all_distances)/DETECTOR_RADIUS:.3f} - {np.max(all_distances)/DETECTOR_RADIUS:.3f}")
    
        # 检查没有产生PE的事件
        all_events = set(range(len(self.vertices)))
        pe_events_set = set(pe_events_unique)
        no_pe_events = all_events - pe_events_set
    
        print(f"没有产生PE的事件数: {len(no_pe_events)}")
        if len(no_pe_events) > 0:
            no_pe_vertices = self.vertices[list(no_pe_events)]
            no_pe_distances = np.linalg.norm(no_pe_vertices, axis=1)
            print(f"没有产生PE的顶点径向距离:")
            print(f"  范围: {np.min(no_pe_distances):.1f} - {np.max(no_pe_distances):.1f} mm")
            print(f"  平均: {np.mean(no_pe_distances):.1f} mm")
            print(f"  归一化范围: {np.min(no_pe_distances)/DETECTOR_RADIUS:.3f} - {np.max(no_pe_distances)/DETECTOR_RADIUS:.3f}")
        
    def _create_lookup_tables(self):
        """Create lookup tables for efficient data processing"""
        # Create channel to PMT index mapping
        self.channel_to_pmt = {}
        for i, channel in enumerate(self.pmt_channels):
            self.channel_to_pmt[channel] = i
            
        # Create event to vertex mapping
        self.event_to_vertex = {}
        for i in range(len(self.vertices)):
            self.event_to_vertex[i] = i
            
        print("Created lookup tables")
        
    def calculate_all_pmt_probe_function(self):
        """
        Calculate Probe function R(r,θ) using ALL PMT data
        
        Returns:
            r_values, theta_values, pe_counts: Arrays for probe function
        """
        print("Computing ALL-PMT Probe function R(r,θ)...")
        
        # Lists to store all (r,θ,PE) data points
        all_r = []
        all_theta = []
        
        # Process each PE hit
        print("Processing PE hits...")
        processed_count = 0

        for pe_idx in range(len(self.pe_events)):
            if processed_count % 50000 == 0:
                print(f"Processed {processed_count}/{len(self.pe_events)} PE hits")
                
            # Get PE information
            event_id = self.pe_events[pe_idx]
            channel_id = self.pe_channels[pe_idx]
            
            # Get vertex position for this event
            if event_id >= len(self.vertices):
                continue
            vertex_pos = self.vertices[event_id]
            
            # Get PMT position for this channel
            if channel_id not in self.channel_to_pmt:
                continue
            pmt_idx = self.channel_to_pmt[channel_id]
            pmt_pos = self.pmt_positions[pmt_idx]
            
            # Calculate r (normalized distance from detector center)
            r = np.linalg.norm(vertex_pos) / DETECTOR_RADIUS
            
            # Calculate θ (angle between vertex and PMT vectors from detector center)
            vertex_unit = vertex_pos / np.linalg.norm(vertex_pos)
            pmt_unit = pmt_pos / np.linalg.norm(pmt_pos)
            
            cos_theta = np.dot(vertex_unit, pmt_unit)
            cos_theta = np.clip(cos_theta, -1, 1)
            theta = np.arccos(cos_theta)
            
            # Store data point (each PE hit contributes 1 to the count)
            all_r.append(r)
            all_theta.append(theta)
            
            processed_count += 1
            
        print(f"Generated {len(all_r)} (r,θ,PE) data points")
        
        return np.array(all_r), np.array(all_theta)
    
    def create_vertex_density_plot(self, figure, axis):
        """Create vertex density plot"""
        # Calculate radial distances
        vertex_distances = np.linalg.norm(self.vertices, axis=1)
        
        # Create bins
        bin_edges = np.linspace(0, DETECTOR_RADIUS, 31)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
        # Calculate histogram
        counts, _ = np.histogram(vertex_distances, bins=bin_edges)
        
        # Calculate volumes of spherical shells
        volumes = 4 * np.pi / 3 * (bin_edges[1:]**3 - bin_edges[:-1]**3)
        
        # Calculate densities
        densities = counts / volumes
        
        # Normalize by average density
        avg_density = len(self.vertices) / DETECTOR_VOLUME
        normalized_densities = densities / avg_density
        
        # Plot
        axis.plot(bin_centers/DETECTOR_RADIUS, normalized_densities, 'ro-', linewidth=2, markersize=4)
        axis.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Uniform density')
        
        axis.set_xlabel('Radius from Origin r/R_LS', fontsize=12)
        axis.set_ylabel('Volume Density ρ(r)/ρ₀', fontsize=12)
        axis.set_title('Volume Density of Vertices ρ(r)', fontsize=14, fontweight='bold')
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 2)
        axis.grid(True, alpha=0.3)
        axis.legend()
        
    def create_pe_time_plot(self, figure, axis):
        """Create PE time histogram"""
        if len(self.pe_times) == 0:
            axis.text(0.5, 0.5, 'No PE data available', 
                     ha='center', va='center', transform=axis.transAxes, 
                     fontsize=14, color='red')
            return
            
        # Create histogram
        axis.hist(self.pe_times, bins=100, alpha=0.7, color='steelblue', edgecolor='black')
        
        axis.set_xlabel('Hit Time t / ns', fontsize=12)
        axis.set_ylabel('Number of PE Hit', fontsize=12)
        axis.set_title('Histogram of PE Hit Time', fontsize=14, fontweight='bold')
        axis.grid(True, alpha=0.3)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        
    def create_probe_function_plot(self, figure, axis):
        """Create the ALL-PMT Probe function plot"""
        # Calculate probe function using ALL PMT data
        r_values, theta_values = self.calculate_all_pmt_probe_function()
        
        print(f"Final probe function statistics:")
        print(f"  Data points: {len(r_values)}")
        print(f"  r range: {np.min(r_values):.3f} - {np.max(r_values):.3f}")
        print(f"  θ range: {np.min(theta_values):.3f} - {np.max(theta_values):.3f} rad ({np.min(theta_values)*180/np.pi:.1f}° - {np.max(theta_values)*180/np.pi:.1f}°)")
        print(f"  Total PE count: {len(self.pe_events)}")
        
        # Create 2D histogram to bin the data
        print("Creating 2D histogram...")
        n_r_bins = 50
        n_theta_bins = 50
        
        r_bins = np.linspace(0, 1, n_r_bins + 1)
        theta_bins = np.linspace(0, np.pi, n_theta_bins + 1)
        
        # Calculate 2D histogram
        H, r_edges, theta_edges = np.histogram2d(r_values, theta_values, 
                                                bins=[r_bins, theta_bins])

        H_avg = H / self.hit_channel_count
        
        # Get bin centers
        r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
        theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])

        # Correct for non-uniform vertex distribution in polar coordinates
        r_corr = r_centers
        theta_corr = np.sin(theta_centers)        
        # Create correction matrix
        correction_matrix = 2 * np.pi * np.outer(r_corr, theta_corr) * len(self.vertices) / DETECTOR_VOLUME        
        # Avoid division by zero
        correction_matrix[correction_matrix == 0] = 1e-9
        
        # Apply correction.
        probe_function = H_avg / correction_matrix.T # Transpose to match H_avg shape
        probe_function = probe_function / np.max(probe_function) * 1000
        
        # Extend to full circle for visualization (mirror the data)
        theta_full = np.concatenate([theta_centers, 2*np.pi - theta_centers[::-1]])
        H_full = np.concatenate([probe_function, probe_function[:, ::-1]], axis=1)
        
        # Create mesh for polar plot
        Theta_mesh, R_mesh = np.meshgrid(theta_full, r_centers)
        
        # Set minimum value for log scale and smooth the data
        H_full = np.where(H_full > 0, H_full, 1e-3)
        
        # Apply some smoothing to reduce noise
        from scipy.ndimage import gaussian_filter
        H_smooth = gaussian_filter(H_full, sigma=1.0)
        
        print(f"Response function range: {np.min(H_smooth):.2f} - {np.max(H_smooth):.2f}")
        
        # Create polar plot
        pcm = axis.pcolormesh(Theta_mesh, R_mesh, H_smooth,
                             shading='auto', cmap='jet',
                             norm=colors.LogNorm(vmin=0.1, vmax=max(10, np.max(H_smooth))))
        
        # Add colorbar
        cbar = figure.colorbar(pcm, ax=axis, shrink=0.8, pad=0.1)
        cbar.set_label('PE', fontsize=11)
        
        # Configure polar plot
        axis.set_ylim(0, 1)
        axis.set_theta_zero_location('N')
        axis.set_theta_direction(-1)
        axis.set_title('All-PMT Probe Function R(r,θ)', fontsize=14, fontweight='bold', pad=20)
        
        # Add radial grid
        axis.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0])
        axis.set_thetagrids(np.arange(0, 360, 45))
        
        # Add statistics
        stats_text = f'Total PMTs: {len(self.pmt_channels)}\n'
        stats_text += f'Data points: {len(r_values):,}\n'
        stats_text += f'PE range: {np.min(H_smooth):.1f} - {np.max(H_smooth):.1f}\n'
        stats_text += f'Mean PE/PMT: {np.mean(H_smooth):.2f}'
        
        axis.text(0.02, 0.98, stats_text, transform=axis.transAxes, 
                 verticalalignment='top', fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))
        
    def close_files(self):
        """Clean up"""
        self.sim_data.close()
        self.geo_data.close()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="All-PMT Probe Function Visualization")
    parser.add_argument("input_file", help="Input simulation file")
    parser.add_argument("-g", "--geometry", required=True, help="Geometry file")
    parser.add_argument("-o", "--output", required=True, help="Output PDF file")
    
    args = parser.parse_args()
    
    try:
        probe = AllPMTProbeFunction(args.input_file, args.geometry)
        
        with PdfPages(args.output) as pdf:
            # Page 1: Vertex density
            print("Creating vertex density plot...")
            fig1 = plt.figure(figsize=(10, 8))
            ax1 = fig1.add_subplot(1, 1, 1)
            probe.create_vertex_density_plot(fig1, ax1)
            pdf.savefig(fig1, bbox_inches='tight', dpi=300)
            plt.close(fig1)
            
            # Page 2: PE time histogram
            print("Creating PE time histogram...")
            fig2 = plt.figure(figsize=(10, 8))
            ax2 = fig2.add_subplot(1, 1, 1)
            probe.create_pe_time_plot(fig2, ax2)
            pdf.savefig(fig2, bbox_inches='tight', dpi=300)
            plt.close(fig2)
            
            # Page 3: All-PMT Probe function
            print("Creating All-PMT Probe function plot...")
            fig3 = plt.figure(figsize=(12, 10))
            ax3 = fig3.add_subplot(1, 1, 1, projection='polar')
            probe.create_probe_function_plot(fig3, ax3)
            pdf.savefig(fig3, bbox_inches='tight', dpi=300)
            plt.close(fig3)
        
        probe.close_files()
        print(f"All-PMT Probe function visualization complete! Saved to {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
