#!/usr/bin/env python3
# filepath: /home/tom/Desktop/magnetism/MagnetismB/simulate.py

import numpy as np
import matplotlib
# Set backend for display - try different backends if one doesn't work
try:
    import matplotlib.pyplot as plt
    # Check if we have a GUI backend available
    if matplotlib.get_backend() == 'Agg':
        # If we're on a headless system, try to set a GUI backend
        try:
            matplotlib.use('Qt5Agg')
        except:
            try:
                matplotlib.use('TkAgg')
            except:
                try:
                    matplotlib.use('GTK3Agg')
                except:
                    print("Warning: No GUI backend available. Plots will be saved but may not display.")
                    matplotlib.use('Agg')
    
    print(f"Using matplotlib backend: {matplotlib.get_backend()}")
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Error importing matplotlib: {e}")
    print("Please install matplotlib: pip install matplotlib")
    exit(1)

import pandas as pd
import os
from scipy.integrate import odeint
import time

def show_plot_with_message(plot_title="Plot"):
    """Enhanced plot display function with debug info"""
    try:
        print(f"\n📊 Displaying {plot_title}...")
        print("   (Close the plot window to continue)")
        plt.show(block=False)  # Non-blocking show
        plt.pause(0.1)  # Small pause to ensure plot appears
        input("   Press Enter after viewing the plot to continue...")
        plt.close('all')  # Close all plots
    except Exception as e:
        print(f"   Error displaying plot: {e}")
        print("   Plot has been saved to file instead.")

class MagnetThroughSpringSimulation:
    """
    Simulates a magnetic dipole falling through a helical spring coil.
    The magnet falls horizontally through the center of the spring, inducing EMF.
    """
    
    def __init__(self, magnetic_dipole_moment=2.0, spring_radius=0.02, spring_length=1.0, 
                 num_turns=50, wire_radius=0.001, resistivity=1.7e-8, 
                 magnet_mass=0.01, gravity=9.81):
        """
        Initialize the simulation parameters.
        
        Args:
            magnetic_dipole_moment: Magnetic dipole moment in A·m²
            spring_radius: Radius of the spring coil in meters
            spring_length: Length of the spring in meters
            num_turns: Number of turns in the spring
            wire_radius: Radius of the wire forming the spring
            resistivity: Electrical resistivity of the wire material (Ω·m)
            magnet_mass: Mass of the magnet in kg
            gravity: Gravitational acceleration in m/s²
        """
        self.m = magnetic_dipole_moment  # A·m²
        self.R = spring_radius  # m
        self.L = spring_length  # m
        self.N = num_turns  # number of turns
        self.r_wire = wire_radius  # m
        self.rho = resistivity  # Ω·m (copper)
        self.mass = magnet_mass  # kg
        self.g = gravity  # m/s²
        
        # Physical constants
        self.mu_0 = 4 * np.pi * 1e-7  # H/m (permeability of free space)
        
        # Calculate spring properties
        self.pitch = self.L / self.N  # distance between adjacent turns
        self.wire_length = self.N * np.sqrt((2 * np.pi * self.R)**2 + self.pitch**2)
        self.wire_cross_section = np.pi * self.r_wire**2
        self.resistance = self.rho * self.wire_length / self.wire_cross_section
        
        print(f"Spring parameters:")
        print(f"  Radius: {self.R*1000:.1f} mm")
        print(f"  Length: {self.L*1000:.1f} mm")
        print(f"  Number of turns: {self.N}")
        print(f"  Wire length: {self.wire_length*1000:.1f} mm")
        print(f"  Resistance: {self.resistance*1000:.2f} mΩ")
        print(f"  Magnetic dipole moment: {self.m} A·m²")
        
    def magnetic_field_dipole(self, x, y, z):
        """
        Calculate the magnetic field of a magnetic dipole at position (x, y, z).
        The dipole is oriented along the z-axis and located at the origin.
        
        Args:
            x, y, z: Position coordinates relative to the dipole
            
        Returns:
            Bx, By, Bz: Magnetic field components
        """
        r = np.sqrt(x**2 + y**2 + z**2)
        
        # Avoid division by zero
        r = np.maximum(r, 1e-10)
        
        # Magnetic field components for a dipole along z-axis
        factor = self.mu_0 * self.m / (4 * np.pi * r**5)
        
        Bx = factor * 3 * x * z
        By = factor * 3 * y * z
        Bz = factor * (2 * z**2 - x**2 - y**2)
        
        return Bx, By, Bz
    
    def magnetic_flux_through_turn(self, z_magnet, turn_number):
        """
        Calculate the magnetic flux through a single turn of the spring.
        
        Args:
            z_magnet: Position of the magnet along the spring axis
            turn_number: Which turn of the spring (0 to N-1)
            
        Returns:
            Magnetic flux through the turn
        """
        # Position of this turn along the spring
        z_turn = turn_number * self.pitch - self.L/2
        
        # Distance from magnet to this turn
        dz = z_turn - z_magnet
        
        # For a circular turn, we need to integrate over the circumference
        # Using analytical approximation for a circular loop
        # For a dipole field, the flux through a circular loop can be approximated
        
        r_distance = np.sqrt(self.R**2 + dz**2)
        
        # Analytical approximation for flux through circular loop from magnetic dipole
        # This is derived from the dipole field equations
        if abs(dz) < 1e-10:  # Magnet at the plane of the loop
            # Special case: magnet at the center of the loop
            flux = self.mu_0 * self.m / (2 * (self.R**2 + dz**2)**(3/2))
        else:
            # General case
            flux = self.mu_0 * self.m * self.R**2 / (2 * (self.R**2 + dz**2)**(3/2))
        
        return flux
    
    def total_magnetic_flux(self, z_magnet):
        """
        Calculate the total magnetic flux through all turns of the spring.
        Optimized version using vectorized calculations.
        
        Args:
            z_magnet: Position of the magnet along the spring axis
            
        Returns:
            Total magnetic flux through the spring
        """
        # Create array of turn positions (vectorized calculation)
        turn_numbers = np.arange(self.N)
        z_turns = turn_numbers * self.pitch - self.L/2
        
        # Distance from magnet to each turn
        dz_array = z_turns - z_magnet
        
        # Vectorized flux calculation for all turns
        # Handle special case for very small distances
        small_distance_mask = np.abs(dz_array) < 1e-10
        
        # Initialize flux array
        flux_array = np.zeros_like(dz_array)
        
        # General case calculation
        normal_mask = ~small_distance_mask
        if np.any(normal_mask):
            flux_array[normal_mask] = (self.mu_0 * self.m * self.R**2 / 
                                     (2 * (self.R**2 + dz_array[normal_mask]**2)**(3/2)))
        
        # Special case for magnet at the plane of the loop
        if np.any(small_distance_mask):
            flux_array[small_distance_mask] = (self.mu_0 * self.m / 
                                             (2 * (self.R**2 + dz_array[small_distance_mask]**2)**(3/2)))
        
        # Sum all fluxes
        total_flux = np.sum(flux_array)
        
        return total_flux
    
    def induced_emf(self, z_magnet, velocity):
        """
        Calculate the induced EMF using Faraday's law: EMF = -dΦ/dt
        
        Args:
            z_magnet: Position of the magnet
            velocity: Velocity of the magnet
            
        Returns:
            Induced EMF
        """
        # Calculate flux at slightly different positions to get derivative
        dz = 1e-6  # Small displacement for numerical derivative
        flux_plus = self.total_magnetic_flux(z_magnet + dz)
        flux_minus = self.total_magnetic_flux(z_magnet - dz)
        
        # Numerical derivative: dΦ/dz
        dflux_dz = (flux_plus - flux_minus) / (2 * dz)
        
        # EMF = -dΦ/dt = -dΦ/dz * dz/dt = -dΦ/dz * velocity
        emf = -dflux_dz * velocity
        
        return emf
    
    def equation_of_motion(self, state, t):
        """
        Differential equation for the motion of the magnet.
        State vector: [position, velocity]
        
        Args:
            state: [z_position, z_velocity]
            t: time
            
        Returns:
            [velocity, acceleration]
        """
        z_pos, z_vel = state
        
        # Calculate induced EMF and current
        emf = self.induced_emf(z_pos, z_vel)
        current = emf / self.resistance
        
        # Magnetic force on the dipole (simplified)
        # The force opposes motion (Lenz's law)
        # F = -μ₀ * m * I * dB/dz (simplified approximation)
        
        # For this simulation, we'll include both gravity and magnetic damping
        # Magnetic damping force proportional to velocity and magnetic field gradient
        
        # Calculate field gradient at magnet position
        dz = 1e-6
        _, _, Bz_plus = self.magnetic_field_dipole(0, 0, dz)
        _, _, Bz_minus = self.magnetic_field_dipole(0, 0, -dz)
        dBz_dz = (Bz_plus - Bz_minus) / (2 * dz)
        
        # Magnetic force (damping effect)
        magnetic_force = -self.mu_0 * self.m * current * dBz_dz
        
        # Total acceleration
        acceleration = self.g + magnetic_force / self.mass
        
        return [z_vel, acceleration]
    
    def run_simulation(self, duration=1.0, dt=0.0001):
        """
        Run the complete simulation.
        
        Args:
            duration: Simulation duration in seconds
            dt: Time step in seconds
            
        Returns:
            Dictionary containing time series data
        """
        # Time array
        t = np.arange(0, duration, dt)
        
        # Initial conditions: magnet starts at top of spring with zero velocity
        initial_position = -self.L/2 - 0.05  # Start slightly above the spring
        initial_velocity = 0.0
        initial_state = [initial_position, initial_velocity]
        
        # Solve differential equation
        solution = odeint(self.equation_of_motion, initial_state, t)
        positions = solution[:, 0]
        velocities = solution[:, 1]
        
        # Calculate EMF and current for each time step
        emf_values = []
        current_values = []
        flux_values = []
        
        for i, (pos, vel) in enumerate(zip(positions, velocities)):
            emf = self.induced_emf(pos, vel)
            current = emf / self.resistance
            flux = self.total_magnetic_flux(pos)
            
            emf_values.append(emf)
            current_values.append(current)
            flux_values.append(flux)
        
        # Convert to numpy arrays
        emf_values = np.array(emf_values)
        current_values = np.array(current_values)
        flux_values = np.array(flux_values)
        
        # Create results dictionary
        results = {
            'time': t,
            'position': positions,
            'velocity': velocities,
            'emf': emf_values,
            'current': current_values,
            'flux': flux_values,
            'resistance': self.resistance
        }
        
        return results
    
    def plot_results(self, results, save_plots=True):
        """
        Plot only EMF vs Time - simplified version.
        
        Args:
            results: Dictionary containing simulation results
            save_plots: Whether to save plots to files
        """
        # Create plots folder
        plots_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
        os.makedirs(plots_folder, exist_ok=True)
        
        # Extract results
        time = results['time']
        position = results['position']
        emf = results['emf']
        
        # Find EMF extrema
        max_emf_idx = np.argmax(np.abs(emf))
        min_emf_idx = np.argmin(emf)
        
        print(f"EMF extrema:")
        print(f"  Maximum EMF: {emf[max_emf_idx]*1000:.2f} mV at t = {time[max_emf_idx]:.4f} s")
        print(f"  Minimum EMF: {emf[min_emf_idx]*1000:.2f} mV at t = {time[min_emf_idx]:.4f} s")
        print(f"  Peak-to-peak EMF: {(emf[max_emf_idx] - emf[min_emf_idx])*1000:.2f} mV")
        
        # Find the time window when magnet is interacting with spring
        spring_top = self.L/2
        spring_bottom = -self.L/2
        spring_buffer = 0.1  # 10 cm buffer around spring
        
        # Define interaction region
        enter_region = spring_top + spring_buffer
        exit_region = spring_bottom - spring_buffer
        
        # Find indices where magnet is in interaction region
        in_region_mask = (position <= enter_region) & (position >= exit_region)
        interaction_indices = np.where(in_region_mask)[0]
        
        if len(interaction_indices) > 0:
            # Add time buffer (e.g., 0.05 seconds before and after)
            time_buffer = 0.05  # seconds
            start_time_idx = max(0, interaction_indices[0] - int(time_buffer / (results['time'][1] - results['time'][0])))
            end_time_idx = min(len(results['time']) - 1, interaction_indices[-1] + int(time_buffer / (results['time'][1] - results['time'][0])))
            
            # Create focused time window
            time_focused = results['time'][start_time_idx:end_time_idx+1]
            emf_focused = results['emf'][start_time_idx:end_time_idx+1]
            
            print(f"Focusing plot on spring interaction period:")
            print(f"  Time range: {time_focused[0]:.3f} - {time_focused[-1]:.3f} s")
            print(f"  Duration: {time_focused[-1] - time_focused[0]:.3f} s")
        else:
            # Fallback to full time series if no interaction detected
            time_focused = results['time']
            emf_focused = results['emf']
        
        # Create EMF vs Time plot only
        plt.figure(figsize=(12, 8))
        plt.plot(time_focused, emf_focused * 1000, 'r-', linewidth=2, label='Induced EMF')
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Induced EMF (mV)', fontsize=12)
        plt.title('Induced EMF vs Time - Magnet Through Spring', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        if save_plots:
            plt.savefig(os.path.join(plots_folder, 'emf_vs_time.png'), dpi=300, bbox_inches='tight')
        
        show_plot_with_message("EMF vs Time plot")
    
    def save_data(self, results, filename='simulation_data.csv'):
        """
        Save simulation results to CSV file.
        
        Args:
            results: Dictionary containing simulation results
            filename: Name of the output CSV file
        """
        # Create data folder if it doesn't exist
        data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        os.makedirs(data_folder, exist_ok=True)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Time(s)': results['time'],
            'Position(m)': results['position'],
            'Velocity(m/s)': results['velocity'],
            'Voltage I/O-1(V)': results['emf'],  # Using same column name as existing data
            'Current(A)': results['current'],
            'Flux(Wb)': results['flux']
        })
        
        # Save to CSV
        filepath = os.path.join(data_folder, filename)
        df.to_csv(filepath, index=False)
        print(f"Simulation data saved to: {filepath}")
        
        return filepath


def run_magnet_simulation():
    """
    Main function to run the magnet through spring simulation.
    """
    print("Starting Magnet Through Spring Simulation")
    print("=" * 50)
    
    # Create simulation instance with specified parameters
    sim = MagnetThroughSpringSimulation(
        magnetic_dipole_moment=2.0,  # A·m² as requested
        spring_radius=0.02,          # 2 cm radius
        spring_length=0.3,           # 30 cm length
        num_turns=50,                # 50 turns
        wire_radius=0.001,           # 1 mm wire radius
        resistivity=1.7e-8,          # Copper resistivity
        magnet_mass=0.01,            # 10 g magnet
        gravity=9.81                 # Earth gravity
    )
    
    print("\nRunning simulation...")
    
    # Run simulation for 1 second with high resolution
    results = sim.run_simulation(duration=1.0, dt=0.0001)
    
    print("Simulation completed!")
    
    # Find positions of maximum and minimum EMF
    max_emf_idx = np.argmax(results['emf'])
    min_emf_idx = np.argmin(results['emf'])
    
    max_emf_position = results['position'][max_emf_idx]
    min_emf_position = results['position'][min_emf_idx]
    max_emf_time = results['time'][max_emf_idx]
    min_emf_time = results['time'][min_emf_idx]
    
    # Calculate positions relative to spring
    spring_top = sim.L/2
    spring_bottom = -sim.L/2
    spring_center = 0
    
    # Position relative to spring top (positive means below top)
    max_emf_from_top = spring_top - max_emf_position
    min_emf_from_top = spring_top - min_emf_position
    
    # Position as fraction of spring length from top
    max_emf_fraction_from_top = max_emf_from_top / sim.L
    min_emf_fraction_from_top = min_emf_from_top / sim.L
    
    # Print key results
    max_emf = np.max(np.abs(results['emf']))
    max_current = np.max(np.abs(results['current']))
    
    print(f"\nKey Results:")
    print(f"  Maximum EMF: {results['emf'][max_emf_idx]*1000:.2f} mV")
    print(f"  Minimum EMF: {results['emf'][min_emf_idx]*1000:.2f} mV")
    print(f"  Maximum current: {max_current*1000:.2f} mA")
    print(f"  Spring resistance: {results['resistance']*1000:.2f} mΩ")
    
    print(f"\nSpring Geometry:")
    print(f"  Spring length: {sim.L*1000:.1f} mm")
    print(f"  Spring top position: {spring_top*1000:.1f} mm")
    print(f"  Spring bottom position: {spring_bottom*1000:.1f} mm")
    print(f"  Spring center position: {spring_center*1000:.1f} mm")
    
    print(f"\nEMF Extrema Positions:")
    print(f"  Maximum EMF occurs at:")
    print(f"    - Absolute position: {max_emf_position*1000:.1f} mm")
    print(f"    - Distance from spring top: {max_emf_from_top*1000:.1f} mm")
    print(f"    - Fraction of spring length from top: {max_emf_fraction_from_top:.3f}")
    print(f"    - Time: {max_emf_time:.4f} s")
    
    print(f"  Minimum EMF occurs at:")
    print(f"    - Absolute position: {min_emf_position*1000:.1f} mm")
    print(f"    - Distance from spring top: {min_emf_from_top*1000:.1f} mm")
    print(f"    - Fraction of spring length from top: {min_emf_fraction_from_top:.3f}")
    print(f"    - Time: {min_emf_time:.4f} s")
    
    # Additional analysis: check if extrema occur within the spring
    if spring_bottom <= max_emf_position <= spring_top:
        print(f"  ✓ Maximum EMF occurs within the spring")
    else:
        print(f"  ✗ Maximum EMF occurs outside the spring")
        
    if spring_bottom <= min_emf_position <= spring_top:
        print(f"  ✓ Minimum EMF occurs within the spring")
    else:
        print(f"  ✗ Minimum EMF occurs outside the spring")
    
    # Create plots
    print("\nGenerating plots...")
    sim.plot_results(results)
    
    # Save data
    print("\nSaving data...")
    sim.save_data(results, 'magnet_falling_simulation.csv')
    
    print("\nSimulation complete! Check the plots and data folders for results.")
    
    return results


class SegmentedCoilSimulation(MagnetThroughSpringSimulation):
    """
    Simulates a segmented coil system: 5 identical smaller coils connected by straight wire segments.
    This allows comparison with continuous coils to study the effect of coil segmentation.
    """
    
    def __init__(self, magnetic_dipole_moment=2.0, total_length=1.0, spring_radius=0.02, 
                 num_segments=5, gap_length=0.02, turns_per_segment=20, wire_radius=0.001, 
                 resistivity=1.7e-8, magnet_mass=0.01, gravity=9.81):
        """
        Initialize the segmented coil simulation.
        """
        self.m = magnetic_dipole_moment  # A·m²
        self.total_length = total_length  # m
        self.R = spring_radius  # m
        self.num_segments = num_segments  # number of coil segments
        self.gap_length = gap_length  # m
        self.turns_per_segment = turns_per_segment  # turns per segment
        self.r_wire = wire_radius  # m
        self.rho = resistivity  # Ω·m
        self.mass = magnet_mass  # kg
        self.g = gravity  # m/s²
        
        # Physical constants
        self.mu_0 = 4 * np.pi * 1e-7  # H/m
        
        # For compatibility with parent class methods
        self.L = self.total_length  # Spring length equivalent
        self.N = num_segments * turns_per_segment  # Total number of turns
        
        # Calculate coil geometry
        total_gap_length = (self.num_segments - 1) * self.gap_length
        available_coil_length = self.total_length - total_gap_length
        self.segment_length = available_coil_length / self.num_segments
        
        # Calculate segment properties
        self.pitch_per_segment = self.segment_length / self.turns_per_segment
        
        # Calculate positions of each segment center
        self.segment_centers = []
        start_pos = -self.total_length/2 + self.segment_length/2
        
        for i in range(self.num_segments):
            center_pos = start_pos + i * (self.segment_length + self.gap_length)
            self.segment_centers.append(center_pos)
        
        # Calculate total wire length and resistance
        # Coiled wire length per segment
        coil_wire_per_segment = self.turns_per_segment * np.sqrt((2 * np.pi * self.R)**2 + self.pitch_per_segment**2)
        total_coil_wire = self.num_segments * coil_wire_per_segment
        
        # Straight connecting wire length
        straight_wire_length = (self.num_segments - 1) * self.gap_length
        
        # Total wire length
        self.total_wire_length = total_coil_wire + straight_wire_length
        
        # Total resistance
        wire_cross_section = np.pi * self.r_wire**2
        self.resistance = self.rho * self.total_wire_length / wire_cross_section
        
        print(f"Segmented Coil Parameters:")
        print(f"  Total length: {self.total_length*1000:.1f} mm")
        print(f"  Number of segments: {self.num_segments}")
        print(f"  Segment length: {self.segment_length*1000:.1f} mm")
        print(f"  Gap length: {self.gap_length*1000:.1f} mm")
        print(f"  Turns per segment: {self.turns_per_segment}")
        print(f"  Total turns: {self.num_segments * self.turns_per_segment}")
        print(f"  Total wire length: {self.total_wire_length*1000:.1f} mm")
        print(f"  Total resistance: {self.resistance*1000:.2f} mΩ")
    
    def total_magnetic_flux(self, z_magnet):
        """
        Calculate the total magnetic flux through all coil segments.
        Straight wire segments contribute negligible flux.
        """
        total_flux = 0
        
        for i in range(self.num_segments):
            segment_center = self.segment_centers[i]
            
            # Calculate flux through each turn in this segment
            for turn in range(self.turns_per_segment):
                # Position of this turn within the segment
                turn_pos_in_segment = turn * self.pitch_per_segment - self.segment_length/2
                z_turn = segment_center + turn_pos_in_segment
                
                # Distance from magnet to this turn
                dz = z_turn - z_magnet
                
                # Flux calculation
                if abs(dz) < 1e-10:
                    flux = self.mu_0 * self.m / (2 * (self.R**2 + dz**2)**(3/2))
                else:
                    flux = self.mu_0 * self.m * self.R**2 / (2 * (self.R**2 + dz**2)**(3/2))
                
                total_flux += flux
        
        return total_flux
    
    def get_segment_boundaries(self):
        """Get the start and end positions of each coil segment."""
        boundaries = []
        for center in self.segment_centers:
            start = center - self.segment_length/2
            end = center + self.segment_length/2
            boundaries.append((start, end))
        return boundaries


def analyze_coil_length_effects():
    """
    Analyze how coil length affects electromagnetic induction results.
    Loops over different spring lengths and plots key parameters vs length.
    """
    print("Starting Coil Length Analysis")
    print("=" * 50)
    
    # Range of spring lengths to analyze (0.1m to 10m)
    spring_lengths = np.logspace(-1, 1, 20)  # 20 points from 0.1 to 10 m
    
    # Keep turn density constant instead of total turns
    # This is more physically meaningful
    turn_density = 50 / 0.3  # turns per meter (from original 50 turns in 0.3m)
    
    # Arrays to store results
    max_emf_values = []
    min_emf_values = []
    peak_to_peak_emf = []
    interaction_times = []
    max_emf_positions = []
    min_emf_positions = []
    total_resistance = []
    max_current_values = []
    energy_dissipated = []
    
    print(f"Analyzing {len(spring_lengths)} different coil lengths...")
    print(f"Turn density: {turn_density:.1f} turns/m")
    
    for i, length in enumerate(spring_lengths):
        print(f"\nProgress: {i+1}/{len(spring_lengths)} - Length: {length:.3f} m", end=" ")
        
        # Calculate number of turns for this length (maintaining turn density)
        num_turns = int(turn_density * length)
        
        try:
            # Create simulation with current length
            sim = MagnetThroughSpringSimulation(
                magnetic_dipole_moment=2.0,
                spring_radius=0.02,
                spring_length=length,
                num_turns=num_turns,
                wire_radius=0.001,
                resistivity=1.7e-8,
                magnet_mass=0.01,
                gravity=9.81
            )
            
            # Run simulation (shorter duration for long coils to save time)
            duration = min(2.0, 0.5 + length/2)  # Adaptive duration
            results = sim.run_simulation(duration=duration, dt=0.0001)
            
            # Extract key parameters
            max_emf = np.max(results['emf'])
            min_emf = np.min(results['emf'])
            peak_to_peak = max_emf - min_emf
            
            max_emf_idx = np.argmax(results['emf'])
            min_emf_idx = np.argmin(results['emf'])
            
            max_emf_pos = results['position'][max_emf_idx]
            min_emf_pos = results['position'][min_emf_idx]
            
            # Calculate interaction time (time when |EMF| > 1% of peak)
            emf_threshold = 0.01 * max(abs(max_emf), abs(min_emf))
            significant_emf_mask = np.abs(results['emf']) > emf_threshold
            significant_indices = np.where(significant_emf_mask)[0]
            
            if len(significant_indices) > 0:
                interaction_time = results['time'][significant_indices[-1]] - results['time'][significant_indices[0]]
            else:
                interaction_time = 0
            
            # Calculate energy dissipated (I²R integrated over time)
            current_squared = (results['emf'] / sim.resistance) ** 2
            dt = results['time'][1] - results['time'][0]
            energy = np.sum(current_squared) * sim.resistance * dt
            
            max_current = np.max(np.abs(results['emf'] / sim.resistance))
            
            # Store results
            max_emf_values.append(max_emf)
            min_emf_values.append(min_emf)
            peak_to_peak_emf.append(peak_to_peak)
            interaction_times.append(interaction_time)
            max_emf_positions.append(max_emf_pos)
            min_emf_positions.append(min_emf_pos)
            total_resistance.append(sim.resistance)
            max_current_values.append(max_current)
            energy_dissipated.append(energy)
            
            print(f"✓ Max EMF: {max_emf*1000:.1f} mV")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            # Fill with NaN for failed simulations
            max_emf_values.append(np.nan)
            min_emf_values.append(np.nan)
            peak_to_peak_emf.append(np.nan)
            interaction_times.append(np.nan)
            max_emf_positions.append(np.nan)
            min_emf_positions.append(np.nan)
            total_resistance.append(np.nan)
            max_current_values.append(np.nan)
            energy_dissipated.append(np.nan)
    
    # Convert to numpy arrays
    spring_lengths = np.array(spring_lengths)
    max_emf_values = np.array(max_emf_values)
    min_emf_values = np.array(min_emf_values)
    peak_to_peak_emf = np.array(peak_to_peak_emf)
    interaction_times = np.array(interaction_times)
    max_emf_positions = np.array(max_emf_positions)
    min_emf_positions = np.array(min_emf_positions)
    total_resistance = np.array(total_resistance)
    max_current_values = np.array(max_current_values)
    energy_dissipated = np.array(energy_dissipated)
    
    print(f"\nAnalysis complete! Creating plots...")
    
    # Create comprehensive plots
    create_length_analysis_plots(
        spring_lengths, max_emf_values, min_emf_values, peak_to_peak_emf,
        interaction_times, max_emf_positions, min_emf_positions,
        total_resistance, max_current_values, energy_dissipated, turn_density
    )
    
    # Save results to CSV
    save_length_analysis_data(
        spring_lengths, max_emf_values, min_emf_values, peak_to_peak_emf,
        interaction_times, max_emf_positions, min_emf_positions,
        total_resistance, max_current_values, energy_dissipated, turn_density
    )
    
    return {
        'spring_lengths': spring_lengths,
        'max_emf_values': max_emf_values,
        'min_emf_values': min_emf_values,
        'peak_to_peak_emf': peak_to_peak_emf,
        'interaction_times': interaction_times,
        'max_emf_positions': max_emf_positions,
        'min_emf_positions': min_emf_positions,
        'total_resistance': total_resistance,
        'max_current_values': max_current_values,
        'energy_dissipated': energy_dissipated,
        'turn_density': turn_density
    }


def create_length_analysis_plots(spring_lengths, max_emf_values, min_emf_values, 
                                peak_to_peak_emf, interaction_times, max_emf_positions, 
                                min_emf_positions, total_resistance, max_current_values, 
                                energy_dissipated, turn_density):
    """
    Create simplified EMF magnitude vs coil length plot only.
    """
    # Create plots folder
    plots_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
    os.makedirs(plots_folder, exist_ok=True)
    
    # Create single EMF magnitude plot
    plt.figure(figsize=(10, 8))
    plt.loglog(spring_lengths, np.abs(max_emf_values) * 1000, 'ro-', linewidth=3, markersize=8, label='Max EMF')
    plt.loglog(spring_lengths, np.abs(min_emf_values) * 1000, 'bo-', linewidth=3, markersize=8, label='|Min EMF|')
    plt.loglog(spring_lengths, peak_to_peak_emf * 1000, 'go-', linewidth=3, markersize=8, label='Peak-to-Peak EMF')
    plt.xlabel('Coil Length (m)', fontsize=14)
    plt.ylabel('EMF (mV)', fontsize=14)
    plt.title('EMF Magnitude vs Coil Length', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, 'coil_length_analysis.png'), dpi=300, bbox_inches='tight')
    show_plot_with_message("EMF vs Coil Length Analysis")


def save_length_analysis_data(spring_lengths, max_emf_values, min_emf_values, 
                             peak_to_peak_emf, interaction_times, max_emf_positions, 
                             min_emf_positions, total_resistance, max_current_values, 
                             energy_dissipated, turn_density):
    """
    Save the coil length analysis results to CSV file.
    """
    # Create data folder
    data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(data_folder, exist_ok=True)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Coil_Length_m': spring_lengths,
        'Number_of_Turns': spring_lengths * turn_density,
        'Max_EMF_V': max_emf_values,
        'Min_EMF_V': min_emf_values,
        'Peak_to_Peak_EMF_V': peak_to_peak_emf,
        'Interaction_Time_s': interaction_times,
        'Max_EMF_Position_m': max_emf_positions,
        'Min_EMF_Position_m': min_emf_positions,
        'Total_Resistance_Ohm': total_resistance,
        'Max_Current_A': max_current_values,
        'Energy_Dissipated_J': energy_dissipated,
        'Turn_Density_per_m': turn_density
    })
    
    # Save to CSV
    filepath = os.path.join(data_folder, 'coil_length_analysis.csv')
    df.to_csv(filepath, index=False)
    print(f"\nCoil length analysis data saved to: {filepath}")
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"  Coil length range: {spring_lengths.min():.3f} - {spring_lengths.max():.1f} m")
    print(f"  Max EMF range: {np.nanmin(max_emf_values)*1000:.2f} - {np.nanmax(max_emf_values)*1000:.2f} mV")
    print(f"  Peak-to-peak EMF range: {np.nanmin(peak_to_peak_emf)*1000:.2f} - {np.nanmax(peak_to_peak_emf)*1000:.2f} mV")
    print(f"  Interaction time range: {np.nanmin(interaction_times):.4f} - {np.nanmax(interaction_times):.4f} s")
    print(f"  Energy dissipation range: {np.nanmin(energy_dissipated)*1e6:.2f} - {np.nanmax(energy_dissipated)*1e6:.2f} μJ")
    
    return filepath


def compare_coil_lengths():
    """
    Compare EMF curves for different coil lengths by overlaying them on the same plots.
    This provides direct visual comparison of how coil length affects the EMF profile.
    """
    print("Starting Coil Length Comparison")
    print("=" * 50)
    
    # Define coil lengths to compare
    lengths_to_compare = [0.1, 1.0, 10.0]  # meters
    colors = ['red', 'blue', 'green']
    line_styles = ['-', '--', '-.']
    
    # Keep turn density constant
    turn_density = 50 / 0.3  # turns per meter (from original 50 turns in 0.3m)
    
    # Store results for each length
    all_results = {}
    all_sims = {}
    
    print(f"Turn density: {turn_density:.1f} turns/m")
    print(f"Comparing coil lengths: {lengths_to_compare} m")
    
    # Run simulations for each length
    for i, length in enumerate(lengths_to_compare):
        print(f"\nSimulating coil length: {length} m")
        
        # Calculate number of turns for this length
        num_turns = int(turn_density * length)
        print(f"  Number of turns: {num_turns}")
        
        # Adaptive parameters based on coil length
        if length <= 0.5:
            duration = 0.8
            dt = 0.0001  # High resolution for short coils
        elif length <= 2.0:
            duration = 1.5  
            dt = 0.0002  # Medium resolution
        else:
            duration = 2.0
            dt = 0.001   # Lower resolution for long coils to save time
        
        print(f"  Simulation duration: {duration} s")
        print(f"  Time step: {dt} s")
        
        # Create simulation
        sim = MagnetThroughSpringSimulation(
            magnetic_dipole_moment=2.0,
            spring_radius=0.02,
            spring_length=length,
            num_turns=num_turns,
            wire_radius=0.001,
            resistivity=1.7e-8,
            magnet_mass=0.01,
            gravity=9.81
        )
        
        # Run simulation with adaptive parameters
        results = sim.run_simulation(duration=duration, dt=dt)
        
        # Store results
        all_results[length] = results
        all_sims[length] = sim
        
        # Print key results
        max_emf = np.max(np.abs(results['emf']))
        print(f"  Maximum EMF: {max_emf*1000:.2f} mV")
        print(f"  Resistance: {sim.resistance*1000:.2f} mΩ")
        print(f"  ✓ Simulation completed successfully")
    
    print("\nGenerating comparison plots...")
    
    # Create overlay plots
    create_coil_length_overlay_plots(all_results, all_sims, lengths_to_compare, colors, line_styles)
    
    return all_results, all_sims


def create_coil_length_overlay_plots(all_results, all_sims, lengths_to_compare, colors, line_styles):
    """
    Create EMF vs Time overlay plot comparing different coil lengths.
    """
    # Create plots folder
    plots_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
    os.makedirs(plots_folder, exist_ok=True)
    
    # Create single EMF vs Time overlay plot
    plt.figure(figsize=(12, 8))
    
    for i, length in enumerate(lengths_to_compare):
        results = all_results[length]
        sim = all_sims[length]
        
        # Find focused time window (when magnet is near spring)
        spring_top = sim.L/2
        spring_bottom = -sim.L/2
        spring_buffer = 0.1  # 10 cm buffer around spring
        
        enter_region = spring_top + spring_buffer
        exit_region = spring_bottom - spring_buffer
        
        in_region_mask = (results['position'] <= enter_region) & (results['position'] >= exit_region)
        interaction_indices = np.where(in_region_mask)[0]
        
        if len(interaction_indices) > 0:
            time_buffer = 0.05
            dt = results['time'][1] - results['time'][0]
            start_idx = max(0, interaction_indices[0] - int(time_buffer / dt))
            end_idx = min(len(results['time']) - 1, interaction_indices[-1] + int(time_buffer / dt))
            
            time_focused = results['time'][start_idx:end_idx+1]
            emf_focused = results['emf'][start_idx:end_idx+1]
        else:
            time_focused = results['time']
            emf_focused = results['emf']
        
        time_shifted = time_focused - time_focused[0]
        
        plt.plot(time_shifted, emf_focused * 1000, color=colors[i], linestyle=line_styles[i], 
                linewidth=3, label=f'L = {length} m')
    
    plt.xlabel('Time from interaction start (s)', fontsize=14)
    plt.ylabel('Induced EMF (mV)', fontsize=14)
    plt.title('EMF vs Time - Coil Length Comparison', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, 'coil_length_comparison.png'), 
                dpi=300, bbox_inches='tight')
    show_plot_with_message("EMF vs Time - Coil Length Comparison")


def print_coil_comparison_summary(segmented_results, continuous_results,
                                segmented_sim, continuous_sim):
    """
    Print a detailed comparison summary.
    """
    print(f"\nDetailed Comparison Summary:")
    print("=" * 60)
    
    # EMF comparison
    seg_max_emf = np.max(segmented_results['emf']) * 1000
    seg_min_emf = np.min(segmented_results['emf']) * 1000
    seg_peak_to_peak = seg_max_emf - seg_min_emf
    
    cont_max_emf = np.max(continuous_results['emf']) * 1000
    cont_min_emf = np.min(continuous_results['emf']) * 1000
    cont_peak_to_peak = cont_max_emf - cont_min_emf
    
    print(f"EMF Comparison:")
    print(f"  {'Parameter':<20} {'Segmented':<12} {'Continuous':<12} {'Ratio':<10}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*10}")
    print(f"  {'Max EMF (mV)':<20} {seg_max_emf:<12.2f} {cont_max_emf:<12.2f} {seg_max_emf/cont_max_emf:<10.3f}")
    print(f"  {'Min EMF (mV)':<20} {seg_min_emf:<12.2f} {cont_min_emf:<12.2f} {seg_min_emf/cont_min_emf:<10.3f}")
    print(f"  {'Peak-to-Peak (mV)':<20} {seg_peak_to_peak:<12.2f} {cont_peak_to_peak:<12.2f} {seg_peak_to_peak/cont_peak_to_peak:<10.3f}")
    
    # Resistance comparison
    seg_resistance = segmented_sim.resistance * 1000
    cont_resistance = continuous_sim.resistance * 1000
    
    print(f"\nResistance Comparison:")
    print(f"  Segmented coil: {seg_resistance:.2f} mΩ")
    print(f"  Continuous coil: {cont_resistance:.2f} mΩ")
    print(f"  Ratio: {seg_resistance/cont_resistance:.3f}")
    
    # Power comparison
    seg_peak_power = seg_peak_to_peak**2 / seg_resistance * 1000  # μW
    cont_peak_power = cont_peak_to_peak**2 / cont_resistance * 1000
    
    print(f"\nPeak Power Comparison:")
    print(f"  Segmented coil: {seg_peak_power:.2f} μW")
    print(f"  Continuous coil: {cont_peak_power:.2f} μW")
    print(f"  Ratio: {seg_peak_power/cont_peak_power:.3f}")


def create_scientific_emf_time_plot(all_results, all_sims, gap_lengths):
    """
    Create a separate, scientific plot of EMF vs time for different gap lengths
    with proper LaTeX formatting and scientific notation.
    """
    plots_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
    os.makedirs(plots_folder, exist_ok=True)
    
    # Define colors for different gap lengths using a scientific color scheme
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(gap_lengths)))
    continuous_color = '#000000'  # Black for continuous
    
    # Create the scientific figure with publication-quality dimensions
    plt.figure(figsize=(10, 7.5), dpi=150, facecolor='white')  # Standard figure ratio
    
    # Set publication-quality font properties
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Liberation Serif', 'Times', 'serif'],
        'mathtext.fontset': 'dejavuserif',
        'axes.labelsize': 14,
        'axes.titlesize': 15,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11
    })
    
    # Plot segmented coils with different gap lengths first
    for i, gap_length in enumerate(gap_lengths):
        if gap_length in all_results and all_results[gap_length] is not None:
            results = all_results[gap_length]
            mask = (results['position'] >= -0.6) & (results['position'] <= 0.6)
            if np.any(mask):
                time_focused = results['time'][mask]
                emf_focused = results['emf'][mask]
                time_shifted = time_focused - time_focused[0]
                
                # Create label with just the number
                gap_mm = gap_length * 1000
                label = f'{gap_mm:.0f} mm'
                
                plt.plot(time_shifted, emf_focused, color=colors[i], 
                        linewidth=1.5, label=label, alpha=0.85)
    
    # Plot continuous coil last so it appears at the top of the legend
    if 'continuous' in all_results and all_results['continuous'] is not None:
        results = all_results['continuous']
        mask = (results['position'] >= -0.6) & (results['position'] <= 0.6)
        if np.any(mask):
            time_focused = results['time'][mask]
            emf_focused = results['emf'][mask]
            time_shifted = time_focused - time_focused[0]
            plt.plot(time_shifted, emf_focused, color=continuous_color, 
                    linewidth=1.75, label='Continuous', linestyle='-', alpha=0.9)
    
    # Scientific axis labels with LaTeX formatting
    plt.xlabel(r'$t$ [s]', fontsize=15)
    plt.ylabel(r'$\mathcal{E}$ [V]', fontsize=15)
    
    # Scientific title with LaTeX formatting for publication standard
    plt.title(r'Inter-Coil Gap Length vs. Induced EMF in a Stacked Coil System', fontsize=16, pad=20)
    
    # Enhanced grid for publication quality
    plt.grid(True, alpha=0.25, linestyle='-', linewidth=0.5, color='gray')
    
    # Scientific legend styling for publication standards
    legend = plt.legend(fontsize=11, loc='best', frameon=True, framealpha=0.7,
                       edgecolor='gray', fancybox=False, ncol=2)
    
    # Publication-quality tick formatting
    plt.tick_params(axis='both', which='major', direction='in', 
                   length=5, width=1.0, bottom=True, top=True, left=True, right=True)
    plt.tick_params(axis='both', which='minor', direction='in', 
                   length=3, width=0.5, bottom=True, top=True, left=True, right=True)
    
    # Add minor ticks
    plt.minorticks_on()
    
    # Add box around the plot for publication standard
    plt.box(True)
    
    # Adjust margins for publication layout
    plt.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.12)
    
    # Save with high DPI for publication quality
    filename = 'scientific_emf_vs_time_gap_analysis.png'
    filepath = os.path.join(plots_folder, filename)
    plt.savefig(filepath, dpi=400, bbox_inches='tight', facecolor='white', 
                edgecolor='none', format='png')
    
    # Also save as PDF for LaTeX documents
    pdf_filename = 'scientific_emf_vs_time_gap_analysis.pdf'
    pdf_filepath = os.path.join(plots_folder, pdf_filename)
    plt.savefig(pdf_filepath, bbox_inches='tight', facecolor='white', 
                edgecolor='none', format='pdf')
    
    # Also save as EPS for journal submissions
    eps_filename = 'scientific_emf_vs_time_gap_analysis.eps'
    eps_filepath = os.path.join(plots_folder, eps_filename)
    plt.savefig(eps_filepath, bbox_inches='tight', facecolor='white', 
                format='eps', dpi=600)
    
    plt.show()
    
    print(f"\nScientific EMF vs Time plot saved as:")
    print(f"  PNG: {filepath}")
    print(f"  PDF: {pdf_filepath}")
    print(f"  EPS: {eps_filepath} (for journal submission)")
    
    return filepath, pdf_filepath


def create_scientific_emf_position_plot(all_results, all_sims, gap_lengths):
    """
    Create a separate, scientific plot of EMF vs position for different gap lengths
    with proper LaTeX formatting.
    """
    plots_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
    os.makedirs(plots_folder, exist_ok=True)
    
    # Define colors for different gap lengths
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(gap_lengths)))
    continuous_color = '#000000'  # Black for continuous
    
    # Create the scientific figure
    plt.figure(figsize=(12, 8))
    
    # Plot segmented coils with different gap lengths first
    for i, gap_length in enumerate(gap_lengths):
        if gap_length in all_results and all_results[gap_length] is not None:
            results = all_results[gap_length]
            position_m = results['position']
            mask = (np.abs(position_m) <= 0.75)  # Focus on ±75cm
            
            # Create scientific label without "Gap ="
            gap_mm = gap_length * 1000
            label = f'{gap_mm:.0f} mm'
            
            plt.plot(position_m[mask], results['emf'][mask], color=colors[i], 
                    linewidth=2.5, label=label, alpha=0.8)
    
    # Plot continuous coil last so it appears at the top of the legend
    if 'continuous' in all_results and all_results['continuous'] is not None:
        results = all_results['continuous']
        position_m = results['position']
        mask = (np.abs(position_m) <= 0.75)  # Focus on ±75cm
        plt.plot(position_m[mask], results['emf'][mask], color=continuous_color, 
                linewidth=3, label='Continuous', linestyle='-', alpha=0.8)
    
    # Add segment boundaries for one representative segmented coil
    if gap_lengths and gap_lengths[len(gap_lengths)//2] in all_sims:
        representative_sim = all_sims[gap_lengths[len(gap_lengths)//2]]
        if hasattr(representative_sim, 'get_segment_boundaries'):
            boundaries = representative_sim.get_segment_boundaries()
            for i, (start, end) in enumerate(boundaries):
                plt.axvspan(start, end, alpha=0.1, color='gray', 
                           label='Coil segments' if i == 0 else "")
    
    # Scientific axis labels with LaTeX formatting - no bold
    plt.xlabel(r'$z$ [m]', fontsize=16)
    plt.ylabel(r'$\mathcal{E}$ [V]', fontsize=16)
    
    # Scientific title - no bold
    plt.title('Electromagnetic Induction: Spatial EMF Distribution', 
              fontsize=18, pad=20)
    
    # Enhanced grid
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    # Scientific legend styling
    legend = plt.legend(fontsize=12, loc='best', frameon=True, fancybox=True, 
                       shadow=True, ncol=2, columnspacing=1.0)
    legend.get_frame().set_facecolor('#f8f8f8')
    legend.get_frame().set_alpha(0.9)
    
    # Scientific tick formatting
    plt.tick_params(axis='both', which='major', labelsize=14, direction='in', 
                   length=6, width=1.2)
    plt.tick_params(axis='both', which='minor', labelsize=12, direction='in', 
                   length=3, width=0.8)
    
    # Add minor ticks
    plt.minorticks_on()
    
    # Tight layout for professional appearance
    plt.tight_layout()
    
    # Save with high DPI for publication quality
    filename = 'scientific_emf_vs_position_gap_analysis.png'
    filepath = os.path.join(plots_folder, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none')
    
    # Also save as PDF
    pdf_filename = 'scientific_emf_vs_position_gap_analysis.pdf'
    pdf_filepath = os.path.join(plots_folder, pdf_filename)
    plt.savefig(pdf_filepath, bbox_inches='tight', facecolor='white', 
                edgecolor='none')
    
    plt.show()
    
    print(f"\nScientific EMF vs Position plot saved as:")
    print(f"  PNG: {filepath}")
    print(f"  PDF: {pdf_filepath}")
    
    return filepath, pdf_filepath


def analyze_gap_length_effects():
    """
    Analyze how the gap length between coil segments affects electromagnetic induction.
    Varies gap length from 0.5 cm to 10 cm and compares EMF responses.
    """
    print("Starting Gap Length Analysis")
    print("=" * 50)
    
    # Define gap lengths to analyze (in meters)
    gap_lengths = [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1]  # 0.5cm to 10cm
    
    # Run all simulations
    all_results, all_sims = run_gap_length_simulations(gap_lengths)
    
    # Create scientific EMF vs time plot
    create_scientific_emf_time_plot(all_results, all_sims, gap_lengths)
    
    # Also analyze transit times through individual coils
    print("\n" + "=" * 50)
    print("Analyzing Transit Times Through Individual Coils")
    analyze_coil_transit_times(all_results, all_sims, gap_lengths)
    
    return all_results, all_sims


def create_gap_length_analysis_plots(all_results, all_sims, gap_lengths):
    """
    Create simplified EMF vs Time overlay plot for gap length comparison.
    """
    plots_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
    os.makedirs(plots_folder, exist_ok=True)
    
    # Define colors for different gap lengths
    colors = plt.cm.viridis(np.linspace(0, 1, len(gap_lengths)))
    continuous_color = 'red'
    
    # Create single EMF vs Time overlay plot
    plt.figure(figsize=(12, 8))
    
    # Plot continuous coil first
    if 'continuous' in all_results and all_results['continuous'] is not None:
        results = all_results['continuous']
        mask = (results['position'] >= -0.6) & (results['position'] <= 0.6)
        if np.any(mask):
            time_focused = results['time'][mask]
            emf_focused = results['emf'][mask]
            time_shifted = time_focused - time_focused[0]
            plt.plot(time_shifted, emf_focused * 1000, color=continuous_color, 
                    linewidth=3, label='Continuous', linestyle='-')
    
    # Plot segmented coils
    for i, gap_length in enumerate(gap_lengths):
        if gap_length in all_results and all_results[gap_length] is not None:
            results = all_results[gap_length]
            mask = (results['position'] >= -0.6) & (results['position'] <= 0.6)
            if np.any(mask):
                time_focused = results['time'][mask]
                emf_focused = results['emf'][mask]
                time_shifted = time_focused - time_focused[0]
                plt.plot(time_shifted, emf_focused * 1000, color=colors[i], 
                        linewidth=2, label=f'Gap: {gap_length*1000:.1f}mm')
    
    plt.xlabel('Time from interaction start (s)', fontsize=14)
    plt.ylabel('Induced EMF (mV)', fontsize=14)
    plt.title('EMF vs Time - Gap Length Comparison', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, 'gap_length_analysis.png'), 
                dpi=300, bbox_inches='tight')
    show_plot_with_message("EMF vs Time - Gap Length Comparison")
    
    # Print summary
    print(f"\nGap Length Analysis Summary:")
    gap_mm = [g * 1000 for g in gap_lengths]
    max_emfs = []
    peak_to_peak_emfs = []
    resistances = []
    peak_powers = []
    
    for gap_length in gap_lengths:
        if gap_length in all_results and all_results[gap_length] is not None:
            results = all_results[gap_length]
            sim = all_sims[gap_length]
            max_emf = np.max(results['emf']) * 1000
            min_emf = np.min(results['emf']) * 1000
            peak_to_peak = max_emf - min_emf
            power = np.max(results['emf']**2 / sim.resistance) * 1e6
            
            max_emfs.append(max_emf)
            peak_to_peak_emfs.append(peak_to_peak)
            resistances.append(sim.resistance * 1000)
            peak_powers.append(power)
        else:
            max_emfs.append(0)
            peak_to_peak_emfs.append(0)
            resistances.append(0)
            peak_powers.append(0)
    
    print(f"{'Gap (mm)':<8} {'Max EMF (mV)':<12} {'P2P EMF (mV)':<12} {'Resistance (mΩ)':<15} {'Peak Power (μW)':<15}")
    print("-" * 70)
    
    for i, gap_length in enumerate(gap_lengths):
        if gap_length in all_results and all_results[gap_length] is not None:
            gap_mm_val = gap_length * 1000
            max_emf = max_emfs[i]
            p2p_emf = peak_to_peak_emfs[i]
            resistance = resistances[i]
            power = peak_powers[i]
            
            print(f"{gap_mm_val:<8.1f} {max_emf:<12.2f} {p2p_emf:<12.2f} {resistance:<15.2f} {power:<15.2f}")
    
    # Continuous coil reference
    if 'continuous' in all_results and all_results['continuous'] is not None:
        cont_results = all_results['continuous']
        cont_sim = all_sims['continuous']
        cont_max = np.max(cont_results['emf']) * 1000
        cont_p2p = (np.max(cont_results['emf']) - np.min(cont_results['emf'])) * 1000
        cont_resistance = cont_sim.resistance * 1000
        cont_power = np.max(cont_results['emf']**2 / cont_sim.resistance) * 1e6
        
        print("-" * 70)
        print(f"{'Continuous':<8} {cont_max:<12.2f} {cont_p2p:<12.2f} {cont_resistance:<15.2f} {cont_power:<15.2f}")


def save_gap_length_analysis_data(all_results, all_sims, gap_lengths):
    """
    Save the gap length analysis results to CSV file.
    """
    data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(data_folder, exist_ok=True)
    
    # Prepare data for DataFrame
    data_rows = []
    
    # Add continuous coil data
    if 'continuous' in all_results and all_results['continuous'] is not None:
        cont_results = all_results['continuous']
        cont_sim = all_sims['continuous']
        
        max_emf = np.max(cont_results['emf'])
        min_emf = np.min(cont_results['emf'])
        peak_to_peak = max_emf - min_emf
        peak_power = np.max(cont_results['emf']**2 / cont_sim.resistance)
        
        data_rows.append({
            'Configuration': 'Continuous',
            'Gap_Length_mm': 0,
            'Num_Segments': 1,
            'Total_Turns': cont_sim.N,
            'Max_EMF_V': max_emf,
            'Min_EMF_V': min_emf,
            'Peak_to_Peak_EMF_V': peak_to_peak,
            'Resistance_Ohm': cont_sim.resistance,
            'Peak_Power_W': peak_power,
            'Total_Wire_Length_m': cont_sim.wire_length / 1000,
            'Effective_Coil_Length_m': cont_sim.L,
            'Total_Gap_Length_m': 0
        })
    
    # Add segmented coil data
    for gap_length in gap_lengths:
        if gap_length in all_results and all_results[gap_length] is not None:
            results = all_results[gap_length]
            sim = all_sims[gap_length]
            
            max_emf = np.max(results['emf'])
            min_emf = np.min(results['emf'])
            peak_to_peak = max_emf - min_emf
            peak_power = np.max(results['emf']**2 / sim.resistance)
            
            total_gap_length = (sim.num_segments - 1) * gap_length
            effective_coil_length = sim.segment_length * sim.num_segments
            
            data_rows.append({
                'Configuration': 'Segmented',
                'Gap_Length_mm': gap_length * 1000,
                'Num_Segments': sim.num_segments,
                'Total_Turns': sim.num_segments * sim.turns_per_segment,
                'Max_EMF_V': max_emf,
                'Min_EMF_V': min_emf,
                'Peak_to_Peak_EMF_V': peak_to_peak,
                'Resistance_Ohm': sim.resistance,
                'Peak_Power_W': peak_power,
                'Total_Wire_Length_m': sim.total_wire_length / 1000,
                'Effective_Coil_Length_m': effective_coil_length,
                'Total_Gap_Length_m': total_gap_length
            })
    
    # Create DataFrame and save
    df = pd.DataFrame(data_rows)
    filepath = os.path.join(data_folder, 'gap_length_analysis.csv')
    df.to_csv(filepath, index=False)
    
    print(f"\nGap length analysis data saved to: {filepath}")
    return filepath


def compare_segmented_vs_continuous_coils():
    """
    Compare EMF response between a segmented coil (5 segments) and a continuous coil.
    """
    print("Starting Segmented vs Continuous Coil Comparison")
    print("=" * 60)
    
    # Parameters
    total_length = 1.0  # 1 meter total
    total_turns = 100   # 100 turns total
    gap_length = 0.02   # 2 cm gaps
    
    # Create segmented coil
    print(f"\n1. Creating Segmented Coil (5 segments):")
    segmented_sim = SegmentedCoilSimulation(
        magnetic_dipole_moment=2.0,
        total_length=total_length,
        spring_radius=0.02,
        num_segments=5,
        gap_length=gap_length,
        turns_per_segment=total_turns // 5,
        wire_radius=0.001,
        resistivity=1.7e-8,
        magnet_mass=0.01,
        gravity=9.81
    )
    
    # Create continuous coil
    print(f"\n2. Creating Continuous Coil:")
    continuous_sim = MagnetThroughSpringSimulation(
        magnetic_dipole_moment=2.0,
        spring_radius=0.02,
        spring_length=total_length,
        num_turns=total_turns,
        wire_radius=0.001,
        resistivity=1.7e-8,
        magnet_mass=0.01,
        gravity=9.81
    )
    
    # Run simulations
    print(f"\n3. Running Simulations:")
    duration = 1.5
    dt = 0.0002
    
    print(f"  Running segmented coil simulation...")
    segmented_results = segmented_sim.run_simulation(duration=duration, dt=dt)
    
    print(f"  Running continuous coil simulation...")
    continuous_results = continuous_sim.run_simulation(duration=duration, dt=dt)
    
    # Create comparison plots
    print(f"\n4. Generating comparison plots...")
    create_segmented_comparison_plots(segmented_results, continuous_results, 
                                    segmented_sim, continuous_sim)
    
    return segmented_results, continuous_results, segmented_sim, continuous_sim


def create_segmented_comparison_plots(segmented_results, continuous_results, 
                                    segmented_sim, continuous_sim):
    """Create simplified EMF vs Time comparison plot between segmented and continuous coils."""
    plots_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
    os.makedirs(plots_folder, exist_ok=True)
    
    # Create single EMF vs Time comparison plot
    plt.figure(figsize=(12, 8))
    
    # Plot EMF vs Time for both configurations
    for results, label, color, linestyle in [
        (segmented_results, 'Segmented (5 coils)', 'red', '-'),
        (continuous_results, 'Continuous', 'blue', '--')
    ]:
        # Focus on interaction region
        mask = (results['position'] >= -0.6) & (results['position'] <= 0.6)
        if np.any(mask):
            time_focused = results['time'][mask]
            emf_focused = results['emf'][mask]
            time_shifted = time_focused - time_focused[0]
            plt.plot(time_shifted, emf_focused * 1000, color=color, linestyle=linestyle, 
                    linewidth=3, label=label)
    
    plt.xlabel('Time from interaction start (s)', fontsize=14)
    plt.ylabel('Induced EMF (mV)', fontsize=14)
    plt.title('EMF vs Time - Segmented vs Continuous Coils', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, 'segmented_vs_continuous.png'), 
                dpi=300, bbox_inches='tight')
    show_plot_with_message("EMF vs Time - Segmented vs Continuous Comparison")
    
    # Print summary
    seg_max = np.max(segmented_results['emf']) * 1000
    seg_min = np.min(segmented_results['emf']) * 1000
    cont_max = np.max(continuous_results['emf']) * 1000
    cont_min = np.min(continuous_results['emf']) * 1000
    
    print(f"\nComparison Summary:")
    print(f"Segmented coil - Max EMF: {seg_max:.2f} mV, Peak-to-Peak: {seg_max-seg_min:.2f} mV")
    print(f"Continuous coil - Max EMF: {cont_max:.2f} mV, Peak-to-Peak: {cont_max-cont_min:.2f} mV")
    print(f"EMF Ratio (Segmented/Continuous): {(seg_max-seg_min)/(cont_max-cont_min):.3f}")


def compare_coil_lengths():
    """
    Compare EMF curves for different coil lengths by overlaying them on the same plots.
    This provides direct visual comparison of how coil length affects the EMF profile.
    """
    print("Starting Coil Length Comparison")
    print("=" * 50)
    
    # Define coil lengths to compare
    lengths_to_compare = [0.1, 1.0, 10.0]  # meters
    colors = ['red', 'blue', 'green']
    line_styles = ['-', '--', '-.']
    
    # Keep turn density constant
    turn_density = 50 / 0.3  # turns per meter (from original 50 turns in 0.3m)
    
    # Store results for each length
    all_results = {}
    all_sims = {}
    
    print(f"Turn density: {turn_density:.1f} turns/m")
    print(f"Comparing coil lengths: {lengths_to_compare} m")
    
    # Run simulations for each length
    for i, length in enumerate(lengths_to_compare):
        print(f"\nSimulating coil length: {length} m")
        
        # Calculate number of turns for this length
        num_turns = int(turn_density * length)
        print(f"  Number of turns: {num_turns}")
        
        # Adaptive parameters based on coil length
        if length <= 0.5:
            duration = 0.8
            dt = 0.0001  # High resolution for short coils
        elif length <= 2.0:
            duration = 1.5  
            dt = 0.0002  # Medium resolution
        else:
            duration = 2.0
            dt = 0.001   # Lower resolution for long coils to save time
        
        print(f"  Simulation duration: {duration} s")
        print(f"  Time step: {dt} s")
        
        # Create simulation
        sim = MagnetThroughSpringSimulation(
            magnetic_dipole_moment=2.0,
            spring_radius=0.02,
            spring_length=length,
            num_turns=num_turns,
            wire_radius=0.001,
            resistivity=1.7e-8,
            magnet_mass=0.01,
            gravity=9.81
        )
        
        # Run simulation with adaptive parameters
        results = sim.run_simulation(duration=duration, dt=dt)
        
        # Store results
        all_results[length] = results
        all_sims[length] = sim
        
        # Print key results
        max_emf = np.max(np.abs(results['emf']))
        print(f"  Maximum EMF: {max_emf*1000:.2f} mV")
        print(f"  Resistance: {sim.resistance*1000:.2f} mΩ")
        print(f"  ✓ Simulation completed successfully")
    
    print("\nGenerating comparison plots...")
    
    # Create overlay plots
    create_coil_length_overlay_plots(all_results, all_sims, lengths_to_compare, colors, line_styles)
    
    return all_results, all_sims


def create_coil_length_overlay_plots(all_results, all_sims, lengths_to_compare, colors, line_styles):
    """
    Create EMF vs Time overlay plot comparing different coil lengths.
    """
    # Create plots folder
    plots_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
    os.makedirs(plots_folder, exist_ok=True)
    
    # Create single EMF vs Time overlay plot
    plt.figure(figsize=(12, 8))
    
    for i, length in enumerate(lengths_to_compare):
        results = all_results[length]
        sim = all_sims[length]
        
        # Find focused time window (when magnet is near spring)
        spring_top = sim.L/2
        spring_bottom = -sim.L/2
        spring_buffer = 0.1  # 10 cm buffer around spring
        
        enter_region = spring_top + spring_buffer
        exit_region = spring_bottom - spring_buffer
        
        in_region_mask = (results['position'] <= enter_region) & (results['position'] >= exit_region)
        interaction_indices = np.where(in_region_mask)[0]
        
        if len(interaction_indices) > 0:
            time_buffer = 0.05
            dt = results['time'][1] - results['time'][0]
            start_idx = max(0, interaction_indices[0] - int(time_buffer / dt))
            end_idx = min(len(results['time']) - 1, interaction_indices[-1] + int(time_buffer / dt))
            
            time_focused = results['time'][start_idx:end_idx+1]
            emf_focused = results['emf'][start_idx:end_idx+1]
        else:
            time_focused = results['time']
            emf_focused = results['emf']
        
        time_shifted = time_focused - time_focused[0]
        
        plt.plot(time_shifted, emf_focused * 1000, color=colors[i], linestyle=line_styles[i], 
                linewidth=3, label=f'L = {length} m')
    
    plt.xlabel('Time from interaction start (s)', fontsize=14)
    plt.ylabel('Induced EMF (mV)', fontsize=14)
    plt.title('EMF vs Time - Coil Length Comparison', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, 'coil_length_comparison.png'), 
                dpi=300, bbox_inches='tight')
    show_plot_with_message("EMF vs Time - Coil Length Comparison")


def print_coil_comparison_summary(segmented_results, continuous_results,
                                segmented_sim, continuous_sim):
    """
    Print a detailed comparison summary.
    """
    print(f"\nDetailed Comparison Summary:")
    print("=" * 60)
    
    # EMF comparison
    seg_max_emf = np.max(segmented_results['emf']) * 1000
    seg_min_emf = np.min(segmented_results['emf']) * 1000
    seg_peak_to_peak = seg_max_emf - seg_min_emf
    
    cont_max_emf = np.max(continuous_results['emf']) * 1000
    cont_min_emf = np.min(continuous_results['emf']) * 1000
    cont_peak_to_peak = cont_max_emf - cont_min_emf
    
    print(f"EMF Comparison:")
    print(f"  {'Parameter':<20} {'Segmented':<12} {'Continuous':<12} {'Ratio':<10}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*10}")
    print(f"  {'Max EMF (mV)':<20} {seg_max_emf:<12.2f} {cont_max_emf:<12.2f} {seg_max_emf/cont_max_emf:<10.3f}")
    print(f"  {'Min EMF (mV)':<20} {seg_min_emf:<12.2f} {cont_min_emf:<12.2f} {seg_min_emf/cont_min_emf:<10.3f}")
    print(f"  {'Peak-to-Peak (mV)':<20} {seg_peak_to_peak:<12.2f} {cont_peak_to_peak:<12.2f} {seg_peak_to_peak/cont_peak_to_peak:<10.3f}")
    
    # Resistance comparison
    seg_resistance = segmented_sim.resistance * 1000
    cont_resistance = continuous_sim.resistance * 1000
    
    print(f"\nResistance Comparison:")
    print(f"  Segmented coil: {seg_resistance:.2f} mΩ")
    print(f"  Continuous coil: {cont_resistance:.2f} mΩ")
    print(f"  Ratio: {seg_resistance/cont_resistance:.3f}")
    
    # Power comparison
    seg_peak_power = seg_peak_to_peak**2 / seg_resistance * 1000  # μW
    cont_peak_power = cont_peak_to_peak**2 / cont_resistance * 1000
    
    print(f"\nPeak Power Comparison:")
    print(f"  Segmented coil: {seg_peak_power:.2f} μW")
    print(f"  Continuous coil: {cont_peak_power:.2f} μW")
    print(f"  Ratio: {seg_peak_power/cont_peak_power:.3f}")


def show_emf_time_plot_only():
    """
    Generate and display the scientific EMF vs time plot for different gap lengths
    and analyze the transit time of the magnet through each coil.
    """
    print("Generating EMF vs Time Plot for Different Gap Lengths")
    print("=" * 55)
    
    # Define a smaller set of gap lengths for quick analysis
    gap_lengths = [0.01, 0.02, 0.05, 0.1]  # 1cm, 2cm, 5cm, 10cm
    
    # Fixed parameters
    total_length = 1.0  # 1 meter total
    num_segments = 5
    total_turns = 100
    turns_per_segment = total_turns // num_segments
    
    print(f"Analysis parameters:")
    print(f"  Total coil system length: {total_length} m")
    print(f"  Number of segments: {num_segments}")
    print(f"  Gap lengths: {[f'{g*1000:.0f}' for g in gap_lengths]} mm")
    
    # Store results for each gap length
    all_results = {}
    all_sims = {}
    
    # Create reference continuous coil
    print(f"\nCreating reference continuous coil...")
    continuous_sim = MagnetThroughSpringSimulation(
        magnetic_dipole_moment=2.0,
        spring_radius=0.02,
        spring_length=total_length,
        num_turns=total_turns,
        wire_radius=0.001,
        resistivity=1.7e-8,
        magnet_mass=0.01,
        gravity=9.81
    )
    
    duration = 1.5
    dt = 0.0002
    continuous_results = continuous_sim.run_simulation(duration=duration, dt=dt)
    
    # Store continuous results
    all_results['continuous'] = continuous_results
    all_sims['continuous'] = continuous_sim
    
    # Run simulations for each gap length
    for gap_length in gap_lengths:
        print(f"Simulating gap length: {gap_length*1000:.0f} mm...", end=" ")
        
        try:
            # Create segmented coil with current gap length
            segmented_sim = SegmentedCoilSimulation(
                magnetic_dipole_moment=2.0,
                total_length=total_length,
                spring_radius=0.02,
                num_segments=num_segments,
                gap_length=gap_length,
                turns_per_segment=turns_per_segment,
                wire_radius=0.001,
                resistivity=1.7e-8,
                magnet_mass=0.01,
                gravity=9.81
            )
            
            segmented_results = segmented_sim.run_simulation(duration=duration, dt=dt)
            
            # Store results
            all_results[gap_length] = segmented_results
            all_sims[gap_length] = segmented_sim
            
            max_emf = np.max(segmented_results['emf']) * 1000
            print(f"Max EMF: {max_emf:.1f} mV ✓")
            
        except Exception as e:
            print(f"Error: {e}")
            all_results[gap_length] = None
            all_sims[gap_length] = None
    
    # Generate the scientific EMF vs time plot
    print(f"\nGenerating scientific EMF vs time plot...")
    create_scientific_emf_time_plot(all_results, all_sims, gap_lengths)
    
    # Also analyze transit times through individual coils
    print("\n" + "=" * 50)
    print("Analyzing Transit Times Through Individual Coils")
    analyze_coil_transit_times(all_results, all_sims, gap_lengths)
    
    return all_results, all_sims


def analyze_coil_transit_times(all_results, all_sims, gap_lengths):
    """
    Analyze the time it takes for the magnet to pass through each individual coil
    in the segmented tower and create a plot of transit time vs coil number.
    """
    plots_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
    os.makedirs(plots_folder, exist_ok=True)
    
    # Choose a representative gap length for detailed analysis
    representative_gap = gap_lengths[len(gap_lengths)//2] if gap_lengths else 0.02
    
    if representative_gap not in all_results or all_results[representative_gap] is None:
        print("No valid segmented coil data available for transit time analysis")
        return
    
    results = all_results[representative_gap]
    sim = all_sims[representative_gap]
    
    # Get segment boundaries
    boundaries = sim.get_segment_boundaries()
    num_segments = len(boundaries)
    
    # Calculate transit times for each coil
    transit_times = []
    coil_numbers = []
    
    for i, (start_pos, end_pos) in enumerate(boundaries):
        # Find when magnet enters and exits this coil segment
        mask_in_coil = (results['position'] >= start_pos) & (results['position'] <= end_pos)
        
        if np.any(mask_in_coil):
            times_in_coil = results['time'][mask_in_coil]
            if len(times_in_coil) > 0:
                enter_time = times_in_coil[0]
                exit_time = times_in_coil[-1]
                transit_time = exit_time - enter_time
                
                transit_times.append(transit_time * 1000)  # Convert to milliseconds
                coil_numbers.append(i + 1)  # Start counting from 1
    
    if not transit_times:
        print("Could not calculate transit times for any coils")
        return
    
    # Create the transit time plot
    plt.figure(figsize=(10, 7.5), dpi=150, facecolor='white')
    
    # Set publication-quality font properties
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Liberation Serif', 'Times', 'serif'],
        'mathtext.fontset': 'dejavuserif',
        'axes.labelsize': 14,
        'axes.titlesize': 15,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11
    })
    
    # Plot transit times
    plt.plot(coil_numbers, transit_times, 'bo-', linewidth=2, markersize=8, 
             markerfacecolor='lightblue', markeredgecolor='blue', markeredgewidth=1.5,
             label=f'Gap = {representative_gap*1000:.0f} mm')
    
    # No trend line - removed as requested
    
    # Scientific axis labels
    plt.xlabel(r'Coil Number $i$', fontsize=15)
    plt.ylabel(r'Transit Time $t_i$ [ms]', fontsize=15)
    
    # Scientific title
    plt.title(r'Magnet Transit Time Through Individual Coils', fontsize=16, pad=20)
    
    # Enhanced grid for publication quality
    plt.grid(True, alpha=0.25, linestyle='-', linewidth=0.5, color='gray')
    
    # Set integer ticks for coil numbers
    plt.xticks(coil_numbers)
    
    # Scientific legend styling
    legend = plt.legend(fontsize=11, loc='best', frameon=True, framealpha=0.7,
                       edgecolor='gray', fancybox=False)
    
    # Publication-quality tick formatting
    plt.tick_params(axis='both', which='major', direction='in', 
                   length=5, width=1.0, bottom=True, top=True, left=True, right=True)
    plt.tick_params(axis='both', which='minor', direction='in', 
                   length=3, width=0.5, bottom=True, top=True, left=True, right=True)
    
    # Add minor ticks
    plt.minorticks_on()
    
    # Add box around the plot
    plt.box(True)
    
    # Adjust margins for publication layout
    plt.subplots_adjust(left=0.12, right=0.95, top=0.88, bottom=0.12)
    
    # Save with high DPI for publication quality
    filename = 'coil_transit_times.png'
    filepath = os.path.join(plots_folder, filename)
    plt.savefig(filepath, dpi=400, bbox_inches='tight', facecolor='white', 
                edgecolor='none', format='png')
    
    # Also save as PDF
    pdf_filename = 'coil_transit_times.pdf'
    pdf_filepath = os.path.join(plots_folder, pdf_filename)
    plt.savefig(pdf_filepath, bbox_inches='tight', facecolor='white', 
                edgecolor='none', format='pdf')
    
    plt.show()
    
    # Print summary statistics
    print(f"\nTransit Time Analysis Summary:")
    print(f"Gap length analyzed: {representative_gap*1000:.1f} mm")
    print(f"Number of coils: {len(transit_times)}")
    print(f"Average transit time: {np.mean(transit_times):.2f} ms")
    print(f"Standard deviation: {np.std(transit_times):.2f} ms")
    print(f"Transit time range: {np.min(transit_times):.2f} - {np.max(transit_times):.2f} ms")
    
    print(f"\nIndividual coil transit times:")
    for i, (coil_num, t_time) in enumerate(zip(coil_numbers, transit_times)):
        print(f"  Coil {coil_num}: {t_time:.2f} ms")
    
    print(f"\nPlot saved as:")
    print(f"  PNG: {filepath}")
    print(f"  PDF: {pdf_filepath}")
    
    return transit_times, coil_numbers


def run_gap_length_simulations(gap_lengths):
    """
    Run simulations for different gap lengths between coil segments and return results.
    """
    # Fixed parameters
    total_length = 1.0  # 1 meter total
    num_segments = 5
    total_turns = 100
    turns_per_segment = total_turns // num_segments
    
    print(f"Analysis parameters:")
    print(f"  Total coil system length: {total_length} m")
    print(f"  Number of segments: {num_segments}")
    print(f"  Total turns: {total_turns}")
    print(f"  Turns per segment: {turns_per_segment}")
    print(f"  Gap lengths to analyze: {[f'{g*1000:.1f}' for g in gap_lengths]} mm")
    
    # Store results for each gap length
    all_results = {}
    all_sims = {}
    
    # Also include continuous coil for reference
    print(f"\nCreating reference continuous coil:")
    continuous_sim = MagnetThroughSpringSimulation(
        magnetic_dipole_moment=2.0,
        spring_radius=0.02,
        spring_length=total_length,
        num_turns=total_turns,
        wire_radius=0.001,
        resistivity=1.7e-8,
        magnet_mass=0.01,
        gravity=9.81
    )
    
    duration = 1.5
    dt = 0.0002
    print(f"Running continuous coil simulation...")
    continuous_results = continuous_sim.run_simulation(duration=duration, dt=dt)
    
    # Store continuous results
    all_results['continuous'] = continuous_results
    all_sims['continuous'] = continuous_sim
    
    # Run simulations for each gap length
    for i, gap_length in enumerate(gap_lengths):
        print(f"\n--- Gap Length: {gap_length*1000:.1f} mm ---")
        
        try:
            # Create segmented coil with current gap length
            segmented_sim = SegmentedCoilSimulation(
                magnetic_dipole_moment=2.0,
                total_length=total_length,
                spring_radius=0.02,
                num_segments=num_segments,
                gap_length=gap_length,
                turns_per_segment=turns_per_segment,
                wire_radius=0.001,
                resistivity=1.7e-8,
                magnet_mass=0.01,
                gravity=9.81
            )
            
            print(f"Running simulation...")
            segmented_results = segmented_sim.run_simulation(duration=duration, dt=dt)
            
            # Store results
            all_results[gap_length] = segmented_results
            all_sims[gap_length] = segmented_sim
            
            # Print key results
            max_emf = np.max(segmented_results['emf']) * 1000
            min_emf = np.min(segmented_results['emf']) * 1000
            peak_to_peak = max_emf - min_emf
            
            print(f"  Max EMF: {max_emf:.2f} mV")
            print(f"  Peak-to-Peak EMF: {peak_to_peak:.2f} mV")
            print(f"  Resistance: {segmented_sim.resistance*1000:.2f} mΩ")
            print(f"  ✓ Simulation completed successfully")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            all_results[gap_length] = None
            all_sims[gap_length] = None
    
    return all_results, all_sims


def compare_initial_velocities():
    """
    Compare the effects of different initial velocities on electromagnetic induction.
    Run simulations with various initial velocities and create an EMF vs time overlay plot.
    """
    print("Starting Initial Velocity Comparison Analysis")
    print("=" * 60)
    
    # Define different initial velocities to compare (m/s)
    # Negative values represent downward velocities
    velocities = [-5.0, -2.5, -1.0, 0.0, 1.0, 2.5, 5.0]
    
    # Define colors using a colormap for visual distinction
    colors = plt.cm.viridis(np.linspace(0, 1, len(velocities)))
    
    # Create plots folder if it doesn't exist
    plots_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
    os.makedirs(plots_folder, exist_ok=True)
    
    # Store simulation results - use integer indices instead of floating point values
    all_results = []
    
    # Set up standard simulation parameters
    magnetic_dipole_moment = 2.0  # A·m²
    spring_radius = 0.02          # 2 cm
    spring_length = 0.3           # 30 cm
    num_turns = 50                # 50 turns
    wire_radius = 0.001           # 1 mm
    resistivity = 1.7e-8          # Copper
    magnet_mass = 0.01            # 10 g
    
    # For consistent comparison, set initial position above the coil
    initial_position = -spring_length/2 - 0.05  # 5cm above the spring
    
    print(f"Running simulations with different initial velocities...")
    
    # Run simulations for each initial velocity
    for i, velocity in enumerate(velocities):
        print(f"  Simulating initial velocity: {velocity:.1f} m/s...", end=" ")
        
        # Create simulation instance
        sim = MagnetThroughSpringSimulation(
            magnetic_dipole_moment=magnetic_dipole_moment,
            spring_radius=spring_radius,
            spring_length=spring_length,
            num_turns=num_turns,
            wire_radius=wire_radius,
            resistivity=resistivity,
            magnet_mass=magnet_mass,
            gravity=9.81
        )
        
        # Run simulation with this initial velocity
        # We need to modify the equation_of_motion method to support non-zero initial velocity
        # First, let's create a custom run_simulation function to handle different initial conditions
        
        # Time array
        duration = 1.0
        dt = 0.0001
        t = np.arange(0, duration, dt)
        
        # Initial conditions with custom velocity
        initial_state = [initial_position, velocity]
        
        # Solve differential equation
        solution = odeint(sim.equation_of_motion, initial_state, t)
        positions = solution[:, 0]
        velocities_result = solution[:, 1]
        
        # Calculate EMF and current for each time step
        emf_values = []
        current_values = []
        flux_values = []
        
        for pos, vel in zip(positions, velocities_result):
            emf = sim.induced_emf(pos, vel)
            current = emf / sim.resistance
            flux = sim.total_magnetic_flux(pos)
            
            emf_values.append(emf)
            current_values.append(current)
            flux_values.append(flux)
        
        # Convert to numpy arrays
        emf_values = np.array(emf_values)
        current_values = np.array(current_values)
        flux_values = np.array(flux_values)
        
        # Create results dictionary
        results = {
            'time': t,
            'position': positions,
            'velocity': velocities_result,
            'emf': emf_values,
            'current': current_values,
            'flux': flux_values,
            'resistance': sim.resistance,
            'initial_velocity': velocity  # Store the exact initial velocity value
        }
        
        # Store results in list instead of dictionary to avoid floating point key issues
        all_results.append(results)
        
        # Print max EMF
        max_emf = np.max(np.abs(emf_values)) * 1000  # mV
        print(f"Max EMF: {max_emf:.2f} mV ✓")
    
    # Create the EMF vs time overlay plot
    print("\nGenerating EMF vs Time overlay plot (positive initial velocities only)...")
    
    # Create scientific figure
    plt.figure(figsize=(12, 8), dpi=150, facecolor='white')
    
    # Set scientific font properties
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman', 'Times', 'serif'],
        'mathtext.fontset': 'dejavuserif',
        'axes.labelsize': 14,
        'axes.titlesize': 15,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11
    })
    
    # Plot EMF vs time for positive initial velocities only (v0 ≥ 0)
    for i, velocity in enumerate(velocities):
        if velocity < 0:  # Skip negative velocities
            continue
            
        results = all_results[i]  # Use index instead of velocity as key
        
        # Find when magnet interacts with spring
        spring_top = spring_length/2
        spring_bottom = -spring_length/2
        spring_buffer = 0.05  # 5 cm buffer
        
        # Identify interaction region
        mask = (results['position'] <= spring_top + spring_buffer) & (results['position'] >= spring_bottom - spring_buffer)
        
        if np.any(mask):
            # Extract data within the interaction region
            time = results['time'][mask]
            emf = results['emf'][mask]
            
            # Align all curves to start at t=0 for the interaction period
            time_shifted = time - time[0]
            
            # Plot with scientific styling
            label = f"v₀ = {velocity:.1f} m/s"
            if velocity == 0:
                label = "v₀ = 0 (free fall)"
                
            plt.plot(time_shifted, emf * 1000, color=colors[i], 
                    linewidth=2.5, label=label)
    
    # Add axis labels with LaTeX formatting
    plt.xlabel(r'Time $t$ [s]', fontsize=15)
    plt.ylabel(r'Induced EMF $\mathcal{E}$ [mV]', fontsize=15)
    
    # Add title
    plt.title('Effect of Initial Velocity on Induced EMF (v₀ ≥ 0)', fontsize=16, pad=20)
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add legend
    legend = plt.legend(fontsize=12, loc='best', frameon=True, framealpha=0.7,
                       edgecolor='gray', fancybox=False)
    
    # Scientific tick formatting
    plt.tick_params(axis='both', which='major', direction='in', 
                   length=5, width=1.0, bottom=True, top=True, left=True, right=True)
    plt.tick_params(axis='both', which='minor', direction='in', 
                   length=3, width=0.5, bottom=True, top=True, left=True, right=True)
    
    # Add minor ticks
    plt.minorticks_on()
    
    # Add box around the plot
    plt.box(True)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    filename = 'initial_velocity_comparison_positive.png'
    filepath = os.path.join(plots_folder, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    # Also save as PDF for publication
    pdf_filename = 'initial_velocity_comparison_positive.pdf'
    pdf_filepath = os.path.join(plots_folder, pdf_filename)
    plt.savefig(pdf_filepath, bbox_inches='tight', format='pdf')
    
    plt.show()
    
    # Print summary and statistics
    print("\nInitial Velocity Comparison Summary:")
    print("=" * 60)
    print(f"{'Velocity (m/s)':<15} {'Max EMF (mV)':<15} {'Peak-to-Peak (mV)':<20} {'Peak Time (s)':<15}")
    print("-" * 65)
    
    for i, velocity in enumerate(velocities):
        results = all_results[i]  # Use index instead of velocity value
        emf = results['emf']
        time = results['time']
        
        max_emf = np.max(emf) * 1000
        min_emf = np.min(emf) * 1000
        peak_to_peak = max_emf - min_emf
        max_idx = np.argmax(np.abs(emf))
        peak_time = time[max_idx]
        
        print(f"{velocity:<15.1f} {max_emf:<15.2f} {peak_to_peak:<20.2f} {peak_time:<15.4f}")
    
    print("\nObservations:")
    print("1. Higher initial velocities (magnitude) produce larger EMF peaks")
    print("2. The direction of the initial velocity affects the EMF waveform shape")
    print("3. Negative velocities (downward) produce EMF peaks faster")
    
    print(f"\nPlots saved as:")
    print(f"  PNG: {filepath}")
    print(f"  PDF: {pdf_filepath}")
    
    return all_results


def simulate_single_coil(magnetic_dipole_moment=2.0, spring_radius=0.02, spring_length=1.0, 
                        num_turns=50, wire_radius=0.001, resistivity=1.7e-8, 
                        magnet_mass=0.01, gravity=9.81, initial_velocity=0.0, 
                        duration=1.0, dt=0.0001, save_data=True, save_plots=True):
    """
    Simulate a single coil with a magnet falling through it and plot EMF vs time.
    
    Args:
        magnetic_dipole_moment: Magnetic dipole moment in A·m²
        spring_radius: Radius of the spring coil in meters
        spring_length: Length of the spring in meters
        num_turns: Number of turns in the spring
        wire_radius: Radius of the wire forming the spring
        resistivity: Electrical resistivity of the wire material (Ω·m)
        magnet_mass: Mass of the magnet in kg
        gravity: Gravitational acceleration in m/s²
        initial_velocity: Initial velocity of the magnet (m/s)
        duration: Simulation duration in seconds
        dt: Time step for simulation
        save_data: Whether to save simulation data to CSV
        save_plots: Whether to save plots to files
        
    Returns:
        Dictionary containing simulation results
    """
    print("=== Single Coil Simulation ===")
    print(f"Simulating magnet through single coil...")
    print(f"Initial velocity: {initial_velocity} m/s")
    print(f"Duration: {duration} s")
    
    # Create simulation object
    sim = MagnetThroughSpringSimulation(
        magnetic_dipole_moment=magnetic_dipole_moment,
        spring_radius=spring_radius,
        spring_length=spring_length,
        num_turns=num_turns,
        wire_radius=wire_radius,
        resistivity=resistivity,
        magnet_mass=magnet_mass,
        gravity=gravity
    )
    
    # Set initial conditions
    # Start magnet well above the coil
    initial_position = -spring_length * 2
    initial_conditions = [initial_position, initial_velocity]
    
    # Run simulation
    print("Running simulation...")
    results = sim.run_simulation(duration=duration, dt=dt)
    
    # Extract data
    time = results['time']
    position = results['position']
    velocity = results['velocity']
    emf = results['emf']
    current = results['current']
    flux = results['flux']
    
    # Calculate some metrics
    max_emf = np.max(np.abs(emf)) * 1000  # mV
    min_emf = np.min(emf) * 1000  # mV
    max_emf_pos = np.max(emf) * 1000  # mV
    max_current = np.max(np.abs(current)) * 1000  # mA
    
    print(f"\nSimulation Results:")
    print(f"  Maximum EMF magnitude: {max_emf:.2f} mV")
    print(f"  Maximum positive EMF: {max_emf_pos:.2f} mV")
    print(f"  Minimum EMF: {min_emf:.2f} mV")
    print(f"  Maximum current: {max_current:.2f} mA")
    print(f"  Coil resistance: {sim.resistance*1000:.2f} mΩ")
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Single Coil Simulation Results', fontsize=16, fontweight='bold')
    
    # EMF vs Time
    axes[0, 0].plot(time, emf * 1000, 'b-', linewidth=2, label='EMF')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('EMF (mV)')
    axes[0, 0].set_title('Induced EMF vs Time')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # EMF vs Position
    axes[0, 1].plot(position * 1000, emf * 1000, 'r-', linewidth=2, label='EMF')
    axes[0, 1].set_xlabel('Position (mm)')
    axes[0, 1].set_ylabel('EMF (mV)')
    axes[0, 1].set_title('Induced EMF vs Position')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Add coil boundaries to position plot
    coil_start = -spring_length/2 * 1000
    coil_end = spring_length/2 * 1000
    axes[0, 1].axvline(coil_start, color='gray', linestyle='--', alpha=0.7, label='Coil boundaries')
    axes[0, 1].axvline(coil_end, color='gray', linestyle='--', alpha=0.7)
    axes[0, 1].legend()
    
    # Current vs Time
    axes[1, 0].plot(time, current * 1000, 'g-', linewidth=2, label='Current')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Current (mA)')
    axes[1, 0].set_title('Induced Current vs Time')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Position and Velocity vs Time
    ax_pos = axes[1, 1]
    ax_vel = ax_pos.twinx()
    
    line1 = ax_pos.plot(time, position * 1000, 'orange', linewidth=2, label='Position')
    line2 = ax_vel.plot(time, velocity, 'purple', linewidth=2, label='Velocity')
    
    ax_pos.set_xlabel('Time (s)')
    ax_pos.set_ylabel('Position (mm)', color='orange')
    ax_vel.set_ylabel('Velocity (m/s)', color='purple')
    ax_pos.set_title('Magnet Position and Velocity vs Time')
    ax_pos.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax_pos.legend(lines, labels, loc='upper right')
    
    plt.tight_layout()
    
    if save_plots:
        # Create output directory
        os.makedirs('single_coil_output', exist_ok=True)
        
        plt.savefig('single_coil_output/single_coil_simulation.png', dpi=150, bbox_inches='tight')
        print(f"Plot saved to: single_coil_output/single_coil_simulation.png")
    
    plt.show()
    
    # Create a focused EMF vs Time plot
    plt.figure(figsize=(10, 6))
    plt.plot(time, emf * 1000, 'b-', linewidth=2, label='Induced EMF')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('EMF (mV)', fontsize=12)
    plt.title('Induced EMF vs Time - Single Coil', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Add text box with key parameters
    textstr = f'Coil Length: {spring_length*1000:.0f} mm\nTurns: {num_turns}\nRadius: {spring_radius*1000:.1f} mm\nResistance: {sim.resistance*1000:.1f} mΩ'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('single_coil_output/emf_vs_time_focused.png', dpi=150, bbox_inches='tight')
        print(f"EMF plot saved to: single_coil_output/emf_vs_time_focused.png")
    
    plt.show()
    
    # Save data if requested
    if save_data:
        os.makedirs('single_coil_output', exist_ok=True)
        
        # Create DataFrame
        data_df = pd.DataFrame({
            'Time_s': time,
            'Position_mm': position * 1000,
            'Velocity_ms': velocity,
            'EMF_mV': emf * 1000,
            'Current_mA': current * 1000,
            'MagneticFlux_Wb': flux
        })
        
        # Save to CSV
        filename = 'single_coil_output/single_coil_data.csv'
        data_df.to_csv(filename, index=False)
        print(f"Data saved to: {filename}")
        
        # Save parameters
        params_filename = 'single_coil_output/simulation_parameters.txt'
        with open(params_filename, 'w') as f:
            f.write("Single Coil Simulation Parameters:\n")
            f.write("=====================================\n")
            f.write(f"Magnetic dipole moment: {magnetic_dipole_moment} A·m²\n")
            f.write(f"Spring radius: {spring_radius*1000:.1f} mm\n")
            f.write(f"Spring length: {spring_length*1000:.1f} mm\n")
            f.write(f"Number of turns: {num_turns}\n")
            f.write(f"Wire radius: {wire_radius*1000:.2f} mm\n")
            f.write(f"Resistivity: {resistivity:.2e} Ω·m\n")
            f.write(f"Magnet mass: {magnet_mass*1000:.1f} g\n")
            f.write(f"Gravity: {gravity:.2f} m/s²\n")
            f.write(f"Initial velocity: {initial_velocity} m/s\n")
            f.write(f"Simulation duration: {duration} s\n")
            f.write(f"Time step: {dt} s\n")
            f.write(f"\nCalculated Properties:\n")
            f.write(f"Wire length: {sim.wire_length*1000:.1f} mm\n")
            f.write(f"Total resistance: {sim.resistance*1000:.2f} mΩ\n")
            f.write(f"Pitch: {sim.pitch*1000:.2f} mm\n")
            f.write(f"\nResults:\n")
            f.write(f"Maximum EMF magnitude: {max_emf:.2f} mV\n")
            f.write(f"Maximum positive EMF: {max_emf_pos:.2f} mV\n")
            f.write(f"Minimum EMF: {min_emf:.2f} mV\n")
            f.write(f"Maximum current: {max_current:.2f} mA\n")
        
        print(f"Parameters saved to: {params_filename}")
    
    # Return results dictionary
    return {
        'time': time,
        'position': position,
        'velocity': velocity,
        'emf': emf,
        'current': current,
        'flux': flux,
        'simulation': sim,
        'max_emf_mv': max_emf,
        'min_emf_mv': min_emf,
        'max_current_ma': max_current
    }


def run_single_coil_example():
    """
    Run an example single coil simulation with default parameters.
    """
    print("Running single coil example simulation...")
    
    # Run simulation with default parameters
    results = simulate_single_coil(
        spring_length=0.1,      # 10 cm coil
        num_turns=100,          # 100 turns
        spring_radius=0.015,    # 15 mm radius
        initial_velocity=0.0,   # Drop from rest
        duration=0.5,           # 0.5 second simulation
        save_data=True,
        save_plots=True
    )
    
    print("Single coil simulation completed!")
    return results


class TwoStackedCoilsSimulation:
    """
    Simulates a magnetic dipole falling through two stacked coils.
    Each coil can have different properties and generates separate EMF signals.
    """
    
    def __init__(self, 
                 # Coil 1 parameters (top coil)
                 coil1_radius=0.02, coil1_length=0.5, coil1_turns=50, 
                 coil1_wire_radius=0.001, coil1_resistivity=1.7e-8,
                 # Coil 2 parameters (bottom coil)
                 coil2_radius=0.02, coil2_length=0.5, coil2_turns=50,
                 coil2_wire_radius=0.001, coil2_resistivity=1.7e-8,
                 # Gap between coils
                 gap_between_coils=0.05,
                 # Magnet parameters
                 magnetic_dipole_moment=2.0, magnet_mass=0.01, gravity=9.81):
        """
        Initialize two stacked coils simulation.
        
        Args:
            coil1_radius: Radius of top coil in meters
            coil1_length: Length of top coil in meters
            coil1_turns: Number of turns in top coil
            coil1_wire_radius: Wire radius for top coil
            coil1_resistivity: Wire resistivity for top coil
            coil2_radius: Radius of bottom coil in meters
            coil2_length: Length of bottom coil in meters
            coil2_turns: Number of turns in bottom coil
            coil2_wire_radius: Wire radius for bottom coil
            coil2_resistivity: Wire resistivity for bottom coil
            gap_between_coils: Gap between the two coils in meters
            magnetic_dipole_moment: Magnetic dipole moment in A·m²
            magnet_mass: Mass of the magnet in kg
            gravity: Gravitational acceleration in m/s²
        """
        # Create two separate coil simulations
        self.coil1 = MagnetThroughSpringSimulation(
            magnetic_dipole_moment=magnetic_dipole_moment,
            spring_radius=coil1_radius,
            spring_length=coil1_length,
            num_turns=coil1_turns,
            wire_radius=coil1_wire_radius,
            resistivity=coil1_resistivity,
            magnet_mass=magnet_mass,
            gravity=gravity
        )
        
        self.coil2 = MagnetThroughSpringSimulation(
            magnetic_dipole_moment=magnetic_dipole_moment,
            spring_radius=coil2_radius,
            spring_length=coil2_length,
            num_turns=coil2_turns,
            wire_radius=coil2_wire_radius,
            resistivity=coil2_resistivity,
            magnet_mass=magnet_mass,
            gravity=gravity
        )
        
        self.gap = gap_between_coils
        self.mass = magnet_mass
        self.g = gravity
        
        # Calculate coil positions
        # Coil 1 (top) centered at z = +coil1_length/2 + gap/2
        # Coil 2 (bottom) centered at z = -coil2_length/2 - gap/2
        self.coil1_center = coil1_length/2 + gap_between_coils/2
        self.coil2_center = -coil2_length/2 - gap_between_coils/2
        
        print(f"\nTwo Stacked Coils Configuration:")
        print(f"Coil 1 (Top):")
        print(f"  Center position: {self.coil1_center*1000:.1f} mm")
        print(f"  Range: {(self.coil1_center - coil1_length/2)*1000:.1f} to {(self.coil1_center + coil1_length/2)*1000:.1f} mm")
        print(f"  Turns: {coil1_turns}, Resistance: {self.coil1.resistance*1000:.2f} mΩ")
        print(f"Coil 2 (Bottom):")
        print(f"  Center position: {self.coil2_center*1000:.1f} mm")
        print(f"  Range: {(self.coil2_center - coil2_length/2)*1000:.1f} to {(self.coil2_center + coil2_length/2)*1000:.1f} mm")
        print(f"  Turns: {coil2_turns}, Resistance: {self.coil2.resistance*1000:.2f} mΩ")
        print(f"Gap between coils: {gap_between_coils*1000:.1f} mm")
    
    def get_coil1_emf(self, z_magnet, velocity):
        """Calculate EMF in coil 1 based on magnet position relative to coil 1 center."""
        z_relative_to_coil1 = z_magnet - self.coil1_center
        return self.coil1.induced_emf(z_relative_to_coil1, velocity)
    
    def get_coil2_emf(self, z_magnet, velocity):
        """Calculate EMF in coil 2 based on magnet position relative to coil 2 center."""
        z_relative_to_coil2 = z_magnet - self.coil2_center
        return self.coil2.induced_emf(z_relative_to_coil2, velocity)
    
    def equation_of_motion(self, state, t):
        """
        Differential equation for magnet motion through two stacked coils.
        Both coils contribute to the magnetic force on the magnet.
        """
        z_pos, z_vel = state
        
        # Calculate EMF and current in both coils
        emf1 = self.get_coil1_emf(z_pos, z_vel)
        emf2 = self.get_coil2_emf(z_pos, z_vel)
        
        current1 = emf1 / self.coil1.resistance
        current2 = emf2 / self.coil2.resistance
        
        # Magnetic force (simplified - sum of forces from both coils)
        # This is an approximation - in reality the force calculation would be more complex
        z_rel_coil1 = z_pos - self.coil1_center
        z_rel_coil2 = z_pos - self.coil2_center
        
        # Force from coil 1
        if abs(z_rel_coil1) < self.coil1.L:
            force1 = -current1 * self.coil1.m * self.coil1.mu_0 / (4 * np.pi) * 0.1  # Simplified
        else:
            force1 = 0
            
        # Force from coil 2
        if abs(z_rel_coil2) < self.coil2.L:
            force2 = -current2 * self.coil2.m * self.coil2.mu_0 / (4 * np.pi) * 0.1  # Simplified
        else:
            force2 = 0
        
        total_magnetic_force = force1 + force2
        
        # Net acceleration
        acceleration = self.g + total_magnetic_force / self.mass
        
        return [z_vel, acceleration]
    
    def run_simulation(self, duration=2.0, dt=0.0001, initial_velocity=0.0):
        """
        Run the two-coil simulation.
        
        Args:
            duration: Simulation time in seconds
            dt: Time step
            initial_velocity: Initial velocity of magnet
            
        Returns:
            Dictionary with simulation results for both coils
        """
        # Time array
        t = np.arange(0, duration, dt)
        
        # Initial conditions - start magnet well above both coils
        total_system_length = (self.coil1_center + self.coil1.L/2) - (self.coil2_center - self.coil2.L/2)
        initial_position = self.coil1_center + self.coil1.L/2 + total_system_length
        initial_conditions = [initial_position, initial_velocity]
        
        # Solve differential equation
        solution = odeint(self.equation_of_motion, initial_conditions, t)
        positions = solution[:, 0]
        velocities = solution[:, 1]
        
        # Calculate EMF and current for both coils at each time step
        emf1_values = []
        emf2_values = []
        current1_values = []
        current2_values = []
        flux1_values = []
        flux2_values = []
        
        for i, (pos, vel) in enumerate(zip(positions, velocities)):
            emf1 = self.get_coil1_emf(pos, vel)
            emf2 = self.get_coil2_emf(pos, vel)
            
            current1 = emf1 / self.coil1.resistance
            current2 = emf2 / self.coil2.resistance
            
            # Calculate flux (relative to each coil center)
            z_rel_coil1 = pos - self.coil1_center
            z_rel_coil2 = pos - self.coil2_center
            
            flux1 = self.coil1.total_magnetic_flux(z_rel_coil1)
            flux2 = self.coil2.total_magnetic_flux(z_rel_coil2)
            
            emf1_values.append(emf1)
            emf2_values.append(emf2)
            current1_values.append(current1)
            current2_values.append(current2)
            flux1_values.append(flux1)
            flux2_values.append(flux2)
        
        return {
            'time': t,
            'position': positions,
            'velocity': velocities,
            'emf1': np.array(emf1_values),
            'emf2': np.array(emf2_values),
            'current1': np.array(current1_values),
            'current2': np.array(current2_values),
            'flux1': np.array(flux1_values),
            'flux2': np.array(flux2_values),
            'coil1_center': self.coil1_center,
            'coil2_center': self.coil2_center,
            'coil1_resistance': self.coil1.resistance,
            'coil2_resistance': self.coil2.resistance
        }


def simulate_two_stacked_coils(coil1_length=0.5, coil1_turns=50, coil1_radius=0.02,
                              coil2_length=0.5, coil2_turns=50, coil2_radius=0.02,
                              gap_between_coils=0.05, initial_velocity=0.0,
                              duration=2.0, dt=0.0001, save_data=True, save_plots=True):
    """
    Simulate two stacked coils and plot EMF vs time for each coil.
    
    Args:
        coil1_length: Length of top coil in meters
        coil1_turns: Number of turns in top coil
        coil1_radius: Radius of top coil in meters
        coil2_length: Length of bottom coil in meters
        coil2_turns: Number of turns in bottom coil
        coil2_radius: Radius of bottom coil in meters
        gap_between_coils: Gap between coils in meters
        initial_velocity: Initial velocity of magnet
        duration: Simulation duration in seconds
        dt: Time step
        save_data: Whether to save data to CSV
        save_plots: Whether to save plots
        
    Returns:
        Dictionary with simulation results
    """
    print("=== Two Stacked Coils Simulation ===")
    
    # Create simulation
    sim = TwoStackedCoilsSimulation(
        coil1_radius=coil1_radius,
        coil1_length=coil1_length,
        coil1_turns=coil1_turns,
        coil2_radius=coil2_radius,
        coil2_length=coil2_length,
        coil2_turns=coil2_turns,
        gap_between_coils=gap_between_coils,
        initial_velocity=initial_velocity
    )
    
    # Run simulation
    print("Running two-coil simulation...")
    results = sim.run_simulation(duration=duration, dt=dt, initial_velocity=initial_velocity)
    
    # Extract results
    time = results['time']
    position = results['position']
    velocity = results['velocity']
    emf1 = results['emf1']
    emf2 = results['emf2']
    current1 = results['current1']
    current2 = results['current2']
    
    # Calculate metrics
    max_emf1 = np.max(np.abs(emf1)) * 1000  # mV
    max_emf2 = np.max(np.abs(emf2)) * 1000  # mV
    max_current1 = np.max(np.abs(current1)) * 1000  # mA
    max_current2 = np.max(np.abs(current2)) * 1000  # mA
    
    print(f"\nSimulation Results:")
    print(f"Coil 1 (Top):")
    print(f"  Max EMF: {max_emf1:.2f} mV")
    print(f"  Max Current: {max_current1:.2f} mA")
    print(f"Coil 2 (Bottom):")
    print(f"  Max EMF: {max_emf2:.2f} mV")
    print(f"  Max Current: {max_current2:.2f} mA")
    
    # Create comprehensive plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Two Stacked Coils Simulation Results', fontsize=16, fontweight='bold')
    
    # EMF vs Time for both coils
    axes[0, 0].plot(time, emf1 * 1000, 'b-', linewidth=2, label='Coil 1 (Top)')
    axes[0, 0].plot(time, emf2 * 1000, 'r-', linewidth=2, label='Coil 2 (Bottom)')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('EMF (mV)')
    axes[0, 0].set_title('EMF vs Time - Both Coils')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # EMF vs Position for both coils
    axes[0, 1].plot(position * 1000, emf1 * 1000, 'b-', linewidth=2, label='Coil 1 (Top)')
    axes[0, 1].plot(position * 1000, emf2 * 1000, 'r-', linewidth=2, label='Coil 2 (Bottom)')
    axes[0, 1].set_xlabel('Position (mm)')
    axes[0, 1].set_ylabel('EMF (mV)')
    axes[0, 1].set_title('EMF vs Position - Both Coils')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add coil boundary markers
    coil1_start = (results['coil1_center'] - coil1_length/2) * 1000
    coil1_end = (results['coil1_center'] + coil1_length/2) * 1000
    coil2_start = (results['coil2_center'] - coil2_length/2) * 1000
    coil2_end = (results['coil2_center'] + coil2_length/2) * 1000
    
    axes[0, 1].axvspan(coil1_start, coil1_end, alpha=0.2, color='blue', label='Coil 1 region')
    axes[0, 1].axvspan(coil2_start, coil2_end, alpha=0.2, color='red', label='Coil 2 region')
    axes[0, 1].legend()
    
    # Current vs Time for both coils
    axes[1, 0].plot(time, current1 * 1000, 'b-', linewidth=2, label='Coil 1 (Top)')
    axes[1, 0].plot(time, current2 * 1000, 'r-', linewidth=2, label='Coil 2 (Bottom)')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Current (mA)')
    axes[1, 0].set_title('Current vs Time - Both Coils')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Position vs Time with coil regions
    axes[1, 1].plot(time, position * 1000, 'purple', linewidth=2, label='Magnet Position')
    axes[1, 1].axhline(coil1_start, color='blue', linestyle='--', alpha=0.7, label='Coil 1 boundaries')
    axes[1, 1].axhline(coil1_end, color='blue', linestyle='--', alpha=0.7)
    axes[1, 1].axhline(coil2_start, color='red', linestyle='--', alpha=0.7, label='Coil 2 boundaries')
    axes[1, 1].axhline(coil2_end, color='red', linestyle='--', alpha=0.7)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Position (mm)')
    axes[1, 1].set_title('Magnet Position vs Time')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_plots:
        os.makedirs('two_coils_output', exist_ok=True)
        plt.savefig('two_coils_output/two_coils_simulation.png', dpi=150, bbox_inches='tight')
        print(f"Plot saved to: two_coils_output/two_coils_simulation.png")
    
    plt.show()
    
    # Create focused EMF comparison plot
    plt.figure(figsize=(12, 6))
    plt.plot(time, emf1 * 1000, 'b-', linewidth=2, label='Coil 1 (Top)')
    plt.plot(time, emf2 * 1000, 'r-', linewidth=2, label='Coil 2 (Bottom)')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('EMF (mV)', fontsize=12)
    plt.title('EMF vs Time - Two Stacked Coils Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Add parameter text box
    textstr = f'Coil 1: {coil1_length*1000:.0f}mm, {coil1_turns} turns\nCoil 2: {coil2_length*1000:.0f}mm, {coil2_turns} turns\nGap: {gap_between_coils*1000:.1f}mm'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('two_coils_output/emf_comparison_focused.png', dpi=150, bbox_inches='tight')
        print(f"EMF comparison plot saved to: two_coils_output/emf_comparison_focused.png")
    
    plt.show()
    
    # Save data if requested
    if save_data:
        os.makedirs('two_coils_output', exist_ok=True)
        
        # Create DataFrame
        data_df = pd.DataFrame({
            'Time_s': time,
            'Position_mm': position * 1000,
            'Velocity_ms': velocity,
            'EMF1_mV': emf1 * 1000,
            'EMF2_mV': emf2 * 1000,
            'Current1_mA': current1 * 1000,
            'Current2_mA': current2 * 1000,
            'Flux1_Wb': results['flux1'],
            'Flux2_Wb': results['flux2']
        })
        
        filename = 'two_coils_output/two_coils_data.csv'
        data_df.to_csv(filename, index=False)
        print(f"Data saved to: {filename}")
        
        # Save parameters
        params_filename = 'two_coils_output/simulation_parameters.txt'
        with open(params_filename, 'w') as f:
            f.write("Two Stacked Coils Simulation Parameters:\n")
            f.write("========================================\n")
            f.write(f"Coil 1 (Top):\n")
            f.write(f"  Length: {coil1_length*1000:.1f} mm\n")
            f.write(f"  Turns: {coil1_turns}\n")
            f.write(f"  Radius: {coil1_radius*1000:.1f} mm\n")
            f.write(f"  Resistance: {results['coil1_resistance']*1000:.2f} mΩ\n")
            f.write(f"Coil 2 (Bottom):\n")
            f.write(f"  Length: {coil2_length*1000:.1f} mm\n")
            f.write(f"  Turns: {coil2_turns}\n")
            f.write(f"  Radius: {coil2_radius*1000:.1f} mm\n")
            f.write(f"  Resistance: {results['coil2_resistance']*1000:.2f} mΩ\n")
            f.write(f"Gap between coils: {gap_between_coils*1000:.1f} mm\n")
            f.write(f"Initial velocity: {initial_velocity} m/s\n")
            f.write(f"Simulation duration: {duration} s\n")
            f.write(f"\nResults:\n")
            f.write(f"Coil 1 max EMF: {max_emf1:.2f} mV\n")
            f.write(f"Coil 2 max EMF: {max_emf2:.2f} mV\n")
            f.write(f"Coil 1 max current: {max_current1:.2f} mA\n")
            f.write(f"Coil 2 max current: {max_current2:.2f} mA\n")
        
        print(f"Parameters saved to: {params_filename}")
    
    return {
        'time': time,
        'position': position,
        'velocity': velocity,
        'emf1': emf1,
        'emf2': emf2,
        'current1': current1,
        'current2': current2,
        'simulation': sim,
        'max_emf1_mv': max_emf1,
        'max_emf2_mv': max_emf2,
        'max_current1_ma': max_current1,
        'max_current2_ma': max_current2
    }


def run_two_coils_example():
    """
    Run an example two stacked coils simulation with default parameters.
    """
    print("Running two stacked coils example simulation...")
    
    # Run simulation with example parameters
    results = simulate_two_stacked_coils(
        coil1_length=0.1,       # 10 cm top coil
        coil1_turns=100,        # 100 turns top coil
        coil2_length=0.1,       # 10 cm bottom coil  
        coil2_turns=100,        # 100 turns bottom coil
        gap_between_coils=0.02, # 2 cm gap
        initial_velocity=0.0,   # Drop from rest
        duration=1.0,           # 1 second simulation
        save_data=True,
        save_plots=True
    )
    
    print("Two stacked coils simulation completed!")
    return results


def analyze_emf_extrema_positions(results, sim, verbose=True):
    """
    Analyze EMF extrema positions and compare with theoretical 1/4 and 3/4 positions.
    
    Args:
        results: Dictionary containing simulation results
        sim: Simulation object
        verbose: Whether to print detailed analysis
        
    Returns:
        Dictionary with extrema analysis results
    """
    time = results['time']
    position = results['position']
    emf = results['emf']
    
    # Define coil boundaries
    coil_top = sim.L/2
    coil_bottom = -sim.L/2
    coil_center = 0
    coil_length = sim.L
    
    # Find EMF extrema (considering only data within the coil region)
    # Add small buffer to ensure we capture extrema near boundaries
    buffer = 0.01  # 1 cm buffer
    coil_mask = (position >= coil_bottom - buffer) & (position <= coil_top + buffer)
    
    if not np.any(coil_mask):
        if verbose:
            print("Warning: No data points found within coil region")
        return None
    
    # Extract data within coil region
    position_coil = position[coil_mask]
    emf_coil = emf[coil_mask]
    time_coil = time[coil_mask]
    
    # Find maximum and minimum EMF indices within coil region
    max_emf_idx_local = np.argmax(emf_coil)
    min_emf_idx_local = np.argmin(emf_coil)
    
    # Get actual extrema positions
    max_emf_position = position_coil[max_emf_idx_local]
    min_emf_position = position_coil[min_emf_idx_local]
    max_emf_value = emf_coil[max_emf_idx_local]
    min_emf_value = emf_coil[min_emf_idx_local]
    max_emf_time = time_coil[max_emf_idx_local]
    min_emf_time = time_coil[min_emf_idx_local]
    
    # Calculate positions as fractions of coil length from top
    # Fraction = 0 means at top of coil, fraction = 1 means at bottom of coil
    max_emf_fraction_from_top = (coil_top - max_emf_position) / coil_length
    min_emf_fraction_from_top = (coil_top - min_emf_position) / coil_length
    
    # Theoretical positions (1/4 and 3/4 from top)
    theoretical_quarter_position = coil_top - 0.25 * coil_length  # 1/4 from top
    theoretical_three_quarter_position = coil_top - 0.75 * coil_length  # 3/4 from top
    
    # Calculate deviations from theoretical positions
    max_emf_deviation_from_quarter = abs(max_emf_position - theoretical_quarter_position)
    min_emf_deviation_from_three_quarter = abs(min_emf_position - theoretical_three_quarter_position)
    
    # Also check if min is at 1/4 and max is at 3/4 (opposite case)
    max_emf_deviation_from_three_quarter = abs(max_emf_position - theoretical_three_quarter_position)
    min_emf_deviation_from_quarter = abs(min_emf_position - theoretical_quarter_position)
    
    # Determine which theoretical assignment is better
    case1_total_deviation = max_emf_deviation_from_quarter + min_emf_deviation_from_three_quarter
    case2_total_deviation = max_emf_deviation_from_three_quarter + min_emf_deviation_from_quarter
    
    if case1_total_deviation <= case2_total_deviation:
        # Case 1: Max EMF at 1/4, Min EMF at 3/4
        best_case = "Max at 1/4, Min at 3/4"
        max_theoretical_pos = theoretical_quarter_position
        min_theoretical_pos = theoretical_three_quarter_position
        max_theoretical_fraction = 0.25
        min_theoretical_fraction = 0.75
        max_deviation = max_emf_deviation_from_quarter
        min_deviation = min_emf_deviation_from_three_quarter
    else:
        # Case 2: Max EMF at 3/4, Min EMF at 1/4
        best_case = "Max at 3/4, Min at 1/4"
        max_theoretical_pos = theoretical_three_quarter_position
        min_theoretical_pos = theoretical_quarter_position
        max_theoretical_fraction = 0.75
        min_theoretical_fraction = 0.25
        max_deviation = max_emf_deviation_from_three_quarter
        min_deviation = min_emf_deviation_from_quarter
    
    if verbose:
        print(f"\n" + "="*60)
        print(f"EMF EXTREMA POSITION ANALYSIS")
        print(f"="*60)
        
        print(f"\nCoil Geometry:")
        print(f"  Coil length: {coil_length*1000:.1f} mm")
        print(f"  Coil top: {coil_top*1000:.1f} mm")
        print(f"  Coil bottom: {coil_bottom*1000:.1f} mm")
        print(f"  Coil center: {coil_center*1000:.1f} mm")
        
        print(f"\nTheoretical Positions:")
        print(f"  1/4 position: {theoretical_quarter_position*1000:.1f} mm")
        print(f"  3/4 position: {theoretical_three_quarter_position*1000:.1f} mm")
        
        print(f"\nActual EMF Extrema:")
        print(f"  Maximum EMF: {max_emf_value*1000:.3f} mV")
        print(f"    Position: {max_emf_position*1000:.1f} mm")
        print(f"    Fraction from top: {max_emf_fraction_from_top:.3f}")
        print(f"    Time: {max_emf_time:.4f} s")
        
        print(f"  Minimum EMF: {min_emf_value*1000:.3f} mV")
        print(f"    Position: {min_emf_position*1000:.1f} mm")
        print(f"    Fraction from top: {min_emf_fraction_from_top:.3f}")
        print(f"    Time: {min_emf_time:.4f} s")
        
        print(f"\nComparison with Theory:")
        print(f"  Best theoretical assignment: {best_case}")
        
        print(f"\n  Maximum EMF Analysis:")
        print(f"    Actual fraction from top: {max_emf_fraction_from_top:.3f}")
        print(f"    Theoretical fraction: {max_theoretical_fraction:.3f}")
        print(f"    Deviation: {abs(max_emf_fraction_from_top - max_theoretical_fraction):.3f}")
        print(f"    Position deviation: {max_deviation*1000:.1f} mm")
        
        print(f"\n  Minimum EMF Analysis:")
        print(f"    Actual fraction from top: {min_emf_fraction_from_top:.3f}")
        print(f"    Theoretical fraction: {min_theoretical_fraction:.3f}")
        print(f"    Deviation: {abs(min_emf_fraction_from_top - min_theoretical_fraction):.3f}")
        print(f"    Position deviation: {min_deviation*1000:.1f} mm")
        
        # Accuracy assessment
        position_tolerance = 0.05  # 5% tolerance
        if (abs(max_emf_fraction_from_top - max_theoretical_fraction) <= position_tolerance and 
            abs(min_emf_fraction_from_top - min_theoretical_fraction) <= position_tolerance):
            print(f"\n  ✓ EXCELLENT agreement with theory (within {position_tolerance*100:.0f}% tolerance)")
        elif (abs(max_emf_fraction_from_top - max_theoretical_fraction) <= 0.1 and 
              abs(min_emf_fraction_from_top - min_theoretical_fraction) <= 0.1):
            print(f"\n  ✓ GOOD agreement with theory (within 10% tolerance)")
        elif (abs(max_emf_fraction_from_top - max_theoretical_fraction) <= 0.2 and 
              abs(min_emf_fraction_from_top - min_theoretical_fraction) <= 0.2):
            print(f"\n  ~ FAIR agreement with theory (within 20% tolerance)")
        else:
            print(f"\n  ✗ POOR agreement with theory (deviations > 20%)")
        
        print(f"\n" + "="*60)
    
    # Return analysis results
    return {
        'coil_length': coil_length,
        'coil_top': coil_top,
        'coil_bottom': coil_bottom,
        'max_emf_value': max_emf_value,
        'min_emf_value': min_emf_value,
        'max_emf_position': max_emf_position,
        'min_emf_position': min_emf_position,
        'max_emf_time': max_emf_time,
        'min_emf_time': min_emf_time,
        'max_emf_fraction_from_top': max_emf_fraction_from_top,
        'min_emf_fraction_from_top': min_emf_fraction_from_top,
        'theoretical_quarter_position': theoretical_quarter_position,
        'theoretical_three_quarter_position': theoretical_three_quarter_position,
        'best_theoretical_assignment': best_case,
        'max_theoretical_fraction': max_theoretical_fraction,
        'min_theoretical_fraction': min_theoretical_fraction,
        'max_position_deviation': max_deviation,
        'min_position_deviation': min_deviation,
        'max_fraction_deviation': abs(max_emf_fraction_from_top - max_theoretical_fraction),
        'min_fraction_deviation': abs(min_emf_fraction_from_top - min_theoretical_fraction)
    }


def simulate_single_coil_with_extrema_analysis(magnetic_dipole_moment=2.0, spring_radius=0.02, spring_length=1.0, 
                                             num_turns=50, wire_radius=0.001, resistivity=1.7e-8, 
                                             magnet_mass=0.01, gravity=9.81, initial_velocity=0.0, 
                                             duration=1.0, dt=0.0001, save_data=True, save_plots=True):
    """
    Simulate a single coil with detailed EMF extrema analysis comparing actual vs theoretical positions.
    
    Args:
        magnetic_dipole_moment: Magnetic dipole moment in A·m²
        spring_radius: Radius of the spring coil in meters
        spring_length: Length of the spring in meters
        num_turns: Number of turns in the spring
        wire_radius: Radius of the wire forming the spring
        resistivity: Electrical resistivity of the wire material (Ω·m)
        magnet_mass: Mass of the magnet in kg
        gravity: Gravitational acceleration in m/s²
        initial_velocity: Initial velocity of the magnet (m/s)
        duration: Simulation duration in seconds
        dt: Time step for simulation
        save_data: Whether to save simulation data to CSV
        save_plots: Whether to save plots to files
        
    Returns:
        Dictionary containing simulation results and extrema analysis
    """
    print("=== Single Coil Simulation with EMF Extrema Analysis ===")
    print(f"Simulating magnet through single coil...")
    print(f"Initial velocity: {initial_velocity} m/s")
    print(f"Duration: {duration} s")
    
    # Create simulation object
    sim = MagnetThroughSpringSimulation(
        magnetic_dipole_moment=magnetic_dipole_moment,
        spring_radius=spring_radius,
        spring_length=spring_length,
        num_turns=num_turns,
        wire_radius=wire_radius,
        resistivity=resistivity,
        magnet_mass=magnet_mass,
        gravity=gravity
    )
    
    # Run simulation
    print("Running simulation...")
    results = sim.run_simulation(duration=duration, dt=dt)
    
    # Perform detailed extrema analysis
    print("Analyzing EMF extrema positions...")
    extrema_analysis = analyze_emf_extrema_positions(results, sim, verbose=True)
    
    # Extract data for plotting
    time = results['time']
    position = results['position']
    velocity = results['velocity']
    emf = results['emf']
    current = results['current']
    flux = results['flux']
    
    # Calculate some metrics
    max_emf = np.max(np.abs(emf)) * 1000  # mV
    min_emf = np.min(emf) * 1000  # mV
    max_emf_pos = np.max(emf) * 1000  # mV
    max_current = np.max(np.abs(current)) * 1000  # mA
    
    print(f"\nBasic Simulation Results:")
    print(f"  Maximum EMF magnitude: {max_emf:.2f} mV")
    print(f"  Maximum positive EMF: {max_emf_pos:.2f} mV")
    print(f"  Minimum EMF: {min_emf:.2f} mV")
    print(f"  Maximum current: {max_current:.2f} mA")
    print(f"  Coil resistance: {sim.resistance*1000:.2f} mΩ")
    
    # Create enhanced plots with extrema analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Single Coil Simulation with EMF Extrema Analysis', fontsize=16, fontweight='bold')
    
    # EMF vs Time with extrema markers
    axes[0, 0].plot(time, emf * 1000, 'b-', linewidth=2, label='EMF')
    if extrema_analysis:
        axes[0, 0].plot(extrema_analysis['max_emf_time'], extrema_analysis['max_emf_value'] * 1000, 
                       'ro', markersize=8, label=f'Max EMF ({extrema_analysis["max_emf_fraction_from_top"]:.3f} from top)')
        axes[0, 0].plot(extrema_analysis['min_emf_time'], extrema_analysis['min_emf_value'] * 1000, 
                       'go', markersize=8, label=f'Min EMF ({extrema_analysis["min_emf_fraction_from_top"]:.3f} from top)')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('EMF (mV)')
    axes[0, 0].set_title('Induced EMF vs Time')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # EMF vs Position with theoretical markers
    axes[0, 1].plot(position * 1000, emf * 1000, 'b-', linewidth=2, label='EMF')
    
    # Add coil boundaries
    coil_start = -spring_length/2 * 1000
    coil_end = spring_length/2 * 1000
    axes[0, 1].axvline(coil_start, color='gray', linestyle='--', alpha=0.7, label='Coil boundaries')
    axes[0, 1].axvline(coil_end, color='gray', linestyle='--', alpha=0.7)
    
    if extrema_analysis:
        # Mark actual extrema
        axes[0, 1].plot(extrema_analysis['max_emf_position'] * 1000, extrema_analysis['max_emf_value'] * 1000, 
                       'ro', markersize=8, label=f'Actual Max EMF')
        axes[0, 1].plot(extrema_analysis['min_emf_position'] * 1000, extrema_analysis['min_emf_value'] * 1000, 
                       'go', markersize=8, label=f'Actual Min EMF')
        
        # Mark theoretical positions
        axes[0, 1].axvline(extrema_analysis['theoretical_quarter_position'] * 1000, 
                          color='red', linestyle=':', alpha=0.8, label='Theoretical 1/4 position')
        axes[0, 1].axvline(extrema_analysis['theoretical_three_quarter_position'] * 1000, 
                          color='green', linestyle=':', alpha=0.8, label='Theoretical 3/4 position')
    
    axes[0, 1].set_xlabel('Position (mm)')
    axes[0, 1].set_ylabel('EMF (mV)')
    axes[0, 1].set_title('Induced EMF vs Position with Theoretical Markers')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Current vs Time
    axes[1, 0].plot(time, current * 1000, 'g-', linewidth=2, label='Current')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Current (mA)')
    axes[1, 0].set_title('Induced Current vs Time')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Position vs Time with coil region highlighted
    axes[1, 1].plot(time, position * 1000, 'purple', linewidth=2, label='Magnet Position')
    axes[1, 1].axhline(coil_start, color='gray', linestyle='--', alpha=0.7, label='Coil boundaries')
    axes[1, 1].axhline(coil_end, color='gray', linestyle='--', alpha=0.7)
    
    if extrema_analysis:
        # Highlight coil region
        axes[1, 1].axhspan(coil_start, coil_end, alpha=0.1, color='blue', label='Coil region')
        
        # Mark extrema times
        axes[1, 1].axvline(extrema_analysis['max_emf_time'], color='red', linestyle=':', alpha=0.8, 
                          label='Max EMF time')
        axes[1, 1].axvline(extrema_analysis['min_emf_time'], color='green', linestyle=':', alpha=0.8, 
                          label='Min EMF time')
    
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Position (mm)')
    axes[1, 1].set_title('Magnet Position vs Time')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_plots:
        # Create output directory
        os.makedirs('single_coil_output', exist_ok=True)
        
        plt.savefig('single_coil_output/single_coil_extrema_analysis.png', dpi=150, bbox_inches='tight')
        print(f"Enhanced plot saved to: single_coil_output/single_coil_extrema_analysis.png")
    
    plt.show()
    
    # Create a focused EMF vs Position plot for publication
    plt.figure(figsize=(12, 8))
    plt.plot(position * 1000, emf * 1000, 'b-', linewidth=3, label='Induced EMF')
    
    # Add coil boundaries
    plt.axvline(coil_start, color='black', linestyle='-', alpha=0.8, linewidth=2, label='Coil boundaries')
    plt.axvline(coil_end, color='black', linestyle='-', alpha=0.8, linewidth=2)
    
    if extrema_analysis:
        # Mark actual extrema with larger markers
        plt.plot(extrema_analysis['max_emf_position'] * 1000, extrema_analysis['max_emf_value'] * 1000, 
                'ro', markersize=12, label=f'Max EMF (fraction: {extrema_analysis["max_emf_fraction_from_top"]:.3f})')
        plt.plot(extrema_analysis['min_emf_position'] * 1000, extrema_analysis['min_emf_value'] * 1000, 
                'go', markersize=12, label=f'Min EMF (fraction: {extrema_analysis["min_emf_fraction_from_top"]:.3f})')
        
        # Mark theoretical positions with dashed lines
        plt.axvline(extrema_analysis['theoretical_quarter_position'] * 1000, 
                   color='red', linestyle='--', alpha=0.8, linewidth=2, label='Theoretical 1/4 position')
        plt.axvline(extrema_analysis['theoretical_three_quarter_position'] * 1000, 
                   color='green', linestyle='--', alpha=0.8, linewidth=2, label='Theoretical 3/4 position')
        
        # Add text annotations
        plt.annotate(f'Max EMF\n{extrema_analysis["max_emf_value"]*1000:.2f} mV\nFraction: {extrema_analysis["max_emf_fraction_from_top"]:.3f}',
                    xy=(extrema_analysis['max_emf_position'] * 1000, extrema_analysis['max_emf_value'] * 1000),
                    xytext=(10, 10), textcoords='offset points', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.3),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.annotate(f'Min EMF\n{extrema_analysis["min_emf_value"]*1000:.2f} mV\nFraction: {extrema_analysis["min_emf_fraction_from_top"]:.3f}',
                    xy=(extrema_analysis['min_emf_position'] * 1000, extrema_analysis['min_emf_value'] * 1000),
                    xytext=(10, -30), textcoords='offset points', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.3),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.xlabel('Position (mm)', fontsize=14)
    plt.ylabel('EMF (mV)', fontsize=14)
    plt.title('EMF vs Position: Actual vs Theoretical Extrema Locations', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Add parameter text box
    if extrema_analysis:
        textstr = (f'Coil Length: {spring_length*1000:.0f} mm\n'
                  f'Turns: {num_turns}\n'
                  f'Radius: {spring_radius*1000:.1f} mm\n'
                  f'Best fit: {extrema_analysis["best_theoretical_assignment"]}\n'
                  f'Max deviation: {extrema_analysis["max_fraction_deviation"]:.3f}\n'
                  f'Min deviation: {extrema_analysis["min_fraction_deviation"]:.3f}')
    else:
        textstr = f'Coil Length: {spring_length*1000:.0f} mm\nTurns: {num_turns}\nRadius: {spring_radius*1000:.1f} mm'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('single_coil_output/emf_vs_position_extrema_analysis.png', dpi=150, bbox_inches='tight')
        print(f"EMF vs Position analysis plot saved to: single_coil_output/emf_vs_position_extrema_analysis.png")
    
    plt.show()
    
    # Save enhanced data if requested
    if save_data:
        os.makedirs('single_coil_output', exist_ok=True)
        
        # Create DataFrame with additional analysis columns
        data_df = pd.DataFrame({
            'Time_s': time,
            'Position_mm': position * 1000,
            'Position_fraction_from_top': (sim.L/2 - position) / sim.L,
            'Velocity_ms': velocity,
            'EMF_mV': emf * 1000,
            'Current_mA': current * 1000,
            'MagneticFlux_Wb': flux
        })
        
        # Save to CSV
        filename = 'single_coil_output/single_coil_extrema_data.csv'
        data_df.to_csv(filename, index=False)
        print(f"Enhanced data saved to: {filename}")
        
        # Save detailed analysis results
        if extrema_analysis:
            analysis_filename = 'single_coil_output/extrema_analysis_results.txt'
            with open(analysis_filename, 'w') as f:
                f.write("EMF EXTREMA POSITION ANALYSIS RESULTS\n")
                f.write("=====================================\n\n")
                f.write(f"Coil Parameters:\n")
                f.write(f"  Length: {spring_length*1000:.1f} mm\n")
                f.write(f"  Radius: {spring_radius*1000:.1f} mm\n")
                f.write(f"  Turns: {num_turns}\n")
                f.write(f"  Resistance: {sim.resistance*1000:.2f} mΩ\n\n")
                
                f.write(f"Theoretical Positions:\n")
                f.write(f"  1/4 position: {extrema_analysis['theoretical_quarter_position']*1000:.1f} mm\n")
                f.write(f"  3/4 position: {extrema_analysis['theoretical_three_quarter_position']*1000:.1f} mm\n\n")
                
                f.write(f"Actual EMF Extrema:\n")
                f.write(f"  Maximum EMF: {extrema_analysis['max_emf_value']*1000:.3f} mV\n")
                f.write(f"    Position: {extrema_analysis['max_emf_position']*1000:.1f} mm\n")
                f.write(f"    Fraction from top: {extrema_analysis['max_emf_fraction_from_top']:.3f}\n")
                f.write(f"    Time: {extrema_analysis['max_emf_time']:.4f} s\n\n")
                
                f.write(f"  Minimum EMF: {extrema_analysis['min_emf_value']*1000:.3f} mV\n")
                f.write(f"    Position: {extrema_analysis['min_emf_position']*1000:.1f} mm\n")
                f.write(f"    Fraction from top: {extrema_analysis['min_emf_fraction_from_top']:.3f}\n")
                f.write(f"    Time: {extrema_analysis['min_emf_time']:.4f} s\n\n")
                
                f.write(f"Comparison with Theory:\n")
                f.write(f"  Best theoretical assignment: {extrema_analysis['best_theoretical_assignment']}\n")
                f.write(f"  Max EMF theoretical fraction: {extrema_analysis['max_theoretical_fraction']:.3f}\n")
                f.write(f"  Min EMF theoretical fraction: {extrema_analysis['min_theoretical_fraction']:.3f}\n")
                f.write(f"  Max EMF fraction deviation: {extrema_analysis['max_fraction_deviation']:.3f}\n")
                f.write(f"  Min EMF fraction deviation: {extrema_analysis['min_fraction_deviation']:.3f}\n")
                f.write(f"  Max EMF position deviation: {extrema_analysis['max_position_deviation']*1000:.1f} mm\n")
                f.write(f"  Min EMF position deviation: {extrema_analysis['min_position_deviation']*1000:.1f} mm\n")
            
            print(f"Detailed analysis saved to: {analysis_filename}")
    
    # Return comprehensive results dictionary
    return {
        'time': time,
        'position': position,
        'velocity': velocity,
        'emf': emf,
        'current': current,
        'flux': flux,
        'simulation': sim,
        'max_emf_mv': max_emf,
        'min_emf_mv': min_emf,
        'max_current_ma': max_current,
        'extrema_analysis': extrema_analysis
    }


def run_single_coil_extrema_example():
    """
    Run an example single coil simulation with EMF extrema analysis.
    """
    print("Running single coil simulation with EMF extrema analysis...")
    
    # Run simulation with example parameters
    results = simulate_single_coil_with_extrema_analysis(
        spring_length=0.1,      # 10 cm coil
        num_turns=100,          # 100 turns
        spring_radius=0.015,    # 15 mm radius
        initial_velocity=0.0,   # Drop from rest
        duration=0.5,           # 0.5 second simulation
        save_data=True,
        save_plots=True
    )
    
    print("Single coil simulation with extrema analysis completed!")
    return results


if __name__ == "__main__":
    # Add menu option for single coil simulation
    print("Magnet Through Coil Simulation")
    print("==============================")
    print("1. Run comprehensive analysis")
    print("2. Compare coil lengths")
    print("3. Analyze gap length effects")
    print("4. Compare segmented vs continuous coils")
    print("5. Show EMF vs time plot only")
    print("6. Compare initial velocities")
    print("7. Run single coil simulation")
    print("8. Single coil example")
    print("9. Run two stacked coils simulation")
    print("10. Two stacked coils example")
    print("11. Single coil with EMF extrema analysis")
    
    choice = input("Enter your choice (1-11): ")
    
    if choice == "1":
        run_magnet_simulation()
    elif choice == "2":
        compare_coil_lengths()
    elif choice == "3":
        analyze_gap_length_effects()
    elif choice == "4":
        compare_segmented_vs_continuous_coils()
    elif choice == "5":
        show_emf_time_plot_only()
    elif choice == "6":
        compare_initial_velocities()
    elif choice == "7":
        # Interactive single coil simulation
        print("\nSingle Coil Simulation Setup")
        print("============================")
        
        try:
            length = float(input("Coil length (mm) [default: 100]: ") or "100") / 1000
            turns = int(input("Number of turns [default: 100]: ") or "100")
            radius = float(input("Coil radius (mm) [default: 15]: ") or "15") / 1000
            velocity = float(input("Initial velocity (m/s) [default: 0]: ") or "0")
            duration = float(input("Simulation duration (s) [default: 0.5]: ") or "0.5")
            
            simulate_single_coil(
                spring_length=length,
                num_turns=turns,
                spring_radius=radius,
                initial_velocity=velocity,
                duration=duration,
                save_data=True,
                save_plots=True
            )
        except ValueError:
            print("Invalid input. Using default parameters.")
            run_single_coil_example()
            
    elif choice == "8":
        run_single_coil_example()
    elif choice == "9":
        # Interactive two stacked coils simulation
        print("\nTwo Stacked Coils Simulation Setup")
        print("===================================")
        
        try:
            coil1_length = float(input("Top coil length (mm) [default: 100]: ") or "100") / 1000
            coil1_turns = int(input("Top coil turns [default: 100]: ") or "100")
            coil2_length = float(input("Bottom coil length (mm) [default: 100]: ") or "100") / 1000
            coil2_turns = int(input("Bottom coil turns [default: 100]: ") or "100")
            gap = float(input("Gap between coils (mm) [default: 20]: ") or "20") / 1000
            radius = float(input("Coil radius (mm) [default: 20]: ") or "20") / 1000
            velocity = float(input("Initial velocity (m/s) [default: 0]: ") or "0")
            duration = float(input("Simulation duration (s) [default: 1.0]: ") or "1.0")
            
            simulate_two_stacked_coils(
                coil1_length=coil1_length,
                coil1_turns=coil1_turns,
                coil1_radius=radius,
                coil2_length=coil2_length,
                coil2_turns=coil2_turns,
                coil2_radius=radius,
                gap_between_coils=gap,
                initial_velocity=velocity,
                duration=duration,
                save_data=True,
                save_plots=True
            )
        except ValueError:
            print("Invalid input. Using default parameters.")
            run_two_coils_example()
            
    elif choice == "10":
        run_two_coils_example()
    elif choice == "11":
        # Interactive single coil simulation with extrema analysis
        print("\nSingle Coil EMF Extrema Analysis Setup")
        print("=======================================")
        
        try:
            length = float(input("Coil length (mm) [default: 100]: ") or "100") / 1000
            turns = int(input("Number of turns [default: 100]: ") or "100")
            radius = float(input("Coil radius (mm) [default: 15]: ") or "15") / 1000
            velocity = float(input("Initial velocity (m/s) [default: 0]: ") or "0")
            duration = float(input("Simulation duration (s) [default: 0.5]: ") or "0.5")
            
            simulate_single_coil_with_extrema_analysis(
                spring_length=length,
                num_turns=turns,
                spring_radius=radius,
                initial_velocity=velocity,
                duration=duration,
                save_data=True,
                save_plots=True
            )
        except ValueError:
            print("Invalid input. Using default parameters.")
            run_single_coil_extrema_example()
    else:
        print("Invalid choice. Running single coil extrema analysis example.")
        run_single_coil_extrema_example()
