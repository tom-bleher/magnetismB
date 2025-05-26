#!/usr/bin/env python3
# filepath: /home/tom/Desktop/magnetism/MagnetismB/simulate.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.integrate import odeint

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
        Create comprehensive plots of the simulation results.
        
        Args:
            results: Dictionary containing simulation results
            save_plots: Whether to save plots to files
        """
        # Create plots folder if it doesn't exist
        if save_plots:
            plots_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
            os.makedirs(plots_folder, exist_ok=True)
        
        # Find EMF extrema for marking on plots
        max_emf_idx = np.argmax(results['emf'])
        min_emf_idx = np.argmin(results['emf'])
        
        # Determine spring interaction period
        spring_top = self.L/2
        spring_bottom = -self.L/2
        spring_buffer = 0.1  # 10 cm buffer around spring
        
        # Find when magnet enters and exits the spring region (with buffer)
        enter_region = spring_top + spring_buffer
        exit_region = spring_bottom - spring_buffer
        
        # Find indices where magnet is in the interaction region
        in_region_mask = (results['position'] <= enter_region) & (results['position'] >= exit_region)
        interaction_indices = np.where(in_region_mask)[0]
        
        if len(interaction_indices) > 0:
            # Add time buffer (e.g., 0.05 seconds before and after)
            time_buffer = 0.05  # seconds
            start_time_idx = max(0, interaction_indices[0] - int(time_buffer / (results['time'][1] - results['time'][0])))
            end_time_idx = min(len(results['time']) - 1, interaction_indices[-1] + int(time_buffer / (results['time'][1] - results['time'][0])))
            
            # Create focused time window
            time_focused = results['time'][start_time_idx:end_time_idx+1]
            position_focused = results['position'][start_time_idx:end_time_idx+1]
            velocity_focused = results['velocity'][start_time_idx:end_time_idx+1]
            emf_focused = results['emf'][start_time_idx:end_time_idx+1]
            flux_focused = results['flux'][start_time_idx:end_time_idx+1]
            
            # Adjust extrema indices for focused window
            max_emf_idx_focused = max_emf_idx - start_time_idx if start_time_idx <= max_emf_idx <= end_time_idx else None
            min_emf_idx_focused = min_emf_idx - start_time_idx if start_time_idx <= min_emf_idx <= end_time_idx else None
            
            print(f"Focusing plots on spring interaction period:")
            print(f"  Time range: {time_focused[0]:.3f} - {time_focused[-1]:.3f} s")
            print(f"  Duration: {time_focused[-1] - time_focused[0]:.3f} s")
        else:
            # Fallback to full time series if no interaction detected
            time_focused = results['time']
            position_focused = results['position']
            velocity_focused = results['velocity']
            emf_focused = results['emf']
            flux_focused = results['flux']
            max_emf_idx_focused = max_emf_idx
            min_emf_idx_focused = min_emf_idx
            start_time_idx = 0
        
        # Create figure with subplots - FOCUSED VERSION
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Position vs Time (Focused)
        ax1.plot(time_focused, position_focused * 1000, 'b-', linewidth=2)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Position (mm)')
        ax1.set_title('Magnet Position vs Time (Spring Interaction Period)')
        ax1.grid(True, alpha=0.3)
        
        # Add spring boundaries
        ax1.axhline(-self.L/2 * 1000, color='red', linestyle='--', alpha=0.7, label='Spring bottom')
        ax1.axhline(self.L/2 * 1000, color='red', linestyle='--', alpha=0.7, label='Spring top')
        
        # Mark EMF extrema positions (if in focused window)
        if max_emf_idx_focused is not None and 0 <= max_emf_idx_focused < len(time_focused):
            ax1.scatter(time_focused[max_emf_idx_focused], position_focused[max_emf_idx_focused] * 1000, 
                       color='orange', s=100, marker='*', zorder=5, label='Max EMF position')
        if min_emf_idx_focused is not None and 0 <= min_emf_idx_focused < len(time_focused):
            ax1.scatter(time_focused[min_emf_idx_focused], position_focused[min_emf_idx_focused] * 1000, 
                       color='purple', s=100, marker='*', zorder=5, label='Min EMF position')
        ax1.legend()
        
        # Plot 2: Velocity vs Time (Focused)
        ax2.plot(time_focused, velocity_focused, 'g-', linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.set_title('Magnet Velocity vs Time (Spring Interaction Period)')
        ax2.grid(True, alpha=0.3)
        
        # Mark EMF extrema times (if in focused window)
        if max_emf_idx_focused is not None and 0 <= max_emf_idx_focused < len(time_focused):
            ax2.scatter(time_focused[max_emf_idx_focused], velocity_focused[max_emf_idx_focused], 
                       color='orange', s=100, marker='*', zorder=5, label='Max EMF time')
        if min_emf_idx_focused is not None and 0 <= min_emf_idx_focused < len(time_focused):
            ax2.scatter(time_focused[min_emf_idx_focused], velocity_focused[min_emf_idx_focused], 
                       color='purple', s=100, marker='*', zorder=5, label='Min EMF time')
        ax2.legend()
        
        # Plot 3: EMF vs Time (Main result - Focused)
        ax3.plot(time_focused, emf_focused * 1000, 'r-', linewidth=2)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Induced EMF (mV)')
        ax3.set_title('Induced EMF vs Time (Spring Interaction Period)')
        ax3.grid(True, alpha=0.3)
        
        # Mark EMF extrema (if in focused window)
        if max_emf_idx_focused is not None and 0 <= max_emf_idx_focused < len(time_focused):
            ax3.scatter(time_focused[max_emf_idx_focused], emf_focused[max_emf_idx_focused] * 1000, 
                       color='orange', s=150, marker='*', zorder=5, 
                       label=f'Max EMF: {emf_focused[max_emf_idx_focused]*1000:.1f} mV')
        if min_emf_idx_focused is not None and 0 <= min_emf_idx_focused < len(time_focused):
            ax3.scatter(time_focused[min_emf_idx_focused], emf_focused[min_emf_idx_focused] * 1000, 
                       color='purple', s=150, marker='*', zorder=5, 
                       label=f'Min EMF: {emf_focused[min_emf_idx_focused]*1000:.1f} mV')
        ax3.legend()
        
        # Plot 4: Magnetic Flux vs Time (Focused)
        ax4.plot(time_focused, flux_focused * 1e6, 'm-', linewidth=2)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Magnetic Flux (μWb)')
        ax4.set_title('Magnetic Flux vs Time (Spring Interaction Period)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(os.path.join(plots_folder, 'magnet_simulation_overview_focused.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create detailed EMF plot (Focused)
        plt.figure(figsize=(12, 8))
        plt.plot(time_focused, emf_focused * 1000, 'r-', linewidth=2, label='Induced EMF')
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Induced EMF (mV)', fontsize=12)
        plt.title('Induced EMF vs Time - Magnet Through Spring (Focused View)', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Mark EMF extrema with detailed annotations (if in focused window)
        if max_emf_idx_focused is not None and 0 <= max_emf_idx_focused < len(time_focused):
            plt.scatter(time_focused[max_emf_idx_focused], emf_focused[max_emf_idx_focused] * 1000, 
                       color='orange', s=150, marker='*', zorder=5, label='Maximum EMF')
            
            # Position relative to spring
            spring_top = self.L/2
            max_emf_from_top = spring_top - position_focused[max_emf_idx_focused]
            
            plt.annotate(f'Peak EMF: {emf_focused[max_emf_idx_focused]*1000:.2f} mV\nat t = {time_focused[max_emf_idx_focused]:.4f} s\n{max_emf_from_top*1000:.1f} mm from spring top',
                        xy=(time_focused[max_emf_idx_focused], emf_focused[max_emf_idx_focused] * 1000),
                        xytext=(time_focused[max_emf_idx_focused] + (time_focused[-1] - time_focused[0]) * 0.15, 
                               emf_focused[max_emf_idx_focused] * 1000 * 0.8),
                        arrowprops=dict(arrowstyle='->', color='orange'),
                        fontsize=10, ha='left', bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.3))
        
        if min_emf_idx_focused is not None and 0 <= min_emf_idx_focused < len(time_focused):
            plt.scatter(time_focused[min_emf_idx_focused], emf_focused[min_emf_idx_focused] * 1000, 
                       color='purple', s=150, marker='*', zorder=5, label='Minimum EMF')
            
            spring_top = self.L/2
            min_emf_from_top = spring_top - position_focused[min_emf_idx_focused]
            
            plt.annotate(f'Min EMF: {emf_focused[min_emf_idx_focused]*1000:.2f} mV\nat t = {time_focused[min_emf_idx_focused]:.4f} s\n{min_emf_from_top*1000:.1f} mm from spring top',
                        xy=(time_focused[min_emf_idx_focused], emf_focused[min_emf_idx_focused] * 1000),
                        xytext=(time_focused[min_emf_idx_focused] + (time_focused[-1] - time_focused[0]) * 0.15, 
                               emf_focused[min_emf_idx_focused] * 1000 * 0.8),
                        arrowprops=dict(arrowstyle='->', color='purple'),
                        fontsize=10, ha='left', bbox=dict(boxstyle="round,pad=0.3", facecolor="purple", alpha=0.3))
        
        plt.legend(fontsize=12)
        
        if save_plots:
            plt.savefig(os.path.join(plots_folder, 'emf_vs_time_focused.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create phase plot (EMF vs Position) - also focused
        plt.figure(figsize=(10, 8))
        plt.plot(position_focused * 1000, emf_focused * 1000, 'b-', linewidth=2)
        plt.xlabel('Position (mm)', fontsize=12)
        plt.ylabel('Induced EMF (mV)', fontsize=12)
        plt.title('EMF vs Position - Phase Plot (Spring Interaction)', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Add spring boundaries
        plt.axvline(-self.L/2 * 1000, color='red', linestyle='--', alpha=0.7, label='Spring bottom')
        plt.axvline(self.L/2 * 1000, color='red', linestyle='--', alpha=0.7, label='Spring top')
        
        # Mark EMF extrema on position plot (if in focused window)
        if max_emf_idx_focused is not None and 0 <= max_emf_idx_focused < len(time_focused):
            plt.scatter(position_focused[max_emf_idx_focused] * 1000, emf_focused[max_emf_idx_focused] * 1000, 
                       color='orange', s=150, marker='*', zorder=5, label='Maximum EMF')
        if min_emf_idx_focused is not None and 0 <= min_emf_idx_focused < len(time_focused):
            plt.scatter(position_focused[min_emf_idx_focused] * 1000, emf_focused[min_emf_idx_focused] * 1000, 
                       color='purple', s=150, marker='*', zorder=5, label='Minimum EMF')
        
        plt.legend()
        
        if save_plots:
            plt.savefig(os.path.join(plots_folder, 'emf_vs_position_focused.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
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
    Create comprehensive plots showing how coil length affects EMF characteristics.
    """
    # Create plots folder
    plots_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
    os.makedirs(plots_folder, exist_ok=True)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Plot 1: EMF vs Coil Length
    ax1 = plt.subplot(3, 3, 1)
    plt.loglog(spring_lengths, np.abs(max_emf_values) * 1000, 'ro-', linewidth=2, markersize=6, label='Max EMF')
    plt.loglog(spring_lengths, np.abs(min_emf_values) * 1000, 'bo-', linewidth=2, markersize=6, label='|Min EMF|')
    plt.loglog(spring_lengths, peak_to_peak_emf * 1000, 'go-', linewidth=2, markersize=6, label='Peak-to-Peak EMF')
    plt.xlabel('Coil Length (m)')
    plt.ylabel('EMF (mV)')
    plt.title('EMF Magnitude vs Coil Length')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Interaction Time vs Coil Length
    ax2 = plt.subplot(3, 3, 2)
    plt.loglog(spring_lengths, interaction_times, 'mo-', linewidth=2, markersize=6)
    plt.xlabel('Coil Length (m)')
    plt.ylabel('Interaction Time (s)')
    plt.title('EMF Interaction Duration vs Coil Length')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Resistance vs Coil Length
    ax3 = plt.subplot(3, 3, 3)
    plt.loglog(spring_lengths, total_resistance * 1000, 'co-', linewidth=2, markersize=6)
    plt.xlabel('Coil Length (m)')
    plt.ylabel('Resistance (mΩ)')
    plt.title('Coil Resistance vs Length')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Maximum Current vs Coil Length
    ax4 = plt.subplot(3, 3, 4)
    plt.loglog(spring_lengths, max_current_values * 1000, 'yo-', linewidth=2, markersize=6)
    plt.xlabel('Coil Length (m)')
    plt.ylabel('Max Current (mA)')
    plt.title('Maximum Current vs Coil Length')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Energy Dissipated vs Coil Length
    ax5 = plt.subplot(3, 3, 5)
    plt.loglog(spring_lengths, energy_dissipated * 1e6, 'ko-', linewidth=2, markersize=6)
    plt.xlabel('Coil Length (m)')
    plt.ylabel('Energy Dissipated (μJ)')
    plt.title('Energy Dissipation vs Coil Length')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: EMF Position Analysis
    ax6 = plt.subplot(3, 3, 6)
    # Convert positions to fraction of coil length from top
    max_emf_fractions = []
    min_emf_fractions = []
    
    for i, length in enumerate(spring_lengths):
        if not np.isnan(max_emf_positions[i]):
            spring_top = length/2
            max_pos_from_top = spring_top - max_emf_positions[i]
            max_emf_fractions.append(max_pos_from_top / length)
        else:
            max_emf_fractions.append(np.nan)
            
        if not np.isnan(min_emf_positions[i]):
            spring_top = length/2
            min_pos_from_top = spring_top - min_emf_positions[i]
            min_emf_fractions.append(min_pos_from_top / length)
        else:
            min_emf_fractions.append(np.nan)
    
    plt.semilogx(spring_lengths, max_emf_fractions, 'ro-', linewidth=2, markersize=6, label='Max EMF Position')
    plt.semilogx(spring_lengths, min_emf_fractions, 'bo-', linewidth=2, markersize=6, label='Min EMF Position')
    plt.xlabel('Coil Length (m)')
    plt.ylabel('Position (fraction from top)')
    plt.title('EMF Extrema Positions vs Coil Length')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 7: EMF per unit length
    ax7 = plt.subplot(3, 3, 7)
    emf_per_meter = peak_to_peak_emf / spring_lengths
    plt.loglog(spring_lengths, emf_per_meter * 1000, 'ro-', linewidth=2, markersize=6)
    plt.xlabel('Coil Length (m)')
    plt.ylabel('EMF per Unit Length (mV/m)')
    plt.title('EMF Efficiency vs Coil Length')
    plt.grid(True, alpha=0.3)
    
    # Plot 8: Number of turns vs Length
    ax8 = plt.subplot(3, 3, 8)
    num_turns = spring_lengths * turn_density
    plt.loglog(spring_lengths, num_turns, 'go-', linewidth=2, markersize=6)
    plt.xlabel('Coil Length (m)')
    plt.ylabel('Number of Turns')
    plt.title(f'Number of Turns (Density: {turn_density:.1f} turns/m)')
    plt.grid(True, alpha=0.3)
    
    # Plot 9: Power vs Coil Length
    ax9 = plt.subplot(3, 3, 9)
    # Peak power = EMF²/R
    peak_power = peak_to_peak_emf**2 / total_resistance
    plt.loglog(spring_lengths, peak_power * 1e6, 'mo-', linewidth=2, markersize=6)
    plt.xlabel('Coil Length (m)')
    plt.ylabel('Peak Power (μW)')
    plt.title('Peak Power vs Coil Length')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, 'coil_length_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a summary plot with key trends
    plt.figure(figsize=(14, 10))
    
    # Subplot 1: EMF trends
    plt.subplot(2, 2, 1)
    plt.loglog(spring_lengths, np.abs(max_emf_values) * 1000, 'ro-', linewidth=3, markersize=8, label='Max EMF')
    plt.loglog(spring_lengths, peak_to_peak_emf * 1000, 'go-', linewidth=3, markersize=8, label='Peak-to-Peak EMF')
    plt.xlabel('Coil Length (m)', fontsize=12)
    plt.ylabel('EMF (mV)', fontsize=12)
    plt.title('EMF Scaling with Coil Length', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Subplot 2: Time and resistance scaling
    plt.subplot(2, 2, 2)
    ax2a = plt.gca()
    line1 = ax2a.loglog(spring_lengths, interaction_times, 'mo-', linewidth=3, markersize=8, label='Interaction Time')
    ax2a.set_xlabel('Coil Length (m)', fontsize=12)
    ax2a.set_ylabel('Interaction Time (s)', fontsize=12, color='m')
    ax2a.tick_params(axis='y', labelcolor='m')
    
    ax2b = ax2a.twinx()
    line2 = ax2b.loglog(spring_lengths, total_resistance * 1000, 'co-', linewidth=3, markersize=8, label='Resistance')
    ax2b.set_ylabel('Resistance (mΩ)', fontsize=12, color='c')
    ax2b.tick_params(axis='y', labelcolor='c')
    
    plt.title('Time & Resistance Scaling', fontsize=14)
    ax2a.grid(True, alpha=0.3)
    
    # Subplot 3: Energy and power
    plt.subplot(2, 2, 3)
    ax3a = plt.gca()
    line3 = ax3a.loglog(spring_lengths, energy_dissipated * 1e6, 'ko-', linewidth=3, markersize=8, label='Energy')
    ax3a.set_xlabel('Coil Length (m)', fontsize=12)
    ax3a.set_ylabel('Energy Dissipated (μJ)', fontsize=12, color='k')
    ax3a.tick_params(axis='y', labelcolor='k')
    
    ax3b = ax3a.twinx()
    line4 = ax3b.loglog(spring_lengths, peak_power * 1e6, 'ro-', linewidth=3, markersize=8, label='Peak Power')
    ax3b.set_ylabel('Peak Power (μW)', fontsize=12, color='r')
    ax3b.tick_params(axis='y', labelcolor='r')
    
    plt.title('Energy & Power Scaling', fontsize=14)
    ax3a.grid(True, alpha=0.3)
    
    # Subplot 4: EMF efficiency
    plt.subplot(2, 2, 4)
    plt.loglog(spring_lengths, emf_per_meter * 1000, 'ro-', linewidth=3, markersize=8)
    plt.xlabel('Coil Length (m)', fontsize=12)
    plt.ylabel('EMF per Unit Length (mV/m)', fontsize=12)
    plt.title('EMF Efficiency vs Coil Length', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, 'coil_length_summary.png'), dpi=300, bbox_inches='tight')
    plt.show()


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
    Create overlay plots comparing EMF curves for different coil lengths.
    """
    # Create plots folder
    plots_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
    os.makedirs(plots_folder, exist_ok=True)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    
    # Plot 1: EMF vs Time (Focused on interaction periods)
    ax1 = plt.subplot(2, 3, 1)
    
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
    
    plt.xlabel('Time from interaction start (s)')
    plt.ylabel('Induced EMF (mV)')
    plt.title('EMF vs Time - Coil Length Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: EMF vs Position (relative to spring)
    ax2 = plt.subplot(2, 3, 2)
    
    for i, length in enumerate(lengths_to_compare):
        results = all_results[length]
        sim = all_sims[length]
        
        # Convert position to fraction of spring length from top
        spring_top = sim.L/2
        position_relative = (spring_top - results['position']) / sim.L
        
        # Only plot when magnet is near the spring
        mask = (position_relative >= -0.5) & (position_relative <= 1.5)
        
        plt.plot(position_relative[mask], results['emf'][mask] * 1000, 
                color=colors[i], linestyle=line_styles[i], linewidth=2, 
                label=f'L = {length} m')
    
    plt.xlabel('Position (fraction of spring length from top)')
    plt.ylabel('Induced EMF (mV)')
    plt.title('EMF vs Relative Position')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add coil boundaries
    if hasattr(sim, 'L'):
        spring_extent = sim.L/2 * 1000  # mm
        plt.axvline(spring_extent, color=colors[i], linestyle=':', alpha=0.5)
        plt.axvline(-spring_extent, color=colors[i], linestyle=':', alpha=0.5)
    
    # Plot 3: Peak EMF comparison
    ax3 = plt.subplot(2, 3, 3)
    
    seg_max = np.max(all_results[length]['emf']) * 1000
    seg_min = np.min(all_results[length]['emf']) * 1000
    cont_max = np.max(all_results[length]['emf']) * 1000
    cont_min = np.min(all_results[length]['emf']) * 1000
    
    x_pos = [0, 1]
    max_emfs = [seg_max, cont_max]
    min_emfs = [abs(seg_min), abs(cont_min)]
    peak_to_peak = [seg_max - seg_min, cont_max - cont_min]
    
    width = 0.25
    plt.bar([x - width for x in x_pos], max_emfs, width, label='Max EMF', alpha=0.7, color='orange')
    plt.bar(x_pos, min_emfs, width, label='|Min EMF|', alpha=0.7, color='purple')
    plt.bar([x + width for x in x_pos], peak_to_peak, width, label='Peak-to-Peak', alpha=0.7, color='green')
    
    plt.xlabel('Coil Type')
    plt.ylabel('EMF (mV)')
    plt.title('EMF Magnitude Comparison')
    plt.xticks(x_pos, ['Segmented', 'Continuous'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Power comparison
    ax4 = plt.subplot(2, 3, 4)
    
    seg_power = all_results[length]['emf']**2 / sim.resistance
    cont_power = all_results[length]['emf']**2 / sim.resistance
    
    seg_peak_power = np.max(seg_power) * 1e6  # μW
    cont_peak_power = np.max(cont_power) * 1e6
    
    seg_avg_power = np.mean(seg_power) * 1e6
    cont_avg_power = np.mean(cont_power) * 1e6
    
    x_pos = [0, 1]
    peak_powers = [seg_peak_power, cont_peak_power]
    avg_powers = [seg_avg_power, cont_avg_power]
    
    width = 0.35
    plt.bar([x - width/2 for x in x_pos], peak_powers, width, label='Peak Power', alpha=0.7)
    plt.bar([x + width/2 for x in x_pos], avg_powers, width, label='Avg Power', alpha=0.7)
    
    plt.xlabel('Coil Type')
    plt.ylabel('Power (μW)')
    plt.title('Power Generation Comparison')
    plt.xticks(x_pos, ['Segmented', 'Continuous'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Resistance comparison
    ax5 = plt.subplot(2, 3, 5)
    
    resistances = [sim.resistance * 1000, sim.resistance * 1000]
    plt.bar(x_pos, resistances, color=['red', 'blue'], alpha=0.7)
    plt.xlabel('Coil Type')
    plt.ylabel('Resistance (mΩ)')
    plt.title('Coil Resistance Comparison')
    plt.xticks(x_pos, ['Segmented', 'Continuous'])
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Coil geometry visualization
    ax6 = plt.subplot(2, 3, 6)
    
    # Draw coil
    if hasattr(sim, 'L'):
        cont_start = -sim.L/2 * 1000
        cont_end = sim.L/2 * 1000
        plt.plot([cont_start, cont_end], [0, 0], 'b-', linewidth=8, alpha=0.7, label='Continuous coil')
    
    plt.ylim(-0.5, 1.5)
    plt.xlabel('Position (mm)')
    plt.title('Coil Geometry Comparison')
    plt.yticks([0, 1], ['Continuous', 'Segmented'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, 'coil_length_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print(f"\nComparison Summary:")
    print(f"Segmented coil - Max EMF: {seg_max:.2f} mV, Peak-to-Peak: {seg_max-seg_min:.2f} mV")
    print(f"Continuous coil - Max EMF: {cont_max:.2f} mV, Peak-to-Peak: {cont_max-cont_min:.2f} mV")
    print(f"EMF Ratio (Segmented/Continuous): {(seg_max-seg_min)/(cont_max-cont_min):.3f}")


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
    seg_power = segmented_results['emf']**2 / segmented_sim.resistance
    cont_power = continuous_results['emf']**2 / continuous_sim.resistance
    
    seg_peak_power = np.max(seg_power) * 1e6
    cont_peak_power = np.max(cont_power) * 1e6
    
    print(f"\nPower Comparison:")
    print(f"  Segmented peak power: {seg_peak_power:.2f} μW")
    print(f"  Continuous peak power: {cont_peak_power:.2f} μW")
    print(f"  Ratio: {seg_peak_power/cont_peak_power:.3f}")
    
    # Geometry summary
    if hasattr(segmented_sim, 'segment_centers'):
        print(f"\nGeometry Summary:")
        print(f"  Segmented coil:")
        print(f"    - {segmented_sim.num_segments} segments of {segmented_sim.segment_length*1000:.1f} mm each")
        print(f"    - {segmented_sim.gap_length*1000:.1f} mm gaps between segments")
        print(f"    - {segmented_sim.turns_per_segment} turns per segment")
        print(f"  Continuous coil:")
        print(f"    - Single coil of {continuous_sim.L*1000:.1f} mm")
        print(f"    - {continuous_sim.N} total turns")


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
    Create comprehensive plots showing how gap length affects EMF characteristics.
    """
    plots_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
    os.makedirs(plots_folder, exist_ok=True)
    
    # Define colors for different gap lengths
    colors = plt.cm.viridis(np.linspace(0, 1, len(gap_lengths)))
    continuous_color = 'red'
    
    # Create main comparison figure
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: EMF vs Time overlay
    ax1 = plt.subplot(3, 3, 1)
    
    # Plot continuous coil first
    if 'continuous' in all_results and all_results['continuous'] is not None:
        results = all_results['continuous']
        mask = (results['position'] >= -0.6) & (results['position'] <= 0.6)
        if np.any(mask):
            time_focused = results['time'][mask]
            emf_focused = results['emf'][mask]
            time_shifted = time_focused - time_focused[0]
            ax1.plot(time_shifted, emf_focused * 1000, color=continuous_color, 
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
                ax1.plot(time_shifted, emf_focused * 1000, color=colors[i], 
                        linewidth=2, label=f'Gap: {gap_length*1000:.1f}mm')
    
    ax1.set_xlabel('Time from interaction start (s)')
    ax1.set_ylabel('Induced EMF (mV)')
    ax1.set_title('EMF vs Time - Gap Length Comparison')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2: EMF vs Position overlay
    ax2 = plt.subplot(3, 3, 2)
    
    # Plot continuous coil
    if 'continuous' in all_results and all_results['continuous'] is not None:
        results = all_results['continuous']
        position_mm = results['position'] * 1000
        mask = (np.abs(position_mm) <= 750)
        ax2.plot(position_mm[mask], results['emf'][mask] * 1000, 
                color=continuous_color, linewidth=3, label='Continuous')
    
    # Plot segmented coils
    for i, gap_length in enumerate(gap_lengths):
        if gap_length in all_results and all_results[gap_length] is not None:
            results = all_results[gap_length]
            position_mm = results['position'] * 1000
            mask = (np.abs(position_mm) <= 750)
            ax2.plot(position_mm[mask], results['emf'][mask] * 1000, 
                    color=colors[i], linewidth=2, label=f'{gap_length*1000:.1f}mm')
    
    ax2.set_xlabel('Position from center (mm)')
    ax2.set_ylabel('Induced EMF (mV)')
    ax2.set_title('EMF vs Position - Gap Length Comparison')
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 3: Peak EMF vs Gap Length
    ax3 = plt.subplot(3, 3, 3)
    
    gap_mm = [g * 1000 for g in gap_lengths]
    max_emfs = []
    min_emfs = []
    peak_to_peak_emfs = []
    
    for gap_length in gap_lengths:
        if gap_length in all_results and all_results[gap_length] is not None:
            results = all_results[gap_length]
            max_emf = np.max(results['emf']) * 1000
            min_emf = np.min(results['emf']) * 1000
            max_emfs.append(max_emf)
            min_emfs.append(abs(min_emf))
            peak_to_peak_emfs.append(max_emf - min_emf)
        else:
            max_emfs.append(np.nan)
            min_emfs.append(np.nan)
            peak_to_peak_emfs.append(np.nan)
    
    ax3.plot(gap_mm, max_emfs, 'ro-', linewidth=2, markersize=8, label='Max EMF')
    ax3.plot(gap_mm, min_emfs, 'bo-', linewidth=2, markersize=8, label='|Min EMF|')
    ax3.plot(gap_mm, peak_to_peak_emfs, 'go-', linewidth=2, markersize=8, label='Peak-to-Peak')
    
    # Add continuous coil reference line
    if 'continuous' in all_results and all_results['continuous'] is not None:
        cont_results = all_results['continuous']
        cont_max = np.max(cont_results['emf']) * 1000
        cont_min = np.min(cont_results['emf']) * 1000
        cont_p2p = cont_max - cont_min
        ax3.axhline(cont_max, color='red', linestyle='--', alpha=0.7, label='Continuous Max')
        ax3.axhline(cont_p2p, color='green', linestyle='--', alpha=0.7, label='Continuous P2P')
    
    ax3.set_xlabel('Gap Length (mm)')
    ax3.set_ylabel('EMF (mV)')
    ax3.set_title('EMF Magnitude vs Gap Length')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Resistance vs Gap Length
    ax4 = plt.subplot(3, 3, 4)
    
    resistances = []
    for gap_length in gap_lengths:
        if gap_length in all_sims and all_sims[gap_length] is not None:
            resistances.append(all_sims[gap_length].resistance * 1000)
        else:
            resistances.append(np.nan)
    
    ax4.plot(gap_mm, resistances, 'co-', linewidth=2, markersize=8)
    
    # Add continuous coil reference
    if 'continuous' in all_sims and all_sims['continuous'] is not None:
        cont_resistance = all_sims['continuous'].resistance * 1000
        ax4.axhline(cont_resistance, color='red', linestyle='--', alpha=0.7, 
                   label=f'Continuous: {cont_resistance:.1f} mΩ')
    
    ax4.set_xlabel('Gap Length (mm)')
    ax4.set_ylabel('Resistance (mΩ)')
    ax4.set_title('Coil Resistance vs Gap Length')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Plot 5: Effective coil length vs Gap Length
    ax5 = plt.subplot(3, 3, 5)
    
    effective_coil_lengths = []
    total_gap_lengths = []
    
    for gap_length in gap_lengths:
        if gap_length in all_sims and all_sims[gap_length] is not None:
            sim = all_sims[gap_length]
            total_gap = (sim.num_segments - 1) * gap_length * 1000  # mm
            effective_coil = sim.segment_length * sim.num_segments * 1000  # mm
            effective_coil_lengths.append(effective_coil)
            total_gap_lengths.append(total_gap)
        else:
            effective_coil_lengths.append(np.nan)
            total_gap_lengths.append(np.nan)
    
    ax5.plot(gap_mm, effective_coil_lengths, 'mo-', linewidth=2, markersize=8, label='Effective coil length')
    ax5.plot(gap_mm, total_gap_lengths, 'yo-', linewidth=2, markersize=8, label='Total gap length')
    
    ax5.set_xlabel('Gap Length (mm)')
    ax5.set_ylabel('Length (mm)')
    ax5.set_title('Coil Geometry vs Gap Length')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # Plot 6: EMF Efficiency (EMF per effective coil length)
    ax6 = plt.subplot(3, 3, 6)
    
    emf_efficiency = []
    for i, gap_length in enumerate(gap_lengths):
        if (gap_length in all_results and all_results[gap_length] is not None and
            not np.isnan(peak_to_peak_emfs[i]) and not np.isnan(effective_coil_lengths[i])):
            efficiency = peak_to_peak_emfs[i] / (effective_coil_lengths[i] / 1000)  # mV per meter
            emf_efficiency.append(efficiency)
        else:
            emf_efficiency.append(np.nan)
    
    ax6.plot(gap_mm, emf_efficiency, 'ko-', linewidth=2, markersize=8)
    
    # Add continuous coil reference
    if 'continuous' in all_results and all_sims['continuous'] is not None:
        cont_results = all_results['continuous']
        cont_sim = all_sims['continuous']
        cont_p2p = (np.max(cont_results['emf']) - np.min(cont_results['emf'])) * 1000
        cont_efficiency = cont_p2p / cont_sim.L  # mV per meter
        ax6.axhline(cont_efficiency, color='red', linestyle='--', alpha=0.7, 
                   label=f'Continuous: {cont_efficiency:.1f} mV/m')
    
    ax6.set_xlabel('Gap Length (mm)')
    ax6.set_ylabel('EMF Efficiency (mV/m of coil)')
    ax6.set_title('EMF Efficiency vs Gap Length')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    # Plot 7: Power vs Gap Length
    ax7 = plt.subplot(3, 3, 7)
    
    peak_powers = []
    for i, gap_length in enumerate(gap_lengths):
        if gap_length in all_results and all_results[gap_length] is not None:
            results = all_results[gap_length]
            sim = all_sims[gap_length]
            power = results['emf']**2 / sim.resistance
            peak_power = np.max(power) * 1e6  # μW
            peak_powers.append(peak_power)
        else:
            peak_powers.append(np.nan)
    
    ax7.plot(gap_mm, peak_powers, 'ro-', linewidth=2, markersize=8)
    
    # Add continuous coil reference
    if 'continuous' in all_results and all_sims['continuous'] is not None:
        cont_results = all_results['continuous']
        cont_sim = all_sims['continuous']
        cont_power = cont_results['emf']**2 / cont_sim.resistance
        cont_peak_power = np.max(cont_power) * 1e6
        ax7.axhline(cont_peak_power, color='red', linestyle='--', alpha=0.7, 
                   label=f'Continuous: {cont_peak_power:.1f} μW')
    
    ax7.set_xlabel('Gap Length (mm)')
    ax7.set_ylabel('Peak Power (μW)')
    ax7.set_title('Peak Power vs Gap Length')
    ax7.grid(True, alpha=0.3)
    ax7.legend()
    
    # Plot 8: Segment visualization for selected gap lengths
    ax8 = plt.subplot(3, 3, 8)
    
    # Show geometry for a few representative gap lengths
    representative_gaps = [gap_lengths[0], gap_lengths[len(gap_lengths)//2], gap_lengths[-1]]
    y_positions = [0, 1, 2]
    
    for i, gap_length in enumerate(representative_gaps):
        if gap_length in all_sims and all_sims[gap_length] is not None:
            sim = all_sims[gap_length]
            boundaries = sim.get_segment_boundaries()
            y_pos = y_positions[i]
            
            # Draw coil segments
            for j, (start, end) in enumerate(boundaries):
                ax8.plot([start*1000, end*1000], [y_pos, y_pos], 'o-', linewidth=6, 
                        markersize=8, alpha=0.7, 
                        label=f'Gap: {gap_length*1000:.1f}mm' if j == 0 else "")
            
            # Draw gaps
            for j in range(len(boundaries)-1):
                gap_start = boundaries[j][1] * 1000
                gap_end = boundaries[j+1][0] * 1000
                ax8.plot([gap_start, gap_end], [y_pos, y_pos], '--', linewidth=2, alpha=0.5)
    
    ax8.set_xlabel('Position (mm)')
    ax8.set_ylabel('Configuration')
    ax8.set_title('Coil Segment Layouts')
    ax8.set_yticks(y_positions)
    ax8.set_yticklabels([f'{g*1000:.1f}mm' for g in representative_gaps])
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Plot 9: Summary metrics
    ax9 = plt.subplot(3, 3, 9)
    
    # Calculate relative performance vs continuous coil
    if 'continuous' in all_results and all_results['continuous'] is not None:
        cont_p2p = (np.max(all_results['continuous']['emf']) - 
                   np.min(all_results['continuous']['emf'])) * 1000
        
        relative_performance = []
        for p2p in peak_to_peak_emfs:
            if not np.isnan(p2p):
                relative_performance.append(p2p / cont_p2p)
            else:
                relative_performance.append(np.nan)
        
        ax9.plot(gap_mm, relative_performance, 'go-', linewidth=3, markersize=10)
        ax9.axhline(1.0, color='red', linestyle='--', alpha=0.7, 
                   label='Continuous coil performance')
        
        ax9.set_xlabel('Gap Length (mm)')
        ax9.set_ylabel('Relative EMF Performance')
        ax9.set_title('Performance vs Continuous Coil')
        ax9.grid(True, alpha=0.3)
        ax9.legend()
        valid_performance = [p for p in relative_performance if not np.isnan(p)]
        if valid_performance:
            ax9.set_ylim(0, max(valid_performance) * 1.1)
        else:
            ax9.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, 'gap_length_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print(f"\nGap Length Analysis Summary:")
    print(f"{'Gap (mm)':<8} {'Max EMF (mV)':<12} {'P2P EMF (mV)':<12} {'Resistance (mΩ)':<15} {'Peak Power (μW)':<15}")
    print("-" * 70)
    
    for i, gap_length in enumerate(gap_lengths):
        if gap_length in all_results and all_results[gap_length] is not None:
            gap_mm_val = gap_length * 1000
            max_emf = max_emfs[i] if i < len(max_emfs) else 0
            p2p_emf = peak_to_peak_emfs[i] if i < len(peak_to_peak_emfs) else 0
            resistance = resistances[i] if i < len(resistances) else 0
            power = peak_powers[i] if i < len(peak_powers) else 0
            
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
    """Create comparison plots between segmented and continuous coils."""
    plots_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
    os.makedirs(plots_folder, exist_ok=True)
    
    # Main comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: EMF vs Time
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
            ax1.plot(time_shifted, emf_focused * 1000, color=color, linestyle=linestyle, 
                    linewidth=3, label=label)
    
    ax1.set_xlabel('Time from interaction start (s)')
    ax1.set_ylabel('Induced EMF (mV)')
    ax1.set_title('EMF vs Time - Segmented vs Continuous')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: EMF vs Position
    for results, label, color, linestyle in [
        (segmented_results, 'Segmented', 'red', '-'),
        (continuous_results, 'Continuous', 'blue', '--')
    ]:
        position_mm = results['position'] * 1000
        mask = (np.abs(position_mm) <= 750)  # Focus on ±75cm
        ax2.plot(position_mm[mask], results['emf'][mask] * 1000, 
                color=color, linestyle=linestyle, linewidth=3, label=label)
    
    # Add segment boundaries for segmented coil
    boundaries = segmented_sim.get_segment_boundaries()
    for i, (start, end) in enumerate(boundaries):
        ax2.axvspan(start*1000, end*1000, alpha=0.2, color='red', 
                   label='Coil segments' if i == 0 else "")
    
    ax2.set_xlabel('Position from center (mm)')
    ax2.set_ylabel('Induced EMF (mV)')
    ax2.set_title('EMF vs Position')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Peak values comparison
    seg_max = np.max(segmented_results['emf']) * 1000
    seg_min = np.min(segmented_results['emf']) * 1000
    cont_max = np.max(continuous_results['emf']) * 1000
    cont_min = np.min(continuous_results['emf']) * 1000
    
    categories = ['Max EMF', '|Min EMF|', 'Peak-to-Peak']
    segmented_values = [seg_max, abs(seg_min), seg_max - seg_min]
    continuous_values = [cont_max, abs(cont_min), cont_max - cont_min]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax3.bar(x - width/2, segmented_values, width, label='Segmented', alpha=0.7, color='red')
    ax3.bar(x + width/2, continuous_values, width, label='Continuous', alpha=0.7, color='blue')
    
    ax3.set_xlabel('EMF Metric')
    ax3.set_ylabel('EMF (mV)')
    ax3.set_title('EMF Magnitude Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Coil geometry visualization
    ax4.set_xlim(-600, 600)
    ax4.set_ylim(-0.5, 1.5)
    
    # Draw segmented coil
    for i, (start, end) in enumerate(boundaries):
        ax4.plot([start*1000, end*1000], [1, 1], 'ro-', linewidth=8, 
                markersize=10, alpha=0.7, label='Coil segments' if i == 0 else "")
    
    # Draw gaps
    for i in range(len(boundaries)-1):
        gap_start = boundaries[i][1] * 1000
        gap_end = boundaries[i+1][0] * 1000
        ax4.plot([gap_start, gap_end], [1, 1], 'r--', linewidth=2, alpha=0.5,
                label='Gaps' if i == 0 else "")
    
    # Draw continuous coil
    cont_start = -continuous_sim.L/2 * 1000
    cont_end = continuous_sim.L/2 * 1000
    ax4.plot([cont_start, cont_end], [0, 0], 'b-', linewidth=8, alpha=0.7, 
            label='Continuous coil')
    
    ax4.set_xlabel('Position (mm)')
    ax4.set_ylabel('Coil Type')
    ax4.set_title('Coil Geometry Comparison')
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['Continuous', 'Segmented'])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, 'segmented_vs_continuous.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
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
    Create overlay plots comparing EMF curves for different coil lengths.
    """
    # Create plots folder
    plots_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
    os.makedirs(plots_folder, exist_ok=True)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    
    # Plot 1: EMF vs Time (Focused on interaction periods)
    ax1 = plt.subplot(2, 3, 1)
    
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
    
    plt.xlabel('Time from interaction start (s)')
    plt.ylabel('Induced EMF (mV)')
    plt.title('EMF vs Time - Coil Length Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: EMF vs Position (relative to spring)
    ax2 = plt.subplot(2, 3, 2)
    
    for i, length in enumerate(lengths_to_compare):
        results = all_results[length]
        sim = all_sims[length]
        
        # Convert position to fraction of spring length from top
        spring_top = sim.L/2
        position_relative = (spring_top - results['position']) / sim.L
        
        # Only plot when magnet is near the spring
        mask = (position_relative >= -0.5) & (position_relative <= 1.5)
        
        plt.plot(position_relative[mask], results['emf'][mask] * 1000, 
                color=colors[i], linestyle=line_styles[i], linewidth=2, 
                label=f'L = {length} m')
    
    plt.xlabel('Position (fraction of spring length from top)')
    plt.ylabel('Induced EMF (mV)')
    plt.title('EMF vs Relative Position')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add coil boundaries
    if hasattr(sim, 'L'):
        spring_extent = sim.L/2 * 1000  # mm
        plt.axvline(spring_extent, color=colors[i], linestyle=':', alpha=0.5)
        plt.axvline(-spring_extent, color=colors[i], linestyle=':', alpha=0.5)
    
    # Plot 3: Peak EMF comparison
    ax3 = plt.subplot(2, 3, 3)
    
    seg_max = np.max(all_results[length]['emf']) * 1000
    seg_min = np.min(all_results[length]['emf']) * 1000
    cont_max = np.max(all_results[length]['emf']) * 1000
    cont_min = np.min(all_results[length]['emf']) * 1000
    
    x_pos = [0, 1]
    max_emfs = [seg_max, cont_max]
    min_emfs = [abs(seg_min), abs(cont_min)]
    peak_to_peak = [seg_max - seg_min, cont_max - cont_min]
    
    width = 0.25
    plt.bar([x - width for x in x_pos], max_emfs, width, label='Max EMF', alpha=0.7, color='orange')
    plt.bar(x_pos, min_emfs, width, label='|Min EMF|', alpha=0.7, color='purple')
    plt.bar([x + width for x in x_pos], peak_to_peak, width, label='Peak-to-Peak', alpha=0.7, color='green')
    
    plt.xlabel('Coil Type')
    plt.ylabel('EMF (mV)')
    plt.title('EMF Magnitude Comparison')
    plt.xticks(x_pos, ['Segmented', 'Continuous'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Power comparison
    ax4 = plt.subplot(2, 3, 4)
    
    seg_power = all_results[length]['emf']**2 / sim.resistance
    cont_power = all_results[length]['emf']**2 / sim.resistance
    
    seg_peak_power = np.max(seg_power) * 1e6  # μW
    cont_peak_power = np.max(cont_power) * 1e6
    
    seg_avg_power = np.mean(seg_power) * 1e6
    cont_avg_power = np.mean(cont_power) * 1e6
    
    x_pos = [0, 1]
    peak_powers = [seg_peak_power, cont_peak_power]
    avg_powers = [seg_avg_power, cont_avg_power]
    
    width = 0.35
    plt.bar([x - width/2 for x in x_pos], peak_powers, width, label='Peak Power', alpha=0.7)
    plt.bar([x + width/2 for x in x_pos], avg_powers, width, label='Avg Power', alpha=0.7)
    
    plt.xlabel('Coil Type')
    plt.ylabel('Power (μW)')
    plt.title('Power Generation Comparison')
    plt.xticks(x_pos, ['Segmented', 'Continuous'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Resistance comparison
    ax5 = plt.subplot(2, 3, 5)
    
    resistances = [sim.resistance * 1000, sim.resistance * 1000]
    plt.bar(x_pos, resistances, color=['red', 'blue'], alpha=0.7)
    plt.xlabel('Coil Type')
    plt.ylabel('Resistance (mΩ)')
    plt.title('Coil Resistance Comparison')
    plt.xticks(x_pos, ['Segmented', 'Continuous'])
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Coil geometry visualization
    ax6 = plt.subplot(2, 3, 6)
    
    # Draw coil
    if hasattr(sim, 'L'):
        cont_start = -sim.L/2 * 1000
        cont_end = sim.L/2 * 1000
        plt.plot([cont_start, cont_end], [0, 0], 'b-', linewidth=8, alpha=0.7, label='Continuous coil')
    
    plt.ylim(-0.5, 1.5)
    plt.xlabel('Position (mm)')
    plt.title('Coil Geometry Comparison')
    plt.yticks([0, 1], ['Continuous', 'Segmented'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, 'coil_length_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()


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
    seg_power = segmented_results['emf']**2 / segmented_sim.resistance
    cont_power = continuous_results['emf']**2 / continuous_sim.resistance
    
    seg_peak_power = np.max(seg_power) * 1e6
    cont_peak_power = np.max(cont_power) * 1e6
    
    print(f"\nPower Comparison:")
    print(f"  Segmented peak power: {seg_peak_power:.2f} μW")
    print(f"  Continuous peak power: {cont_peak_power:.2f} μW")
    print(f"  Ratio: {seg_peak_power/cont_peak_power:.3f}")
    
    # Geometry summary
    if hasattr(segmented_sim, 'segment_centers'):
        print(f"\nGeometry Summary:")
        print(f"  Segmented coil:")
        print(f"    - {segmented_sim.num_segments} segments of {segmented_sim.segment_length*1000:.1f} mm each")
        print(f"    - {segmented_sim.gap_length*1000:.1f} mm gaps between segments")
        print(f"    - {segmented_sim.turns_per_segment} turns per segment")
        print(f"  Continuous coil:")
        print(f"    - Single coil of {continuous_sim.L*1000:.1f} mm")
        print(f"    - {continuous_sim.N} total turns")


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


if __name__ == "__main__":
    import sys
    
    # Check command line arguments for different analysis types
    if len(sys.argv) > 1:
        if sys.argv[1] == "length_analysis":
            print("Running comprehensive coil length analysis...")
            results = analyze_coil_length_effects()
        elif sys.argv[1] == "compare":
            print("Running coil length comparison...")
            results = compare_coil_lengths()
        elif sys.argv[1] == "segmented":
            print("Running segmented vs continuous coil comparison...")
            results = compare_segmented_vs_continuous_coils()
        elif sys.argv[1] == "gap_analysis":
            print("Running gap length analysis...")
            results = analyze_gap_length_effects()
        elif sys.argv[1] == "emf_plot":
            print("Generating EMF vs time plot only...")
            results = show_emf_time_plot_only()
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Available options:")
            print("  python simulate.py                    - Run single magnet simulation")
            print("  python simulate.py length_analysis    - Run comprehensive coil length analysis")
            print("  python simulate.py compare            - Compare EMF curves for 0.1m, 1m, 10m coils")
            print("  python simulate.py segmented          - Compare segmented vs continuous coils")
            print("  python simulate.py gap_analysis       - Analyze effects of gap length between segments")
            print("  python simulate.py emf_plot          - Show EMF vs time plot and coil transit time analysis")
    else:
        print("Running single magnet simulation...")
        print("Available options:")
        print("  python simulate.py                    - Run single magnet simulation")
        print("  python simulate.py length_analysis    - Run comprehensive coil length analysis") 
        print("  python simulate.py compare            - Compare EMF curves for 0.1m, 1m, 10m coils")
        print("  python simulate.py segmented          - Compare segmented vs continuous coils")
        print("  python simulate.py gap_analysis       - Analyze effects of gap length between segments")
        print("  python simulate.py emf_plot          - Show scientific EMF vs time plot only")
        results = run_magnet_simulation()
