import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from dataclasses import dataclass
from typing import List, Tuple, Dict
import math

@dataclass
class SoilLayer:
    """Represents a soil/rock layer with geotechnical properties"""
    name: str
    depth_from: float
    depth_to: float
    cohesion: float  # kPa
    friction_angle: float  # degrees
    unit_weight_sat: float  # kN/m3
    unit_weight_unsat: float  # kN/m3

@dataclass
class SliceData:
    """Data for a single slice in Bishop method"""
    x: float
    width: float
    height: float
    weight: float
    cohesion: float
    friction_angle: float
    water_height: float
    base_angle: float

class BishopCalculator:
    """
    Bishop Simplified Method for Slope Stability Analysis
    Based on: Bishop, A.W. (1955)
    """
    
    def __init__(self, layers: List[SoilLayer], water_table_depth: float):
        self.layers = layers
        self.water_table_depth = water_table_depth
        self.gamma_water = 9.81  # kN/m3
        
    def get_layer_at_depth(self, depth: float) -> SoilLayer:
        """Get the soil layer at a given depth"""
        for layer in self.layers:
            if layer.depth_from <= depth < layer.depth_to:
                return layer
        # Return last layer if beyond
        return self.layers[-1]
    
    def calculate_slice_weight(self, x: float, width: float, height: float, 
                              water_height: float) -> float:
        """Calculate weight of a slice considering multiple layers and water"""
        total_weight = 0.0
        slice_left = x - width / 2
        slice_right = x + width / 2
        
        # Divide slice height into segments based on layers
        current_depth = 0
        remaining_height = height
        
        for layer in self.layers:
            if remaining_height <= 0:
                break
                
            # Check if this layer intersects with the slice
            layer_thickness = min(remaining_height, 
                                 layer.depth_to - max(current_depth, layer.depth_from))
            
            if layer_thickness > 0:
                # Determine if saturated or unsaturated
                mid_depth = current_depth + layer_thickness / 2
                
                if mid_depth < self.water_table_depth:
                    # Unsaturated
                    segment_weight = layer.unit_weight_unsat * layer_thickness * width
                else:
                    # Saturated
                    segment_weight = layer.unit_weight_sat * layer_thickness * width
                
                total_weight += segment_weight
                remaining_height -= layer_thickness
                current_depth += layer_thickness
        
        return total_weight
    
    def create_slices(self, circle_center: Tuple[float, float], radius: float,
                     slope_angle: float, slope_height: float, 
                     num_slices: int = 30) -> List[SliceData]:
        """Create slices for Bishop method analysis"""
        xc, yc = circle_center
        slices = []
        
        # Find intersection points of circle with slope
        # Slope toe at origin (0, 0), crest at (slope_height/tan(slope_angle), slope_height)
        
        # Find leftmost and rightmost points where circle intersects ground
        # This is simplified - in real analysis you'd solve for exact intersections
        x_left = xc - radius
        x_right = xc + radius
        
        # Limit to slope boundaries
        slope_length = slope_height / np.tan(np.radians(slope_angle))
        x_left = max(x_left, -slope_length * 0.2)  # Some extension beyond toe
        x_right = min(x_right, slope_length * 1.2)  # Some extension beyond crest
        
        slice_width = (x_right - x_left) / num_slices
        
        for i in range(num_slices):
            x = x_left + (i + 0.5) * slice_width
            
            # Calculate slice height (distance from base to circle)
            # Ground surface elevation
            if x < 0:
                y_ground = 0
            elif x < slope_length:
                y_ground = x * np.tan(np.radians(slope_angle))
            else:
                y_ground = slope_height
            
            # Circle base at this x
            y_circle = yc - np.sqrt(max(0, radius**2 - (x - xc)**2))
            
            # Slice height
            height = max(0, y_ground - y_circle)
            
            if height > 0:
                # Get layer properties at slice base
                depth = y_circle  # Simplified depth calculation
                layer = self.get_layer_at_depth(abs(depth))
                
                # Water height in slice
                water_height = max(0, y_ground - self.water_table_depth)
                
                # Base angle (angle of slice base with horizontal)
                if (x - xc) != 0:
                    base_angle = np.degrees(np.arctan(-(x - xc) / 
                                  np.sqrt(max(0.01, radius**2 - (x - xc)**2))))
                else:
                    base_angle = 0
                
                # Calculate weight
                weight = self.calculate_slice_weight(x, slice_width, height, water_height)
                
                slice_data = SliceData(
                    x=x,
                    width=slice_width,
                    height=height,
                    weight=weight,
                    cohesion=layer.cohesion,
                    friction_angle=layer.friction_angle,
                    water_height=water_height,
                    base_angle=base_angle
                )
                slices.append(slice_data)
        
        return slices
    
    def calculate_sf_bishop(self, slices: List[SliceData], 
                           max_iterations: int = 50, 
                           tolerance: float = 0.001) -> float:
        """
        Calculate Factor of Safety using Bishop Simplified Method
        Uses iterative approach to solve for SF
        """
        if not slices:
            return 0.0
        
        # Initial guess for SF
        sf = 1.5
        
        for iteration in range(max_iterations):
            sf_new = 0.0
            driving_moment = 0.0
            resisting_moment = 0.0
            
            for s in slices:
                # Convert angles to radians
                phi_rad = np.radians(s.friction_angle)
                alpha_rad = np.radians(s.base_angle)
                
                # Pore water pressure at base
                u = self.gamma_water * s.water_height if s.water_height > 0 else 0
                base_length = s.width / np.cos(alpha_rad)
                
                # Normal force (calculated using current SF estimate)
                # N = [W - (c * b + u * b * tan(phi)) * tan(alpha)] / m_alpha
                # where m_alpha = cos(alpha) + sin(alpha)*tan(phi)/SF
                
                m_alpha = np.cos(alpha_rad) + (np.sin(alpha_rad) * 
                          np.tan(phi_rad) / max(sf, 0.1))
                
                if abs(m_alpha) < 0.01:
                    continue
                
                # Resisting forces along base
                c_term = s.cohesion * base_length
                u_term = u * base_length
                
                numerator = c_term + (s.weight - u_term) * np.tan(phi_rad)
                resisting_force = numerator / m_alpha
                
                # Driving force
                driving_force = s.weight * np.sin(alpha_rad)
                
                resisting_moment += resisting_force
                driving_moment += driving_force
            
            # Calculate new SF
            if driving_moment > 0:
                sf_new = resisting_moment / driving_moment
            else:
                sf_new = sf
            
            # Check convergence
            if abs(sf_new - sf) < tolerance:
                return sf_new
            
            sf = sf_new
        
        return sf
    
    def find_critical_circle(self, slope_angle: float, slope_height: float,
                           x_range: Tuple[float, float],
                           y_range: Tuple[float, float],
                           r_range: Tuple[float, float],
                           grid_points: int = 10) -> Dict:
        """
        Search for critical slip circle (minimum SF) using grid search
        Returns dictionary with circle center, radius, and minimum SF
        """
        min_sf = float('inf')
        critical_circle = None
        
        slope_length = slope_height / np.tan(np.radians(slope_angle))
        
        # Grid search
        x_centers = np.linspace(x_range[0] * slope_length, 
                               x_range[1] * slope_length, grid_points)
        y_centers = np.linspace(y_range[0] * slope_height, 
                               y_range[1] * slope_height, grid_points)
        radii = np.linspace(r_range[0] * slope_height, 
                           r_range[1] * slope_height, grid_points)
        
        for xc in x_centers:
            for yc in y_centers:
                for r in radii:
                    # Create slices for this circle
                    slices = self.create_slices((xc, yc), r, 
                                               slope_angle, slope_height)
                    
                    if len(slices) > 3:  # Need minimum slices for valid analysis
                        sf = self.calculate_sf_bishop(slices)
                        
                        if sf < min_sf and sf > 0:
                            min_sf = sf
                            critical_circle = {
                                'center_x': xc,
                                'center_y': yc,
                                'radius': r,
                                'sf': sf,
                                'slices': slices
                            }
        
        return critical_circle
    
    def optimize_slope_design(self, target_sf: float, 
                             initial_angle: float,
                             max_height: float = 100.0,
                             height_step: float = 5.0,
                             optimize_both: bool = True,
                             angle_range: tuple = (30, 60),
                             angle_step: float = 5.0) -> Dict:
        """
        Find optimal slope design (height and angle) that maximizes pit depth
        while maintaining target SF
        
        If optimize_both=True: Tests multiple angles and finds best combination
        If optimize_both=False: Uses initial_angle and finds max height only
        """
        all_designs = []
        
        if optimize_both:
            # Test multiple angles
            angles_to_test = np.arange(angle_range[0], angle_range[1] + angle_step, angle_step)
            print(f"Testing angles from {angle_range[0]}° to {angle_range[1]}°...")
        else:
            # Only test the initial angle
            angles_to_test = [initial_angle]
        
        for angle in angles_to_test:
            print(f"\nTesting slope angle: {angle}°")
            max_safe_height = 0
            best_for_angle = None
            
            # Find maximum height for this angle
            for height in np.arange(10, max_height + height_step, height_step):
                # Search for critical circle at this height and angle
                critical = self.find_critical_circle(
                    slope_angle=angle,
                    slope_height=height,
                    x_range=(0.3, 0.7),
                    y_range=(0.5, 1.5),
                    r_range=(0.8, 2.0),
                    grid_points=8
                )
                
                if critical and critical['sf'] >= target_sf:
                    max_safe_height = height
                    best_for_angle = {
                        'height': height,
                        'angle': angle,
                        'sf': critical['sf'],
                        'critical_circle': critical,
                        'volume_factor': height / np.tan(np.radians(angle))  # Proxy for pit volume
                    }
                    print(f"  Height {height}m: SF = {critical['sf']:.3f} ✓")
                else:
                    # SF dropped below target
                    print(f"  Height {height}m: SF too low, stopping at {max_safe_height}m")
                    break
            
            if best_for_angle:
                all_designs.append(best_for_angle)
                print(f"✓ Angle {angle}°: Max safe height = {max_safe_height}m, SF = {best_for_angle['sf']:.3f}")
        
        if not all_designs:
            return None
        
        # Select best design based on maximum volume (depth * width)
        # This represents the most economic design
        best_design = max(all_designs, key=lambda d: d['volume_factor'])
        
        print(f"\n{'='*60}")
        print(f"OPTIMAL DESIGN FOUND:")
        print(f"  Recommended Angle: {best_design['angle']}°")
        print(f"  Maximum Safe Height: {best_design['height']}m")
        print(f"  Safety Factor: {best_design['sf']:.3f}")
        print(f"{'='*60}")
        
        return best_design


# Example usage and testing
if __name__ == "__main__":
    # Example: Create sample layers
    layers = [
        SoilLayer("Soil", 0, 5, 90, 26.71, 22.86, 22.07),
        SoilLayer("Mudstone", 5, 15, 120.72, 19.95, 22.48, 21.82),
        SoilLayer("Sandstone", 15, 30, 116.29, 30.83, 22.91, 22.37),
    ]
    
    # Create calculator
    calc = BishopCalculator(layers, water_table_depth=8.0)
    
    # Test: Find optimal design for 45-degree slope
    print("Optimizing slope design...")
    result = calc.optimize_slope_design(
        target_sf=1.3,
        initial_angle=45,
        max_height=50
    )
    
    if result:
        print(f"\nOptimal Design:")
        print(f"Maximum Height: {result['height']:.2f} m")
        print(f"Slope Angle: {result['angle']:.2f} degrees")
        print(f"Safety Factor: {result['sf']:.3f}")
    else:
        print("No safe design found with given parameters")