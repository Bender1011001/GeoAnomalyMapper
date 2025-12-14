import numpy as np
import matplotlib.pyplot as plt
import os
try:
    from sklearn.metrics import auc
except ImportError:
    # Fallback to numpy trapezoidal rule
    def auc(x, y):
        return np.trapz(y, x)

class PredictionAreaPlotter:
    def __init__(self, prospectivity_map, deposit_coords, map_resolution=100.0):
        """
        Args:
            prospectivity_map (ndarray): 2D array of inversion scores (density/probability).
            deposit_coords (list of tuples): [(row, col),...] indices of known deposits.
            map_resolution (float): Pixel size in meters (for area calculation).
        """
        self.pmap = prospectivity_map
        self.deposits = deposit_coords
        self.total_area = prospectivity_map.size
        self.num_deposits = len(deposit_coords)
        self.map_resolution = map_resolution
        
    def calculate_curve(self, num_steps=1000):
        # Flatten map and sort
        flat_vals = self.pmap.flatten()
        # Sort descending
        sorted_vals = np.sort(flat_vals)[::-1] 
        
        # Get values at deposit locations
        # Handle cases where deposits are outside map bounds
        valid_deposits = []
        for r, c in self.deposits:
            if 0 <= r < self.pmap.shape[0] and 0 <= c < self.pmap.shape[1]:
                valid_deposits.append(self.pmap[r, c])
        
        dep_vals = np.array(valid_deposits)
        actual_num_deposits = len(dep_vals)
        
        if actual_num_deposits == 0:
            print("Warning: No deposits fall within the map bounds.")
            return [0, 100], [0, 0]

        # Define thresholds. 
        # Using unique values from the map itself or deposits can be slow for large maps.
        # Let's use percentiles of the map values to ensure even spacing in "Area" axis
        thresholds = np.percentile(sorted_vals, np.linspace(0, 100, num_steps))[::-1]
        
        plot_area = []
        plot_pred = []
        
        for thresh in thresholds:
            # Area Occupied %
            # Count pixels >= threshold
            pixels_above = np.sum(flat_vals >= thresh)
            oa = (pixels_above / self.total_area) * 100
            
            # Prediction Rate %
            # Count deposits >= threshold
            deps_captured = np.sum(dep_vals >= thresh)
            pr = (deps_captured / actual_num_deposits) * 100
            
            plot_area.append(oa)
            plot_pred.append(pr)
            
        # Add (0,0) and (100,100) explicitly to close the curve
        # But our loop likely covers them near 0 and 100 percentile
        # Let's clean up duplicate points and ensure 0,0 and 100,100 are there
        plot_area = [0.0] + plot_area + [100.0]
        plot_pred = [0.0] + plot_pred + [100.0]
        
        return plot_area, plot_pred

    def plot(self, output_path=None):
        oa, pr = self.calculate_curve()
        
        # Calculate Area Under Curve (AUC)
        # Normalize to [0,1] for AUC calc
        try:
            auc_score = auc(np.array(oa)/100.0, np.array(pr)/100.0)
        except:
            auc_score = 0.0
        
        plt.figure(figsize=(8, 8))
        plt.plot(oa, pr, 'b-', linewidth=2, label=f'Model (AUC={auc_score:.3f})')
        plt.plot([0, 100], [0, 100], 'k--', label='Random Prediction')
        
        # Highlight Top 5% Exploration efficiency
        # Find index closest to 5% area
        idx_5 = (np.abs(np.array(oa) - 5.0)).argmin()
        
        plt.scatter(oa[idx_5], pr[idx_5], color='red', zorder=5)
        plt.annotate(f'Captured {pr[idx_5]:.1f}% deposits\nin {oa[idx_5]:.1f}% area', 
                     (oa[idx_5]+2, pr[idx_5]-5))
        
        plt.xlabel('Cumulative Area Occupied (%)')
        plt.ylabel('Prediction Rate of Deposits (%)')
        plt.title('Prediction-Area (P-A) Plot')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        
        if output_path:
            plt.savefig(output_path)
            print(f"P-A Plot saved to {output_path}")
        # plt.show() # Disabled for headless environment
        plt.close()

if __name__ == "__main__":
    # Test Block
    print("Running test for PredictionAreaPlotter...")
    # Create synthetic map
    test_map = np.random.rand(100, 100)
    # Create synthetic deposits (concentrated in high value areas for test)
    # Set a region to high value
    test_map[40:60, 40:60] += 2.0
    
    # Deposits in the "high" region
    test_deposits = [(50, 50), (45, 45), (55, 55), (10,10)] # 3 in high, 1 random
    
    plotter = PredictionAreaPlotter(test_map, test_deposits)
    plotter.plot("test_pa_plot.png")
