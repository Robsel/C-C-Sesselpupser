import random
import math

def generate_points_file(filename, num_points, dimensions, num_clusters, min_val, max_val, cluster_std):
    # Generate random cluster centers
    cluster_centers = []
    for _ in range(num_clusters):
        center = [random.uniform(min_val, max_val) for _ in range(dimensions)]
        cluster_centers.append(center)
    
    # Split points among clusters
    points_per_cluster = num_points // num_clusters
    remaining_points = num_points % num_clusters
    
    with open(filename, "w") as f:
        cluster_idx = 0
        for cluster_id, center in enumerate(cluster_centers):
            # Add one extra point to first clusters if there's a remainder
            points_in_cluster = points_per_cluster + (1 if cluster_id < remaining_points else 0)
            
            for _ in range(points_in_cluster):
                # Generate point around cluster center with Gaussian distribution
                point = []
                for d in range(dimensions):
                    val = random.gauss(center[d], cluster_std)
                    # Clamp to bounds
                    val = max(min_val, min(max_val, val))
                    point.append(f"{val:.5f}")
                f.write(" ".join(point) + "\n")

def main():
    num_points = int(input("Enter total number of points: "))
    num_clusters = int(input("Enter number of cluster centers: "))
    dimensions = int(input("Enter number of dimensions: "))
    min_val = float(input("Enter minimum value: "))
    max_val = float(input("Enter maximum value: "))
    cluster_std = float(input("Enter cluster standard deviation (spread around center): "))
    filename = input("Enter output filename (e.g. data.txt): ")

    if min_val > max_val:
        print("Error: minimum value cannot be greater than maximum value.")
        return
    
    if num_clusters <= 0:
        print("Error: number of clusters must be greater than 0.")
        return
    
    if num_points < num_clusters:
        print("Error: number of points must be at least the number of clusters.")
        return

    generate_points_file(filename, num_points, dimensions, num_clusters, min_val, max_val, cluster_std)
    print(f"File '{filename}' created with {num_points} points in {dimensions} dimensions around {num_clusters} cluster centers.")

if __name__ == "__main__":
    main()