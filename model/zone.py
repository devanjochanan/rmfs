import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_score

class Zone:
    boundaries = [] # 3 Dimension Array berarti appendnya 2D
    penalty = [] # 1D Array of integer
    cluster_num = 2

    def __init__(self, robots_location, warehouse_size,methods):
        if methods == "default":
            self.boundaries = [
        [
            [5,20],
            [19,30],
        ],
        [
            [20,20],
            [29,30],
        ],
        [
            [30,20],
            [38,30],
        ],
        [
            [5,10],
            [19,19],
        ],
        [
            [20,10],
            [29,19],
        ],
        [
            [30,10],
            [38,19],
        ],
        [
            [5,0],
            [19,9],
        ],
        [
            [20,0],
            [29,9],
        ],
        [
            [30,0],
            [38,9],
        ]
        ]
        elif methods == "kmeans":
            self.kmeans_clustering(robots_location)
        elif methods == "affinity_propagation":
            self.affinity_propagation(robots_location)
        elif methods == "route_cluster":
            self.route_cluster(warehouse_size)
        
    def get_boundary(self):
        return self.boundaries
    
    def is_over_threshold(self, boundary, robot_locations):
        bottom_left, top_right = boundary
        width = top_right[0] - bottom_left[0] + 1
        height = top_right[1] - bottom_left[1] + 1
        area = width * height
        threshold = 0.5 * area
        
        robot_count = sum(
            bottom_left[0] <= x <= top_right[0] and bottom_left[1] <= y <= top_right[1]
            for x, y in robot_locations
        )
        
        return robot_count > threshold
    
    def calculate_penalty(self, robots_location, idle_time, warehouse_size, threshold):
        self.penalty = [1] * len(self.boundaries)
        area = [1] * len(self.boundaries)
        robot_count = [1] * len(self.boundaries)
        warehouse_area = warehouse_size[0] * warehouse_size[1]
        
        # print("idle time: ", idle_time)
        # print("robot locations: ", robots_location)
        
        for index, zone in enumerate(self.boundaries):
            area[index] = abs(zone[1][0] - zone[0][0]) * abs(zone[1][1] - zone [0][1])
            # print("area: ", area[index])

        for robot in robots_location:
            for index, zone in enumerate(self.boundaries):
                if ((robot[1] <= zone[0][0] and robot[1] >= zone[1][0]) and (robot[0] >= zone[0][1] and robot[0] <= zone[1][1])):
                    robot_count[index] += 1
                    # self.penalty[index] += 1  

        for index, zone in enumerate(self.boundaries):
            self.penalty[index] = area[index] / robot_count[index]
            # self.penalty[index] = robot_count[index]

        robot_idle_zone = [1] * len(self.boundaries)
        # robot_idle_zone = [1] * len(robots_location)
     
        for index, zone in enumerate(self.boundaries):
            for robot_index, robot in enumerate(robots_location):
                # print("y: ", robot[0])
                if (robot[1] <= zone[0][0] and robot[1] >= zone[1][0]) and (robot[0] >= zone[0][1] and robot[0] <= zone[1][1]) and idle_time[robot_index] > 50:
                    # robot_idle_zone[robot_index] += 1 
                    robot_idle_zone[index] += 1 

        for robot in robots_location:
            for index, zone in enumerate(self.boundaries):
                if robot_idle_zone[index] >= threshold:
                    if robot[0] >= zone[0][1] and robot[0] <= zone[1][1]: # robot in boundary's line
                        if robot[1] == zone[0][0] or robot[1] == zone[1][0]:
                            self.penalty[index] += 100 * warehouse_area
                    elif robot[1] >= zone[0][0] and robot[0] <= zone[1][0]: # robot in boundary's line
                        if robot[1] == zone[0][1] or robot[1] == zone[1][1]:
                            self.penalty[index] += 100 * warehouse_area

        #get robot by coor 
        return self.penalty
    
        
   
    
    @staticmethod
    def _minimum_bounding_rectangle(points):
        x_coords, y_coords = zip(*points)
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        # if x_min < 5:
        #     x_min = 5
        if x_max - x_min + 1 < 3:
            diff = 3 - (x_max - x_min + 1)
            x_max += diff // 2
            x_min -= diff - (diff // 2)
        if y_max - y_min + 1 < 6:
            diff = 6 - (y_max - y_min + 1)
            y_max += diff // 2
            y_min -= diff - (diff // 2)

        if x_min < 5:
            x_min = 5
        if y_min < 0:
            y_min = 0
        
        if y_max > 30:
            y_max = 30
        if x_max > 43:
            x_max = 43
        return [ [int(y_min), int(x_max)],[int(y_max), int(x_min)],]
    
    def _silhouette_score(self, robots_location, min_cluster, max_cluster):
        inertias = []
        silhouette_scores = []
        for k in range(min_cluster, max_cluster + 1):
            if k >= len(robots_location):
                break
            kmeans = KMeans(n_clusters=k, random_state=0).fit(robots_location)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(robots_location, kmeans.labels_))
        
        best_cluster = range(min_cluster, max_cluster + 1)[silhouette_scores.index(max(silhouette_scores))]
        return best_cluster

    def kmeans_clustering(self, robots_location):
        """Clustering using KMeans

        Args:
            robots_location (list): list of robots location.
            min_col (int): Minimum column for a zone.
            min_row (int): Minimum row for a zone.

        Returns:
            lists of zone boundaries
        """
        robots = np.array(robots_location)
        if len(robots) >= 3:
            self.cluster_num = self._silhouette_score(robots, min_cluster=2, max_cluster=9)
            kmeans = KMeans(n_clusters=self.cluster_num, random_state=0)
            labels = kmeans.fit_predict(robots)
            boundaries = []
            for cluster_id in range(self.cluster_num):
                cluster_points = robots[labels == cluster_id]
                cluster_boundary_points = self._minimum_bounding_rectangle(cluster_points)
                boundaries.append(cluster_boundary_points)

            self.boundaries = boundaries
        return
    
    def affinity_propagation(self, robots_location):
        robots = np.array(robots_location)

        damping = 0.9  # Damping factor (between 0.5 and 1) to avoid numerical oscillations
        preference = -50  # Preferences for each point (lower values create more clusters)

        # Apply Affinity Propagation clustering to robot positions
        if len(robots) != 0:
            affinity_propagation = AffinityPropagation(damping=damping, preference=preference).fit(robots)
            labels = affinity_propagation.labels_
            boundaries = []
            unique_labels = np.unique(labels)
            for label in unique_labels:
                if label == -1:
                    continue  # Label -1 indicates noise points
                cluster_points = robots[labels == label]
                cluster_boundary_points = self._minimum_bounding_rectangle(cluster_points)
                boundaries.append(cluster_boundary_points)
            self.boundaries = boundaries
        return 
    
    def route_cluster(self,warehouse_size):
        total_row = warehouse_size[0]
        total_col = warehouse_size[1]
        zones = []

        # Make zone for left highway
        end_row = 3
        for row in range(0, total_row, 4):
            zones.append([[row, 5],[end_row, 9]])
            end_row += 4

        # Make zone for right highway
        
        # Make zone for horizontal paths
        end_col = 14
        for row in range(0, total_row, 3):
            end_col = 14
            for col in range(10, total_col-10,6):
                zones.append([[row, col], [row, end_col]])
                end_col += 6

        # Make zone for vertical paths
        end_row = 3
        for col in range(15, 34, 6):
            for row in range(0, total_row, 4):
                zones.append([[row,col],[row+3, col]])

        self.boundaries = zones
        # print(zones)
        return

