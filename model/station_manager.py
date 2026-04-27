from typing import List, Optional, Dict

from model.station import Station
from .pod_manager import PodManager
import pandas as pd
import numpy as np


class StationManager:
    def __init__(self):
        self.stations: List[Station] = []
        self.picking_stations: List[Station] = []
        self.replenishment_stations: List[Station] = []
        self.stations_by_id: Dict[int, Station] = {}

    def find_available_picking_station(self) -> Optional[Station]:
        # Initialize the available station variable as None
        available_station = None
        # Initialize the minimum number of orders to a high value to find the station with the least orders
        min_orders = float('inf')

        # Iterate through each station to check the number of orders
        for station in self.picking_stations:
            if len(station.order_ids) < station.max_orders:
                # Check if this station has fewer orders than the current minimum
                if len(station.order_ids) < min_orders:
                    min_orders = len(station.order_ids)
                    available_station = station

        return available_station
    
    def find_available_replenish_station(self) -> Optional[Station]:
        # Initialize the available station variable as None
        available_station = None
        # Initialize the minimum number of orders to a high value to find the station with the least orders
        min_robots = float('inf')

        # Iterate through each station to check the number of orders
        for station in self.replenishment_stations:
            if len(station.robot_ids) < station.max_robots:
                # Check if this station has fewer orders than the current minimum
                if len(station.robot_ids) < min_robots:
                    min_robots = len(station.robot_ids)
                    available_station = station

        return available_station

    def find_highest_similarity_station(self, skus_in_order: dict, pod_manager: PodManager) -> Optional[Station]:
        available_station_rank = pd.DataFrame(columns=["station_id", "similarity_score", "current_orders"])
        sku_in_order_list = [i for i in skus_in_order]
        available_station = []
        assign_station = None

        # Store all available station
        for station in self.picking_stations:
            if len(station.order_ids) < station.max_orders:
                available_station.append(station)
        
        # Check if more than one station is available
        if len(available_station) > 0:
            for station in available_station:
                # Check Available Station
                current_orders = len(station.order_ids)
                # print (f"current orders:{current_orders}")
                similarity_score = 0
                if len(station.order_ids) < station.max_orders:
                    # Take pod assigned to this particular station
                    station_incoming_pod = station.incoming_pod
                    station_pod_skus_set = set()  # Jenis SKU yang ada di pod yang ngantri + otw
                    for pod_id in station_incoming_pod:
                        pod  = pod_manager.get_pod_by_id(pod_id)
                        pod_skus = [item for item, details in pod.skus.items() if details['current_qty'] > 0]
                        station_pod_skus_set.update(pod_skus)
                        # print(f"station_pod_skus_set: {station_pod_skus_set}")

                    station_pod_skus_list = list(station_pod_skus_set)
                    # print(f"station_pod_skus_list: {station_pod_skus_list}")
                    # yang kembar (mask)
                    station_pod_skus_in_order_mask = np.isin(sku_in_order_list, station_pod_skus_list)
                    # yang kembar apa aja
                    station_pod_skus_in_order = np.array(sku_in_order_list)[station_pod_skus_in_order_mask]
                    # yang kembar ada berapa
                    similarity_score = len(station_pod_skus_in_order)

                    available_station_rank = pd.concat([available_station_rank , 
                                                pd.DataFrame([[station.station_id, similarity_score, current_orders]], columns=["station_id", "similarity_score", "current_orders"])], ignore_index=True) 
            
            available_station_rank.sort_values(by=["similarity_score"], ascending=False, inplace=True)
            available_station_rank.reset_index(drop=True, inplace=True)
        
        min_orders = available_station_rank['current_orders'].min()
        eligible_stations =  available_station_rank[available_station_rank['current_orders'] - min_orders <= 3]
        if len(eligible_stations) > 0:
            eligible_stations.sort_values(by=["similarity_score"], ascending=False, inplace=True)
            eligible_stations.reset_index(drop=True, inplace=True)
            assign_station_id = eligible_stations["station_id"].head(1).values[0]
            assign_station = self.get_station_by_id(assign_station_id)
        elif len(available_station_rank) > 0:
            assign_station_id = available_station_rank["station_id"].head(1)
            assign_station = self.get_station_by_id(assign_station_id)

        return assign_station
    
    def assign_robot_to_picking_station(self, robot_id):
        # NOTE: no use
        station = self.find_available_picking_station()
        if station is None:
            print("No available station found.")
            return
    
        # Check active robot capacity first
        if len(station.robot_ids) < station.max_robots:
            station.add_robot(robot_id)
            print(f"Robot {robot_id} assigned to station {station.station_id} (active).")
        
        # If active capacity is full, try the queue
        elif len(station.robot_queue) < station.max_robot_queue:
            station.robot_queue.append(robot_id)
            print(f"Robot {robot_id} added to queue at station {station.station_id}.")
        else:
            print(f"Station {station.station_id} is full (active + queue).")


    def add_station(self, station: Station):
        self.stations.append(station)
        self.stations_by_id[station.station_id] = station

        if station.is_picker_station():
            self.picking_stations.append(station)
        elif station.is_replenishment_station():
            self.replenishment_stations.append(station)

    def get_station_by_id(self, station_id):
        return self.stations_by_id[station_id]
    
    def find_highest_supplyrate_station_rika(self, skus_in_order, pod_manager: PodManager) -> Optional[Station]:
        available_station_rank = pd.DataFrame(columns=["station_id", "similarity_score", "current_orders"])
        sku_in_order_list = [i for i in skus_in_order]
        available_station = []
        assign_station = None

        # Store all available station
        for station in self.picking_stations:
            if len(station.order_ids) < station.max_orders:
                available_station.append(station)
        
        # Check if more than one station is available
        if len(available_station) > 0:
            for station in available_station:
                # Check Available Station
                current_orders = len(station.order_ids)
                print (f"current orders:{current_orders}")
                ######################################
                similarity_score = 0
                if len(station.order_ids) < station.max_orders:
                    # Take pod assigned to this particular station
                    station_incoming_pod = station.incoming_pod
                    station_pod_skus_set = set() #kenapa set ? bisa
                    for pod_id in station_incoming_pod:
                        pod  = pod_manager.get_pod_by_id(pod_id)
                        pod_skus = [item for item, details in pod.skus.items() if details['current_qty'] > 0]
                        station_pod_skus_set.update(pod_skus)
                        # print(f"station_pod_skus_set: {station_pod_skus_set}")

                    station_pod_skus_list = list(station_pod_skus_set)
                    print(f"station_pod_skus_list: {station_pod_skus_list}")
                    station_pod_skus_in_order_mask = np.isin(sku_in_order_list, station_pod_skus_list)
                    #isin to compare the order's SKUs with those available in the station:
                    station_pod_skus_in_order = np.array(sku_in_order_list)[station_pod_skus_in_order_mask]
                    similarity_score = len(station_pod_skus_in_order)

                    available_station_rank = pd.concat([available_station_rank , 
                                                pd.DataFrame([[station.station_id, similarity_score, current_orders]], columns=["station_id", "similarity_score", "current_orders"])], ignore_index=True) 
            
            available_station_rank.sort_values(by=["similarity_score"], ascending=False, inplace=True)
            available_station_rank.reset_index(drop=True, inplace=True)
        
        min_orders = available_station_rank['current_orders'].min() # to identify which stations are the least busy.
        eligible_stations =  available_station_rank[available_station_rank['current_orders'] - min_orders <= 3]
        #includes only those stations whose current order count is within 3 orders of the minimum
        if len(eligible_stations) > 0:
            eligible_stations.sort_values(by=["similarity_score"], ascending=False, inplace=True)
            eligible_stations.reset_index(drop=True, inplace=True)
            assign_station_id = eligible_stations["station_id"].head(1).values[0]
            assign_station = self.get_station_by_id(assign_station_id)
        elif len(available_station_rank) > 0:
            assign_station_id = available_station_rank["station_id"].head(1)
            assign_station = self.get_station_by_id(assign_station_id)

        return assign_station