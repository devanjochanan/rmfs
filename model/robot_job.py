from typing import List
import os
import csv

from engine.netlogo_coordinate import NetLogoCoordinate
from .tools.pod_location import upsert_pod_location


class RobotJob:
    counter = 1
    def __init__(self, pod_coordinate: NetLogoCoordinate, station_id, pod):
        self.my_id = RobotJob.counter
        RobotJob.counter +=1
        self.job_id = id
        self.pod_coordinate = pod_coordinate
        self.pod_return_coordinate = NetLogoCoordinate(0,0)
        self.pod = pod
        self.station_id = station_id
        self.orders: list[tuple[int, int, int]] = []  # This will hold tuples of (order_id, sku, quantity)
        self.picking_delay_per_sku = 8 # Time for handling a task
        self.picking_delay = 0
        self.replenishment_delay_per_sku = 20
        self.replenishment_delay = 0
        self.is_finished = False

    def add_picking_task(self, order_id, sku, quantity):
        """Add an order with the specific SKU and quantity to be picked."""
        self.orders.append((order_id, sku, quantity))
        self.picking_delay += self.picking_delay_per_sku * quantity
    
    def add_replenishment_task(self, pod):
        total_skus = len(pod.skus)
        self.replenishment_delay += total_skus * self.replenishment_delay_per_sku

    def is_being_processed(self):
        """Check if the job is being processed based on delays."""
        return self.picking_delay > 0 or self.replenishment_delay > 0

    def decrement_delay(self):
        """Decrement the picking or replenishment delay."""
        if self.picking_delay > 0:
            self.picking_delay -= 1
        elif self.replenishment_delay > 0:
            self.replenishment_delay -= 1

    def set_job_finish(self):
        self.is_finished = True
        upsert_pod_location(self.pod.pod_id, self.pod.pos_x, self.pod.pos_y)

    def pop_order(self):
        return self.orders.pop(0)
    
    def __str__(self):
        return f"RobotJob: {str(self.job_id)}, pid {self.pod.pod_id}, xy {self.pod_coordinate}, stid {self.station_id}, ords {self.orders}"

    def __repr__(self):
        return self.__str__()
    
    def writePodReturnReport(self, manhattan_distance):
        return
        log_file = "log_pod_return.csv"
        file_exists = os.path.isfile(log_file)
        
        with open(log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            
            if not file_exists:
                writer.writerow(["Job ID", "Pod", "Pod Coordinate", "Return Coordinate", "Station ID", "Manhattan Distance"])
            
            # Write the data
            writer.writerow([
                self.job_id,
                self.pod,
                self.pod_coordinate,
                self.pod_return_coordinate,
                self.station_id,
                manhattan_distance
            ])

    def record_delivery(self, tick):
        return
        log_file = "log_pod_delivery.csv"
        file_exists = os.path.isfile(log_file)
        with open(log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            
            if not file_exists:
                writer.writerow(["Tick", "Pod", "Pod Coordinate", "Pod Coordinate inside pod object", "To Station ID"])
            
            # Write the data
            writer.writerow([
                f"{tick:.2f}",
                self.pod,
                self.pod_coordinate,
                (self.pod.pos_x, self.pod.pos_y),
                self.station_id
            ])
