from typing import List, Optional, Dict

from engine.object import Object
from engine.netlogo_coordinate import NetLogoCoordinate

from .order import Order


class Station(Object):
    def __init__(self, station_id: int, station_type: str):
        self.station_id = f"{station_type}-{station_id}"
        self.station_type = station_type
        self.shape = 'empty-space'
        self.object_type = 'station'
        self.mass = 1
        self.coordinate = None
        self.short_path: List[NetLogoCoordinate] = []
        self.long_path: List[NetLogoCoordinate] = []
        self.order_ids: List[int] = []
        self.orders: List[Order] = []
        self.robot_job = 0
        self.max_orders = 8 # Picking station capacity
        self.max_robots = 11
        self.short_path_threshold = 4
        self.robot_ids = {}
        self.robot_queue = []  # New: Queue for robots waiting for their turn  # NOTE: no use
        self.max_robot_queue = 6  # New: Maximum number of robots allowed in the queue  # NOTE: no use
        self.is_using_short_route = True
        self.skus = {} # {A:15, B: 10}
        self.skus_in_station = {} # {A:[5,10], B:[10]}
        self.incoming_pod: List[int] = []
        super().__init__()

    def add_order(self, order_id: int, order:Order):
        self.order_ids.append(order_id)
        self.orders.append(order)

        for sku, value in order.get_remaining_skus().items():
            if sku not in self.skus_in_station:
                self.skus_in_station[sku] = []
            self.skus_in_station[sku].append(value)

    def reduce_sku_from_station(self, sku, value):
        if sku in self.skus_in_station and value in self.skus_in_station[sku]:
            self.skus_in_station[sku].remove(value)
            if len(self.skus_in_station[sku]) == 0:
                self.skus_in_station.pop(sku)

    def remove_order(self, order_id: int, order: Order):
        if order_id in self.order_ids:
            self.order_ids.remove(order_id)
        if order in self.orders:
            self.orders.remove(order)

    def add_pod(self, pod_id):
        self.incoming_pod.append(pod_id)
    
    def remove_pod(self, pod_id):
        if pod_id in self.incoming_pod:
            self.incoming_pod.remove(pod_id)
        
    def add_robot_job(self):
        self.robot_job += 1
    
    def remove_robot_job(self):
        self.robot_job -= 1

    def is_picker_station(self) -> bool:
        return self.station_type == "picker"

    def is_replenishment_station(self) -> bool:
        return self.station_type == "replenishment"

    def get_path(self):
        if self.is_using_short_route:
            return self.short_path
        else:
            return self.long_path

    def get_sub_path(self, robot_id,x: int, y: int):
        sub_path = []
        start = False
        for coord in self.get_robot_route(robot_id):
            if (coord.x, coord.y) == (x, y):
                start = True
            if start:
                sub_path.append(NetLogoCoordinate(coord.x, coord.y))
        return sub_path

    def reevaluate_route(self):
        if self.is_using_short_route and len(self.robot_ids) > self.short_path_threshold:
            self.is_using_short_route = False
        elif not self.is_using_short_route and len(self.robot_ids) == 0:
            self.is_using_short_route = True

    def add_robot(self, robot_id):
        self.robot_ids[robot_id] = self.get_path()
        self.reevaluate_route()

    def remove_robot(self, robot_id):
        if robot_id in self.robot_ids:
            del self.robot_ids[robot_id]
        self.reevaluate_route()

    def update_robot_route_type(self, robot_id):
        if robot_id in self.robot_ids:
            self.robot_ids[robot_id] = self.get_path()

    def get_robot_route(self, robot_id):
        # print(f"[DEBUG] current_station {self.station_id} robot_id {robot_id} robot_ids {self.robot_ids}")
        return self.robot_ids.get(robot_id, None)

    def has_route_changed(self, robot_id):
        return self.get_path() != self.get_robot_route(robot_id)
    
    def get_skus_in_station(self):
        for sku, value in self.skus_in_station.items():
            self.skus[sku] = sum(value)
        return self.skus
    
    def get_orders_in_station(self) -> Optional[List[Order]]: 
        return self.orders
    
    def get_skus_in_station_dict(self) -> Optional[Dict]:
        return self._sort_order_set(self.skus_in_station)

    def _sort_order_set(self,order_set):
        sorted_order_set = {}
        for index, value in order_set.items():
            sorted_order_set[index] = sorted(value, reverse=False)
        return sorted_order_set
