from __future__ import annotations
from typing import List, Dict, Optional, TYPE_CHECKING
from model.storage import Storage
from model.pod import Pod
from engine.netlogo_coordinate import NetLogoCoordinate
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import manhattan_distances

if TYPE_CHECKING:
    from model.inventory import Inventory

class StorageManager:
    def __init__(self, warehouse: "Inventory"):
        self.warehouse = warehouse
        self.storages: List[Storage] = []
        self.storage_counter = 0
        self.pods_to_storage: Dict[Pod, Storage] = {}
        self.coordinate_to_storages: Dict[tuple[int, int], Storage] = {}
        self.empty_storages: List[Storage] = []

    def initStorageManager(self):
        for storage in self.storages:
            storage.setStorageManager(self)

    def getAllStorages(self) -> List[Storage]:
        return self.storages

    def getStorageByPod(self, pod: Pod) -> Optional[Storage]:
        return self.pods_to_storage.get(pod, None)

    def getStorageByCoordinate(self, x: int, y: int) -> Optional[Storage]:
        return self.coordinate_to_storages.get((x, y), None)

    def getEmptyStorage(self) -> Optional[Storage]:
        return self.empty_storages[0] if self.empty_storages else None

    def setStorageNotAvailable(self, coordinate: NetLogoCoordinate):
        storage = self.getStorageByCoordinate(coordinate.x, coordinate.y)
        if storage and storage in self.empty_storages:
            self.empty_storages.remove(storage)
        if storage:
            storage.is_empty = False

    def setStorageAvailable(self, coordinate: NetLogoCoordinate):
        storage = self.getStorageByCoordinate(coordinate.x, coordinate.y)
        if storage and storage not in self.empty_storages:
            self.empty_storages.append(storage)
        if storage:
            storage.is_empty = True

    def createStorage(self, x: int, y: int) -> Storage:
        storage = Storage(self.storage_counter, x, y)
        storage.is_empty = True
        self.storage_counter += 1
        self.storages.append(storage)
        self.coordinate_to_storages[(x, y)] = storage
        self.empty_storages.append(storage)
        return storage

    def addPodToStorage(self, pod: Pod, storage: Storage):
        self.pods_to_storage[pod] = storage
        storage.assigned_pod = pod
        if storage in self.empty_storages:
            self.empty_storages.remove(storage)
        storage.is_empty = False

    def getPodByNumber(self, pod_number: int) -> Optional[Pod]:
        return next((p for p in self.pods_to_storage if p.pod_number == pod_number), None)

    def getNearestEmptyStorage(self, station_coordinate: NetLogoCoordinate, robots_coordinate: List[List[int]]) -> Optional[Storage]:
        station_pos = np.array([[station_coordinate.x, station_coordinate.y]])
        best_storage = None
        best_score = -1

        candidate_data = []

        for storage in self.empty_storages:
            storage_pos = np.array([[storage.pos_x, storage.pos_y]])
            distance_to_station = manhattan_distances(storage_pos, station_pos)[0][0]
            distance_to_robot = self._distanceStorageToRobot(storage_pos[0], robots_coordinate)

            score = -distance_to_station - distance_to_robot  # Lower is better
            candidate_data.append((storage, score))

        if candidate_data:
            best_storage = sorted(candidate_data, key=lambda x: x[1])[0][0]

        return best_storage

    def _distanceStorageToRobot(self, storage_coord, robots_coordinate: List[List[int]]) -> float:
        if not robots_coordinate:
            return 1000.0  # Arbitrary high distance if no robot is present
        distances = manhattan_distances([storage_coord], robots_coordinate)
        return distances.min()
    

    def getNearestEmptyStorageToLocation(self, location_coordinate: NetLogoCoordinate, robot_coordinate: NetLogoCoordinate):
        location_pos = np.array([[location_coordinate.x, location_coordinate.y]])
        robot_coords_np = np.array([[robot_coordinate.x, robot_coordinate.y]])

        available_storages = [
            s for s in self.storages
            if s.is_empty and s.assigned_pod is None
        ]

        if not available_storages:
            return None

        storage_positions = np.array([[s.pos_x, s.pos_y] for s in available_storages])
        # print("[DEBUG] getNearestEmptyStorageToLocation")
        # print(f"[DEBUG] storage_position {storage_positions}")
        distances = manhattan_distances(storage_positions, location_pos).flatten()

        min_idx = distances.argmin()
        return available_storages[min_idx]
