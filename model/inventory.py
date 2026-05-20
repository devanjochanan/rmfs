from typing import Optional, List
import csv
import os
import math
import threading
import re
from collections import defaultdict, deque
import ast
import json
import pandas as pd
from datetime import datetime

from engine.landscape import Landscape
from engine.universe import Universe
from engine.util import *
from .intersection_manager import IntersectionManager
from .order import Order
from .order_manager import OrderManager
from .pod import Pod
from .pod_manager import PodManager
from .robot import Robot
from .robot_job import RobotJob
from .station_manager import StationManager
from .station import Station
from .storage_manager import StorageManager
from .storage import Storage
from .tools.write_record import write_record_to
# DB
from .tools.pod_location import get_pod_location
from .tools.order_history import upsert_order_history
from .tools.job_task import upsert_job_task, update_job_task
from .tools.pre_assign import initialize_pre_assign_table, clear_pre_assign_table, insert_pre_assign
# from .live_advanced_table import start_gui

# Show full column content
pd.set_option('display.max_colwidth', None)

# Show all columns without truncation
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)  # Let it auto-expand

class Inventory(Universe):
    dimension = 60
    map = []
    landscape = None
    stop_and_go = 0
    total_energy = 0
    total_pod = 0
    total_turning = 0
    total_robot_idle = 0
    movement_channel = {}
    graph = None
    graph_pod = None

    def __init__(self):
        self._tick = 0  #current counter
        # self.ignored_types = ["pod", "station", "way-direction"]
        self.ignored_types = ["station", "way-direction"]
        self.tick_to_second = 0.25
        self.fast_train = (
            os.environ.get("RMFS_FAST_TRAIN", "0").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self.dynamic_job_update_enabled = (
            os.environ.get("RMFS_DYNAMIC_JOB_UPDATE", "1").strip().lower()
            not in {"0", "false", "no", "off"}
        )
        self._fast_pod_info_records = []
        self._fast_finished_orders = []
        self.job_queue: list[RobotJob] = []
        # Instance-level mutable state (prevents cross-instance contamination)
        self.map = []
        self.movement_channel = {}
        self.stop_and_go = 0
        self.total_energy = 0
        self.total_pod = 0
        self.total_turning = 0
        self.total_robot_idle = 0
        self.graph = None
        self.graph_pod = None
        self.landscape = Landscape(self.dimension)
        self.pod_manager = PodManager()
        self.station_manager = StationManager()
        self.storage_manager = StorageManager(self)
        self.order_manager = OrderManager()
        self.next_process_tick = 0
        self.intersection_manager = IntersectionManager(self.landscape.current_date_string)
        self.update_intersection_using_RL = False
        self.zoning = False
        self.robot_queue_order = {}
        self.preassign_dict = {}
        self.last_order = {}
        
        self.preassign_per_station = defaultdict(deque)
        # self.currently_picking = {}
        # # Shared wrapper for the DataFrame
        # self.shared_data = {"df": pd.DataFrame()}

        # # Start GUI in a thread
        # self.gui_thread = threading.Thread(target=start_gui, args=(self.shared_data,), daemon=True)
        # self.gui_thread.start()
        self.poa_podmatch = False
        self.poa_first = False  # preasign2 gajelas nih / F3
        self.poa_second = True  # Rika's Future-aware POA (no batching)
        self.poa_aisyahna = False  # Similarity-based order batching + POA

        self.pps_pileon = True
        self.pps_demand = False
        self.pps_rl = False       # When True, PPS is controlled by RL agent (PPSEnv)
        self.pps_picked_quantity = 0
        self.pps_pod_visits = 0
        self.joint_rl = False     # When True, both POA and PPS are controlled by JointEnv

        self.priority_order = False

        # Aisyahna's similarity-based batching parameters
        self.aisyahna_batch_interval = 5   # fire every t ticks
        self._aisyahna_last_batch_tick = 0  # last tick the batching fired


        if self.poa_second and not self.fast_train:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            initialize_pre_assign_table(timestamp)
            clear_pre_assign_table()
        super().__init__()

    @staticmethod
    def _normalize_sku_qty_dict(sku_qty):
        if not isinstance(sku_qty, dict):
            return {}

        normalized = {}
        for sku, qty in sku_qty.items():
            try:
                sku_key = int(sku)
            except (TypeError, ValueError):
                sku_key = sku
            try:
                qty_value = int(qty)
            except (TypeError, ValueError):
                qty_value = qty
            normalized[sku_key] = qty_value
        return normalized

    def addObject(self, object):
        if object.object_type == "robot":
            object._id = self.total_pod + 1
            self.total_pod += 1
        super().addObject(object)

    def addTrafficPolicyHistory(self, sender, target):
        if target not in self.movement_channel:
            self.movement_channel[target] = []
        self.movement_channel[target].append(sender)

    def getTrafficPolicyHistory(self, target):
        if target not in self.movement_channel:
            return []
        return self.movement_channel[target]

    def tick(self):
        # Get initial state
        result = super().generateResult()
        
        print(f"Current tick: {self._tick}")

        # Reset movement tracking
        self.movement_channel = {}
        
        # Process orders at scheduled intervals
        if int(self._tick) == self.next_process_tick:
            print(f"Processing orders at tick {self._tick}")
            self.find_new_orders()
            self.process_orders()
            if self.update_intersection_using_RL:
                self.intersection_manager.update_allowed_direction_using_q_model(int(self._tick))

        for queued_job in list(self.job_queue):
            self.update_robot_job_for_new_orders(queued_job)

        print(f"Current job queue length: {len(self.job_queue)}")

        if len(self.job_queue) > 0:
            job = self.job_queue[0]

            if job is not None:
                current_distance = float("inf")
                nearest_robot: Optional[Robot] = None

                for o in self.get_movable_objects():
                    if o.object_type == "robot" and (o.job is None or o.job.is_finished) and o.current_state == 'idle':
                        dist = calculateDistance(o.pos_x, o.pos_y, job.pod_coordinate.x, job.pod_coordinate.y)
                        if dist < current_distance:
                            nearest_robot = o
                            current_distance = dist

                if nearest_robot is not None:
                    self.job_queue.remove(job)  # Remove the selected job from the queue
                    print(f"Assigning job {job.pod}-{job.station_id} to robot {nearest_robot._id}")
                    nearest_robot.assign_job_and_set_move_to_take_pod(job)
                    if not self.fast_train:
                        for triplet in job.orders:
                            upsert_job_task(
                                pod_id=str(job.pod.pod_id),
                                order_id=str(triplet[0]),
                                sku=str(triplet[1]),
                                qty=str(triplet[2]),
                                status="otw",
                            )
            

        # Update object positions and collect metrics
        total_energy = 0
        total_turning = 0
        total_idle = 0
        for o in self.get_movable_objects():
            if isinstance(o, Robot):
                initial_velocity = o.velocity
                o.move()
                total_energy += o.energy_consumption
                total_turning += o.turning
                total_idle += (o.total_idle * 0.15)
                if o.velocity == 0 and initial_velocity > 0:
                    self.stop_and_go += 1

                # Add newly assigned compatible orders to active pod jobs before
                # deciding whether the pod has finished station processing.
                if o.job is not None and not o.job.is_finished:
                    self.update_robot_job_for_new_orders(o.job)

                # Handle job completion and replenishment
                if o.job is not None and o.job.picking_delay == 0 and not o.job.is_finished:
                    need_replenish_pod = self.finish_task_in_job(o.job)
                    if not self.fast_train:
                        for triplet in o.job.orders:
                            update_job_task(
                                pod_id=str(o.job.pod.pod_id),
                                order_id=str(triplet[0]),
                                sku=str(triplet[1]),
                                qty=str(triplet[2]),
                                status="finish",
                                finish_time=self._tick
                            )
                    if need_replenish_pod:
                        # pod: Pod = self.pod_manager.get_pod_by_coordinate(o.job.pod_coordinate.x, o.job.pod_coordinate.y)
                        pod: Pod = self.pod_manager.get_pod_by_id(o.job.pod.pod_id)
                        latest_pod_location = get_pod_location(pod.pod_id)
                        if latest_pod_location:
                            pod.pos_x, pod.pos_y = latest_pod_location
                        station_replenish = self.station_manager.find_available_replenish_station()
                        if station_replenish is not None:
                            station_replenish.add_pod(pod.pod_id)
                            new_job = RobotJob(pod.coordinate, station_id=station_replenish.station_id, pod=pod)
                            new_job.add_replenishment_task(pod)
                            o.assign_job_and_set_move_to_station(new_job)

                # Reset completed jobs
                if o.current_state == 'idle' and o.job is not None:
                    # self.pod_manager.mark_pod_available(o.job.pod_coordinate)
                    self.pod_manager.mark_pod_available(o.job.pod)
                    o.job = None
                
        # Update global metrics
        self.total_robot_idle = total_idle
        self.total_energy = total_energy
        self.total_turning = total_turning

        # Update process tick and intersection model
        if int(self._tick) == self.next_process_tick:
            self.next_process_tick += 1
            if self.update_intersection_using_RL:
                self.intersection_manager.update_model_after_execution(self._tick)

        # Increment tick
        self._tick += self.tick_to_second

        # Return updated state with station orders
        station_orders = self.get_station_orders_info()
        # with open('result.txt', 'a') as f:
        #     f.write(f"{result}")
        return [result, station_orders]

    def finish_task_in_job(self, job: RobotJob):
        job_station = self.station_manager.get_station_by_id(job.station_id)
        if job_station.is_picker_station():
            try:
                return self.finish_picking_task(job)
            except Exception as e:
                print(f"[ERROR] finish_picking_task for job {job.job_id}")
                print(f"[ERROR] for pod {job.pod} location {job.pod.coordinate}")
                raise e
        elif job_station.is_replenishment_station():
            try:
                return self.finish_replenishment_task(job)
            except Exception as e:
                print(f"[ERROR] finish_replenishment_task for job {job.job_id}")
                print(f"[ERROR] for pod {job.pod} location {job.pod.coordinate}")
                raise e
    
    def finish_picking_task(self, job: RobotJob):
        # pod: Pod = self.pod_manager.get_pod_by_coordinate(job.pod_coordinate.x, job.pod_coordinate.y)
        pod: Pod = self.pod_manager.get_pod_by_id(job.pod.pod_id)
        pod_info_records = self._fast_pod_info_records if self.fast_train else None
        pod_info_df = None if self.fast_train else pd.read_csv('pod_info.csv')
        sku_need_replenished = []
        for order_id, sku, quantity in job.orders:
            order: Order = self.order_manager.get_order_by_id(order_id)
            order.deliver_quantity(sku, quantity)
            print("order, sku, quantity :" ,order_id, sku, quantity)

            # Check for SKU Replenishment
            # sku is sku_id (String)
            
            sku, replenished_status = self.pod_manager.is_sku_need_replenished(sku)

            # SKU Replenished Triggered
            if(replenished_status == True): sku_need_replenished.append(sku)

            assign_order_df = pd.read_csv('assign_order.csv')
            assign_order_df.loc[((assign_order_df['order_id'] == order.order_id) & (assign_order_df['item_id'] == sku)), 'status'] = 1
            assign_order_df.loc[((assign_order_df['order_id'] == order.order_id) & (assign_order_df['item_id'] == sku)), 'order_finished'] = int(self._tick)
            assign_order_df.to_csv('assign_order.csv', index=False)
            new_row = {
                "pod_id": pod.pod_id,
                "item_id": sku,
                "qty": quantity,
                "order_id": order_id,
                "processed_time": int(self._tick),
                "task_type": 1
            }

            if self.fast_train:
                pod_info_records.append(new_row)
            else:
                new_row_df = pd.DataFrame([new_row])
                pod_info_df = pd.concat([pod_info_df, new_row_df], ignore_index=True)
            
            if order.is_order_completed():
                self.order_manager.finish_order(order_id, int(self._tick))
                station_id = order.station_id or job.station_id
                if order.station_id is None:
                    order.assign_station(station_id)
                station = self.station_manager.get_station_by_id(station_id)
                station.remove_order(order_id, order)
                self.insert_finished_order_to_csv(order)
                # DB
                # if not isinstance(order.order_id, int):
                    # raise AssertionError(f"WHAT? order {order} order_id {order.order_id} order_id {order_id}")
                if not self.fast_train:
                    upsert_order_history(order_id, order_finish_time=self._tick)
        station = self.station_manager.get_station_by_id(job.station_id)
        station.remove_pod(pod.pod_id)
        
        if not self.fast_train:
            pod_info_df.to_csv('pod_info.csv', index=False)
        # Replenishment baseline
        # job.is_finished = True
        job.set_job_finish()
        if len(sku_need_replenished) > 0:
            return True
        need_replenish_pod = pod.check_replenishment_needed()
        print(f"reple ga yaaa {need_replenish_pod}")
        # HACK
        # return False
        return need_replenish_pod
    
    def finish_replenishment_task(self, job: RobotJob):
        # pod: Pod = self.pod_manager.get_pod_by_coordinate(job.pod_coordinate.x, job.pod_coordinate.y)
        pod: Pod = self.pod_manager.get_pod_by_id(job.pod.pod_id)
        pod.replenish_all_skus()
        new_row = {
                "pod_id": pod.pod_id,
                "item_id": -1,
                "qty": -1,
                "order_id": -999,
                "processed_time": int(self._tick),
                "task_type": 2
            }

        if self.fast_train:
            self._fast_pod_info_records.append(new_row)
        else:
            pod_info_df = pd.read_csv('pod_info.csv')
            new_row_df = pd.DataFrame([new_row])
            pod_info_df = pd.concat([pod_info_df, new_row_df], ignore_index=True)
            pod_info_df.to_csv('pod_info.csv', index= False)
        # job.is_finished = True
        job.set_job_finish()
        station = self.station_manager.get_station_by_id(job.station_id)
        station.remove_pod(pod.pod_id)
        return False

    def insert_finished_order_to_csv(self, order: Order):
        if self.fast_train:
            self._fast_finished_orders.append(order)
            return
        header = ["order_id", "order_arrival", "process_start_time", "order_complete_time", "station_id"]
        data = [order.order_id, order.order_arrival, order.process_start_time, order.order_complete_time,
                order.station_id]

        self.write_to_csv("order-finished.csv", header, data)

    def find_new_orders(self):
        file_path = 'assign_order.csv'
        if os.path.exists(file_path):
            assign_order_df = pd.read_csv(file_path)
            # pass
        else:
            orders_df = pd.read_csv('generated_order.csv')
            assign_order_df = orders_df.copy()
            assign_order_df['assigned_station'] = None
            assign_order_df['assigned_pod'] = None
            assign_order_df['status'] = -3
            assign_order_df.to_csv('assign_order.csv', index=False)
        new_file_df = pd.read_csv(file_path)
                  
        current_second = self.next_process_tick
        previous_second = (self.next_process_tick - 1)

        # Filter orders that have arrived by the current second and have not been processed before
        new_orders = new_file_df[(new_file_df['order_arrival']<= current_second) & 
                               (new_file_df['order_arrival'] > previous_second) &
                               (new_file_df['status'] == -3)]
        grouped_orders = new_orders.groupby('order_id')

        for order_id, group in grouped_orders:
            order_items = group[['item_id', 'item_quantity']].to_dict('records')
            order = Order(order_id=order_id, order_arrival=current_second)

            # Add each item in the group to the order
            for item in order_items:
                order.add_sku(item['item_id'], item['item_quantity'])

            self.order_manager.add_order(order)
            # DB
            if not self.fast_train:
                upsert_order_history(order.order_id, arrival_time=self._tick)

        return new_orders

    def get_movable_objects(self):
        result = []
        for o in self._objects:
            if o.object_type not in self.ignored_types or self._tick == 0:
                result.append(o)

        return result

    def process_orders(self):
        # Joint RL controls both POA and PPS externally — skip everything except
        # robot job init and order start-processing
        if self.joint_rl:
            # Still need to start processing timer for assigned orders
            if os.path.exists('assign_order.csv'):
                for order in self.order_manager.unfinished_orders:
                    if order.station_id is None:
                        continue
                    if order.process_start_time <= 0:
                        order.start_processing(int(self._tick))
            return

        # Step 1: Robot job initialization
        robots_location = [
            [o.pos_x, o.pos_y] for o in self.get_movable_objects()
            if o.object_type == "robot" and (o.job is None or o.job.is_finished) and o.current_state == 'idle'
            and len(self.job_queue) > 0
        ]
        # Step 2: Trigger preassign logic
        if self.poa_first:
            advanced_table = self.get_advanced_table()

        # Step 3 & 4: POA — dispatch to the selected strategy (no order batching)
        if self.poa_aisyahna:
            # Aisyahna's method has its own timer-based batching
            if self._tick >= 1:
                self.aisyahna_poa()
        else:
            # No threshold batching — run POA whenever there is capacity and demand
            unassigned_count = sum(
                1 for o in self.order_manager.unfinished_orders
                if o.station_id is None and o.order_id not in self.order_manager.preassign_order_ids
            )
            total_empty_bin = self.get_total_empty_bin()
            if (
                unassigned_count > 0
                and sum(total_empty_bin.values()) >= 1
                and self._tick >= 1
            ):
                if self.poa_podmatch:
                    self.assign_order_old()
                if self.poa_first:
                    self.assign_order()
                if self.poa_second:
                    self.xxx()
        # Step 5: Record last order for each station
        if self.poa_first:
            for st in [v for k, v in self.station_manager.stations_by_id.items() if 'picker' in k]:
                self.last_order[st.station_id] = advanced_table.loc[advanced_table['station_id'] == st.station_id, 'order_id'].tolist()
            print(self.last_order)
        # Step 6: Start unfinished orders
        for order in self.order_manager.unfinished_orders:
            if order.station_id is None:
                continue
            if order.process_start_time <= 0:
                order.start_processing(int(self._tick))
        assign_order_df = pd.read_csv('assign_order.csv')
        assign_order_df.to_csv('assign_order.csv', index=False)
        # Step 7: Process PPS logic (skip when RL controls PPS)
        if self.pps_rl:
            return  # RL agent handles PPS externally via PPSEnv
        if self.pps_demand or self.pps_pileon:
            for station in filter(lambda s: s.station_type == 'picker' and len(s.incoming_pod) < 11, self.station_manager.stations):
                priority_orders, general_orders = {}, {}
                for order in station.orders:
                    remaining_skus = order.get_remaining_skus()
                    if 0 < len(remaining_skus) <= 2:
                        if self.priority_order:
                            priority_orders[order.order_id] = remaining_skus
                        general_orders[order.order_id] = remaining_skus
                    else:
                        general_orders[order.order_id] = remaining_skus
                print(f"[DEBUG] priority orders {priority_orders}")
                # Handle priority orders first
                if priority_orders:
                    pod_assigned = False
                    for order_id, remaining_skus in priority_orders.items():
                        # raise AssertionError(f"we have priority {priority_orders}")
                        idle_pods = {pod for pod in self.pod_manager.sku_to_pods.get(list(remaining_skus.keys())[0], []) if self.pod_manager.is_idle(pod.pod_id)}
                        # for pod_id in [k for k, v in self.pod_manager.pod_idle.items() if v]:
                        for pod in idle_pods:
                            pod_id = pod.pod_id
                            print(f"[DEBUG] pod_id {pod_id} with")
                            # pod = self.pod_manager.get_pod_by_id(pod_id)
                            can_fulfill = any(
                                sku in pod.skus and pod.skus[sku]["current_qty"] >= qty
                                for sku, qty in remaining_skus.items()
                            )
                            print(f"[DEBUG] can fulfill {can_fulfill}")
                            if can_fulfill:
                                sku_to_quantity = {sku: qty for sku, qty in remaining_skus.items()}
                                sku_to_order_map = {sku: [(order_id, qty)] for sku, qty in remaining_skus.items()}
                                job = self.add_picking_task_after_pps(station, pod, sku_to_order_map, sku_to_quantity)
                                self.job_queue.append(job)
                                for sku, qty in sku_to_quantity.items():
                                    upsert_job_task(
                                        pod_id=str(pod.pod_id),
                                        order_id=str(order_id),
                                        sku=str(sku),
                                        qty=str(qty),
                                        assigned_station=station.station_id,
                                        pod_assigned_time=self._tick,
                                        status="queue",
                                    )
                                # write_record_to("record_record.csv", [f"{self._tick:.2f}", 'job_append', pod, pod.coordinate], ['Time', 'Event', 'Pod ID', 'Location'])
                                pod_assigned = True
                                break
                        if pod_assigned:
                            break
                    if pod_assigned:
                        continue  # skip general orders if priority already assigned

                # Process general orders
                sku_to_quantity, sku_to_order_map = defaultdict(int), defaultdict(list)
                for o_id, remaining_skus in general_orders.items():
                    for sku, qty in remaining_skus.items():
                        sku_to_quantity[sku] += qty
                        sku_to_order_map[sku].append((o_id, qty))

                if not sku_to_quantity:
                    print(f"skipping pod search for station {station.station_id}")
                    continue

                # Pod selection
                if self.pps_demand:
                    backlog_skus = defaultdict(int)
                    for o in filter(lambda o: o.station_id is None and o.order_id not in self.order_manager.preassign_order_ids, self.order_manager.unfinished_orders):
                        for sku, q in o.skus.items():
                            backlog_skus[sku] += q["total_quantity"]
                    pod, score = self.find_best_pod(backlog_skus, list(sku_to_quantity.keys()), mode="demand")
                else:
                    pod, score = self.find_best_pod(sku_to_quantity, list(sku_to_quantity.keys()), mode="pile_on")

                if not pod:
                    continue

                job = self.add_picking_task_after_pps(station, pod, sku_to_order_map, sku_to_quantity)
                if len(job.orders) > 0:
                    self.job_queue.append(job)
                    for triplet in job.orders:
                        upsert_job_task(
                            pod_id=str(job.pod.pod_id),
                            order_id=str(triplet[0]),
                            sku=str(triplet[1]),
                            qty=str(triplet[2]),
                            assigned_station=station.station_id,
                            pod_assigned_time=self._tick,
                            status="queue",
                        )

    # def process_orders(self):
    #     robots_location = []
    #     for o in self.get_movable_objects():
    #         if len(self.job_queue) > 0:
    #             job: RobotJob = self.job_queue[0]

    #             if o.object_type == "robot" and (o.job is None or o.job.is_finished) and o.current_state == 'idle':
    #                 robots_location.append([o.pos_x, o.pos_y])

    #     if self.poa_first:
    #         advanced_table = self.get_advanced_table()  # di dalam sini ada proses preassign


    #     # misal kamu mau tau total keseluruhan -> x = sum(self.get_total_empty_bin().values())
    #     total_empty_bin = self.get_total_empty_bin()
    #     if sum(total_empty_bin.values()) >= 1 and self._tick >=1:
    #         if self.poa_podmatch:
    #             self.assign_order_old()
    #         if self.poa_first:
    #             self.assign_order()
    #         if self.poa_second:
    #             self.xxx()

    #     if self.poa_first:
    #         picking_station = [v for k, v in self.station_manager.stations_by_id.items() if 'picker' in k]
    #         for st in picking_station:
    #             self.last_order[st.station_id] = advanced_table.loc[advanced_table['station_id'] == st.station_id, 'order_id'].tolist()
    #         print(self.last_order)
    #     for order in self.order_manager.unfinished_orders:
    #         assign_order_df = pd.read_csv('assign_order.csv')
    #         if order.station_id is None:
    #             continue

    #         # print(f"[DEBUG] order {order.order_id} is not None and keep running the rest")
    #         if order.process_start_time <= 0:
    #             # print(f"[DEBUG] start_process {order.order_id}")
    #             order.start_processing(int(self._tick))
                
    #         assign_order_df.to_csv('assign_order.csv', index=False)

    #     if self.pps_demand:
    #         print("pps_demand")
    #         for station in [st for st in self.station_manager.stations if st.station_type == 'picker']:
    #             print(f"incoming_pod {station.station_id} {station.incoming_pod}")
    #             if len(station.incoming_pod) < 11:
    #                 priority_order = {}
    #                 general_order = {}
    #                 for order in station.orders:
    #                     remaining_skus = order.get_remaining_skus()
    #                     if len(remaining_skus) <= 2:
    #                         # priority_order[order.order_id] = remaining_skus
    #                         general_order[order.order_id] = remaining_skus
    #                     else:
    #                         general_order[order.order_id] = remaining_skus
    #                     # update0617 print(f"order {order.order_id} has remaining skus {remaining_skus}")
    #                 sku_to_quantity = defaultdict(int)
    #                 sku_to_list_order_id_and_quantity = defaultdict(list)
    #                 for o_id, remaining_skus in general_order.items():
    #                     for sku, qty in remaining_skus.items():
    #                         sku_to_quantity[sku] += qty
    #                         sku_to_list_order_id_and_quantity[sku].append((o_id, qty))
    #                 print(f"for station {station.station_id} with sku_to_quantity {sku_to_quantity}")
    #                 print(f"sku in station {station.skus_in_station}")
    #                 if not sku_to_quantity:
    #                     print(f"skipping pod search for station {station.station_id}")
    #                     continue

    #                 ## PPS Demand
    #                 backlog_skus = defaultdict(int)
    #                 unassigned_orders = [order for order in self.order_manager.unfinished_orders if 
    #                                      (order.station_id is None and order.order_id not in self.order_manager.preassign_order_ids)]
    #                 for o in unassigned_orders:
    #                     for sku, q in o.skus.items():
    #                         backlog_skus[sku] += q["total_quantity"]
    #                 highest_demand_on_pod, demand_score = self.find_pod_with_the_highest_demand(backlog_skus, list(sku_to_quantity.keys()))
    #                 print("highest_demand_on_pod", highest_demand_on_pod, "demand_score", demand_score)
    #                 if not highest_demand_on_pod:
    #                     print("assign nothing")
    #                     continue

    #                 job = self.add_picking_task_after_pps(
    #                     station,
    #                     highest_demand_on_pod,
    #                     sku_to_list_order_id_and_quantity,
    #                     sku_to_quantity
    #                 )

    #                 if len(job.orders) > 0:
    #                     self.job_queue.append(job)
    #                     write_record_to("record_record.csv", [f"{self._tick:.2f}", 'job_append', job.pod, job.pod.coordinate], ['Time', 'Event', 'Pod ID', 'Location'])



    #     if self.pps_pileon:
    #         print("pps_pileon")
    #         for station in [st for st in self.station_manager.stations if st.station_type == 'picker']:
    #             print(f"incoming_pod {station.station_id} {station.incoming_pod}")
    #             if len(station.incoming_pod) < 11:
    #                 priority_order = {}
    #                 general_order = {}
    #                 for order in station.orders:
    #                     remaining_skus = order.get_remaining_skus()
    #                     if len(remaining_skus) <= 2:
    #                         priority_order[order.order_id] = remaining_skus
    #                         # general_order[order.order_id] = remaining_skus
    #                     else:
    #                         general_order[order.order_id] = remaining_skus
    #                     # update0617 print(f"order {order.order_id} has remaining skus {remaining_skus}")
    #                 # if priority order, then blabla
    #                 # if not
    #                 sku_to_quantity = defaultdict(int)
    #                 sku_to_list_order_id_and_quantity = defaultdict(list)
    #                 # print("general_order", general_order)
    #                 for o_id, remaining_skus in general_order.items():
    #                     for sku, qty in remaining_skus.items():
    #                         sku_to_quantity[sku] += qty
    #                         sku_to_list_order_id_and_quantity[sku].append((o_id, qty))
    #                 print(f"for station {station.station_id} with sku_to_quantity {sku_to_quantity}")
    #                 print(f"sku in station {station.skus_in_station}")
    #                 if not sku_to_quantity:
    #                     print(f"skipping pod search for station {station.station_id}")
    #                     continue
    #                 ## PPS Pile On
    #                 highest_pile_on_pod, pile_on_score = self.find_pod_with_the_highest_pile_on(sku_to_quantity)
    #                 print("highest_pile_on_pod", highest_pile_on_pod, "pile_on_score", pile_on_score)
                    
    #                 ## try to fix teleport
    #                 # highest_pile_on_pod = self.pod_manager.get_pod_by_id(highest_pile_on_pod.pod_id)
    #                 job = self.add_picking_task_after_pps(
    #                     station,
    #                     highest_pile_on_pod,
    #                     sku_to_list_order_id_and_quantity,
    #                     sku_to_quantity
    #                 )

    #                 if len(job.orders) > 0:
    #                     self.job_queue.append(job)
    #                     for triplet in job.orders:
    #                         upsert_job_task(
    #                             pod_id=str(job.pod.pod_id),
    #                             order_id=str(triplet[0]),
    #                             sku=str(triplet[1]),
    #                             qty=str(triplet[2]),
    #                             assigned_station=station.station_id,
    #                             pod_assigned_time=self._tick,
    #                             status="queue",
    #                         )
    #                     write_record_to("record_record.csv", [f"{self._tick:.2f}", 'job_append', job.pod, job.pod.coordinate], ['Time', 'Event', 'Pod ID', 'Location'])


    #                 # TODO: order.commit_quantity (to tell that that order remaining skus are decreased)
    #                 # TODO: pod.pick_sku(sku, quantity_to_take) (reduct the item inside pod)
    #                 # TODO: self.pod_manager.reduct_sku_data(sku, quantity_to_take) (reduct item in global stock list)
    #                 # TODO: station.add_pod(pod.pod_id)
    #                 # TODO: pod.station = station
    #                 # TODO: update assign_order_df or whatever...
    #                 # TODO: self.pod_manager.mark_pod_not_available(pod.coordinate) (set the pod to not idle)
    #                 # TODO: station.reduce_sku_from_station(sku, quantity_to_take) (reduce the remaining skus in station)
    #                 # TODO: job.add_picking_task(order.order_id, sku, quantity_to_take) (the picking task list inside robotjob)
                    
    #                 # TODO: self.job_queue.append(job)
    def find_best_pod(
        self, 
        sku_to_quantity: dict, 
        relevant_skus: list, 
        mode: str = "pile_on"  # or "demand"
    ):  # type: ignore

        # Step 1: Collect pod candidates from relevant skus
        pod_candidates: set[Pod] = set()
        for sku in relevant_skus:
            pod_candidates.update(self.pod_manager.sku_to_pods.get(sku, []))

        # Step 2: Filter only idle pods
        pod_candidates = {pod for pod in pod_candidates if self.pod_manager.is_idle(pod.pod_id)}

        print(f"[DEBUG] Checking candidates for mode={mode} skus={relevant_skus}")
        print(f"[DEBUG] pod_candidates={pod_candidates}")

        if not pod_candidates:
            return None, -1

        # Step 3: Score function
        def score_pod(pod: Pod) -> int:
            score = 0
            for sku, req_qty in sku_to_quantity.items():
                if sku in pod.skus:
                    current_total = pod.skus[sku]['current_qty']
                    score += min(current_total, req_qty)
            return score

        # Step 4: Rank pods by score
        ranked_pods = sorted(
            [(pod, score_pod(pod)) for pod in pod_candidates],
            key=lambda x: x[1],
            reverse=True
        )

        print(f"[DEBUG] ranked_pods (mode={mode}) = {ranked_pods}")
        return ranked_pods[0]

    def add_picking_task_after_pps(self, station: Station, pod: Pod, sku_to_list_order_id_and_quantity: dict, sku_to_quantity: dict):
        latest_pod_location = get_pod_location(pod.pod_id)
        if latest_pod_location:
            pod.pos_x, pod.pos_y = latest_pod_location
        job = RobotJob(pod.coordinate, station_id=station.station_id, pod=pod)
        for sku in sku_to_list_order_id_and_quantity:
            # sort based on the least quantity for each sku
            sku_to_list_order_id_and_quantity[sku] = sorted(sku_to_list_order_id_and_quantity[sku], key=lambda x: x[1])
            if sku in pod.skus:
                if pod.get_quantity(sku) >= sku_to_quantity[sku]:
                    quantity_to_take = sku_to_quantity[sku]
                    # set the order list for job
                else:
                    quantity_to_take = pod.get_quantity(sku)
                    # set the order list for job
                
                tmp = quantity_to_take
                for o_id, qty in sku_to_list_order_id_and_quantity[sku]:
                    if tmp <= 0:
                        break
                    order = self.order_manager.get_order_by_id(o_id)
                    if order.station_id is None:
                        order.assign_station(station.station_id)
                    # order.commit_quantity
                    order.commit_quantity(sku, min(qty, tmp))
                    # job.add_picking_tas
                    job.add_picking_task(o_id, sku, min(qty, tmp))
                    tmp = tmp - min(qty, tmp)
                
                # pod.pick_sku
                pod.pick_sku(sku, quantity_to_take)
                # self.pod_manager
                self.pod_manager.reduce_sku_data(sku, quantity_to_take)
                # station.reduce_sku
                station.reduce_sku_from_station(sku, quantity_to_take)

        station.add_pod(pod.pod_id)
        pod.station = station
        # print(f"[DEBUG] assign job pod {pod.id} coordinate {pod.coordinate}")
        self.pod_manager.mark_pod_not_available(pod)
        picked_quantity = sum(qty for _order_id, _sku, qty in job.orders)
        if picked_quantity > 0:
            self.pps_picked_quantity = getattr(self, "pps_picked_quantity", 0) + picked_quantity
            self.pps_pod_visits = getattr(self, "pps_pod_visits", 0) + 1
        return job

    def find_pod_with_the_highest_pile_on(self, sku_to_quantity: dict) -> (Pod, int): # type: ignore
        # dict of order: {order_id_1: {sku1: X, sku2: Y}, order_id_2: {...}}    
        sku_list = sku_to_quantity.keys()
        pod_candidates: set[Pod] = set()
        for sku in sku_list:
            pod_candidates.update(self.pod_manager.sku_to_pods.get(sku, []))

        print(f"checking candicate for sku {sku_list}")
        print(f"pod_candidates {pod_candidates}")

        def pile_on_score(pod: Pod):
            # if pod.is_idle:
            if self.pod_manager.is_idle(pod.pod_id):
                # print(f"pod {pod} is idle")
                score = 0
                for sku, req_qty in sku_to_quantity.items():
                    if sku in pod.skus:
                        current_total = pod.skus[sku]['current_qty']
                        score += min(current_total, req_qty)  # Only count up to what's needed
            else:
                # print(f"pod {pod} is NOT idle")
                score = -1
            return score

        ranked_pods = sorted(
            [(pod, pile_on_score(pod)) for pod in pod_candidates],
            key=lambda x: x[1],
            reverse=True
        )
        print("ranked_pods", ranked_pods)
        return ranked_pods[0]
    
    def find_pod_with_the_highest_demand(self, sku_to_quantity: dict, station_unfinished_skus: list) -> (Pod, int): # type: ignore
        # dict of order: {order_id_1: {sku1: X, sku2: Y}, order_id_2: {...}}    
        pod_candidates: set[Pod] = set()
        for sku in station_unfinished_skus:
            pod_candidates.update(self.pod_manager.sku_to_pods.get(sku, []))
        # filter the pod status
        # pod_candidates = {po for po in pod_candidates if po.is_idle}
        pod_candidates = {po for po in pod_candidates if self.pod_manager.is_idle(po.pod_id)}
        print(f"checking candidate for sku {station_unfinished_skus}")
        print(f"pod_candidates {pod_candidates}")

        # for early stage, if empty, then assign random ?
        if not pod_candidates:
            return None, -1
            pod_candidates.update([po for po in self.pod_manager.pods if po.is_idle])

        def demand_score(pod: Pod):
            # if pod.is_idle:
            if self.pod_manager.is_idle(pod.pod_id):
                # print(f"pod {pod} is idle")
                score = 0
                for sku, req_qty in sku_to_quantity.items():
                    if sku in pod.skus:
                        current_total = pod.skus[sku]['current_qty']
                        score += min(current_total, req_qty)  # Only count up to what's needed
            else:
                # print(f"pod {pod} is NOT idle")
                score = -1
            return score

        ranked_pods = sorted(
            [(pod, demand_score(pod)) for pod in pod_candidates],
            key=lambda x: x[1],
            reverse=True
        )
        print("ranked_pods", ranked_pods)
        return ranked_pods[0]

    def write_to_csv(self, filename, header, data):
        if self.fast_train:
            return
        # Use CWD-relative 'output/' so SubprocVecEnv workers each have their own.
        folder_path = os.path.join(os.getcwd(), 'output')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        filename = os.path.join(folder_path, filename)
        file_exists = os.path.exists(filename)

        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(data)

    def get_station_orders_info(self):
        station_orders = []
        for station in sorted(self.station_manager.stations, key=lambda x: x.station_id):
            if station.is_picker_station():
                order_list = ', '.join(map(str, station.order_ids)) if station.order_ids else "Empty"
                station_orders.append(order_list)
        while len(station_orders) < 3:
            station_orders.append("Empty")
        return station_orders

    def generateResult(self):
        result = super().generateResult()
        station_orders = self.get_station_orders_info()
        return [result, station_orders]

    def get_fulfilment_table(self, mode="FS", excludes=[]):
        # MODE: 
        # FS=fully supplied 
        # OTW= only on incoming pod - pod in stations
        # F3=incoming pod - 3 queue

        # station_ids = [station.station_id for station in self.station_manager.stations]
        # order_ids = [order.order_id for order in self.order_manager.unfinished_orders]

        picking_stations = [station for station in self.station_manager.stations if station.station_type == "picker"]

        # Gather all assigned order IDs across all stations
        # assigned_order_ids = {order_id for station in self.station_manager.stations for order_id in station.order_ids}

        # Filter orders whose order_id is NOT in assigned_order_ids
        # unassigned_orders = [order for order in self.order_manager.unfinished_orders if order.order_id not in assigned_order_ids]
        # unassigned_orders = [order for order in self.order_manager.unfinished_orders if (order.station_id is None)]
        ## Activate this only if you use advanced table
        unassigned_orders = [order for order in self.order_manager.unfinished_orders if (order.station_id is None and order.order_id not in self.order_manager.preassign_order_ids)]
        
        picking_station_ids = [station.station_id for station in picking_stations]
        unassigned_order_ids = [order.order_id for order in unassigned_orders]

        # Initialize fulfillment matrix with zeros
        fulfilment_matrix = pd.DataFrame(0.0, index=unassigned_order_ids, columns=picking_station_ids)

        for station in picking_stations:
            # Build station SKU availability from incoming pods
            station_sku_quantity = {}
            if mode == "FS":
                list_of_pods = station.incoming_pod
            elif mode == "OTW":
                # TODO: incoming pods - is in station
                robots_otw = [o for o in self.get_movable_objects() if 
                          o.object_type == "robot" 
                          and o.job 
                          and o.job.pod.pod_id in station.incoming_pod
                          and not o.is_in_station_path()]
                list_of_pods = [o.job.pod.pod_id for o in robots_otw]
            elif mode == "F3":
                # TODO: incoming pods - self.robot_queue_order
                robots_otw = [o for o in self.get_movable_objects() if 
                          o.object_type == "robot" 
                          and o.job 
                          and o.job.pod.pod_id in station.incoming_pod
                          and o not in excludes]
                list_of_pods = [o.job.pod.pod_id for o in robots_otw]
            for pod_id in list_of_pods:
                pod = self.pod_manager.get_pod_by_id(pod_id)
                for sku, details in pod.skus.items():
                    if details['current_qty'] > 0:
                        station_sku_quantity[sku] = station_sku_quantity.get(sku, 0) + details['current_qty']

            for order in unassigned_orders:
                total_order_qty = sum([x.get('total_quantity') for x in order.skus.values()])
                fulfilled_qty = 0
                for sku, val in order.skus.items():
                    available_qty = station_sku_quantity.get(sku, 0)
                    fulfilled_qty += min(val.get('total_quantity'), available_qty)

                fulfillment_rate = fulfilled_qty / total_order_qty if total_order_qty > 0 else 0.0
                fulfilment_matrix.at[order.order_id, station.station_id] = fulfillment_rate

        return fulfilment_matrix

    def get_total_empty_bin(self):
        bin_dict = {}
        picking_stations = [station for station in self.station_manager.stations if station.station_type == "picker"]
        for station in picking_stations:
            bin_dict[station.station_id] = station.max_orders - len(station.order_ids)
        return bin_dict
    
    def assign_order_with_advanced_table(self, df):
        empty_bin_dict = self.get_total_empty_bin()
        final_selection = {}
        for picker, count in empty_bin_dict.items():
            # for the order_id list in df, if it's not in station
            # self.last_order[picker]
            # self.station_manager.get_station_by_id(picker).order_ids
            try:
                print(f"count: {count}")
                list_of_just_finished_order = [
                    oid for oid in self.last_order[picker] if oid not in df.loc[df['station_id'] == picker, 'order_id'].tolist()
                ]
                print(f"list_of_just_finished_order: {list_of_just_finished_order}")
                list_of_pre_assigned_order = [
                    self.preassign_dict[k] for k in list_of_just_finished_order
                ]
                print(f"preassign_dict: {self.preassign_dict}")
                print(f"list_of_pre_assigned_order: {list_of_pre_assigned_order}")
                if len(list_of_pre_assigned_order) != count:
                    return self.assign_order()
                for n in list_of_pre_assigned_order:
                    final_selection.setdefault(picker, []).append(n)
            except Exception as e:
                print(f"exception with e: {type(e).__name__} {e}")
                return self.assign_order()
            # candidates = df.loc[df['station_id'] == picker, 'pre_assign'].tolist()
            # candidates = [c for c in candidates if c]
            # if not candidates:
            #     return self.assign_order()
                
            # for n in range(count):
            #     final_selection.setdefault(picker, []).append(candidates[0])
            #     del candidates[0]
        print(f"final_selection: {final_selection}")
        self.put_order_to_picking_station(final_selection)

        return final_selection

    def assign_order(self):
        # TODO: for advanced table version, just use the preassign as the assign
        fulfilment_table = self.get_fulfilment_table(mode="OTW")
        empty_bin_dict = self.get_total_empty_bin()

        print("assign_order is triggered")
        print(fulfilment_table)
        print(empty_bin_dict)

        # Step 1: Flatten all candidate values with their source column
        candidates = []
        for picker, count in empty_bin_dict.items():
            top_rows = fulfilment_table[picker].sort_values(ascending=False).head(count * 3)  # get more to allow fallback if conflict
            for index, value in top_rows.items():
                candidates.append({'index': index, 'picker': picker, 'value': value})

        # Step 2: Sort all candidates by value descending
        candidates = sorted(candidates, key=lambda x: x['value'], reverse=True)

        # Step 3: Pick best combination without duplicate indices
        final_selection = {}
        used_indices = set()

        for candidate in candidates:
            picker = candidate['picker']
            index = candidate['index']

            if index in used_indices:
                continue
            if empty_bin_dict[picker] > 0:
                final_selection.setdefault(picker, []).append(index)
                empty_bin_dict[picker] -= 1
                used_indices.add(index)

            # Stop early if all picks are satisfied
            if all(v == 0 for v in empty_bin_dict.values()):
                break
        
        print("Final selection:", final_selection)

        # Put the decision into action
        self.put_order_to_picking_station(final_selection)

        return final_selection
    
    def assign_order_old(self): # buat yang baseline
        fulfilment_table = self.get_fulfilment_table("FS")
        empty_bin_dict = self.get_total_empty_bin()

        print("assign_order is triggered")
        print(fulfilment_table)
        print(empty_bin_dict)

        # Step 1: Flatten all candidate values with their source column
        candidates = []
        for picker, count in empty_bin_dict.items():
            top_rows = fulfilment_table[picker].sort_values(ascending=False).head(count * 3)  # get more to allow fallback if conflict
            for index, value in top_rows.items():
                candidates.append({'index': index, 'picker': picker, 'value': value})

        # Step 2: Sort all candidates by value descending
        candidates = sorted(candidates, key=lambda x: x['value'], reverse=True)

        # Step 3: Pick best combination without duplicate indices
        final_selection = {}
        used_indices = set()

        for candidate in candidates:
            picker = candidate['picker']
            index = candidate['index']

            if index in used_indices:
                continue
            if empty_bin_dict[picker] > 0:
                final_selection.setdefault(picker, []).append(index)
                empty_bin_dict[picker] -= 1
                used_indices.add(index)

            # Stop early if all picks are satisfied
            if all(v == 0 for v in empty_bin_dict.values()):
                break
        
        print("Final selection:", final_selection)

        # Put the decision into action
        self.put_order_to_picking_station(final_selection)

        return final_selection
        
    def put_order_to_picking_station(self, final_selection):
        assign_order_df = pd.read_csv('assign_order.csv')

        for picker_name, order_ids in final_selection.items():
            for order_id in order_ids:
                order = self.order_manager.get_order_by_id(order_id)
                order.assign_station(picker_name)
                self.station_manager.get_station_by_id(picker_name).add_order(order_id, order)

                assign_order_df.loc[assign_order_df['order_id'] == order.order_id, 'assigned_station'] = picker_name
                assign_order_df.loc[assign_order_df['order_id'] == order.order_id, 'status'] = -1
                # DB
                if not self.fast_train:
                    upsert_order_history(order_id, assigned_station=picker_name, order_assigned_time=self._tick)
            
        assign_order_df.to_csv('assign_order.csv', index=False)

    @staticmethod
    def _calculate_two_coordinates(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    
    def distance_robot_to_station(self, robot: Robot, station: Station):
        return self._calculate_two_coordinates((robot.pos_x, robot.pos_y), (station.coordinate.x, station.coordinate.y))

    def sort_pod_order(self, robots: List[Robot], station: Station):
        print("inside sort_pod_order")
        print(f"robot inside: {robots}")
        others = []
        current = None
        one_right = None
        two_right = None
        one_up = None
        two_up = None
        one, two, three = None, None, None
        station_coordinate = station.coordinate
        for r in robots:
            if r.pos_x == station_coordinate.x and r.pos_y == station_coordinate.y:
                current = r
            elif r.pos_x == station_coordinate.x + 1 and r.pos_y == station_coordinate.y:
                one_right = r
            elif math.floor(r.pos_x) == station_coordinate.x + 2 and r.pos_y == station_coordinate.y:
                two_right = r
            elif r.pos_x == station_coordinate.x and math.floor(r.pos_y) == station_coordinate.y + 1:
                one_up = r
            elif r.pos_x == station_coordinate.x and math.floor(r.pos_y) == station_coordinate.y + 2:
                two_up = r
            else:
                others.append(
                    (r, self.distance_robot_to_station(r, station))
                )
        print(f"total others: {others}")
        # TODO: sort others according to the distance
        if current:
            one = current
            if one_up and not one_right:
                two = one_up
                if two_up:
                    three = two_up
                else:
                    three = others[0][0]
                return (one, two, three)
            elif one_right:
                two = one_right
                if two_right:
                    three = two_right
                else:
                    three = others[0][0]
                return (one, two, three)
            else:
                if two_up:
                    two = two_up
                    three = others[0][0]
                else:
                    two = others[0][0]
                    three = others[1][0]
                return (one, two, three)
        else:
            return (None, None, None)

    def get_advanced_table_only(self):
        picking_stations = [station for station in self.station_manager.stations if station.station_type == "picker"]
        if not self.robot_queue_order:
            self.robot_queue_order = {picker.station_id: [] for picker in picking_stations}

        df_dicts = []
        for picker in picking_stations:
            station_id = picker.station_id
            robots_otw = [o for o in self.get_movable_objects() if 
                          o.object_type == "robot" 
                          and o.job 
                          and o.job.pod.pod_id in picker.incoming_pod]
            
            inside_station = []
            currently_picking = None
            for r in robots_otw:
                if r.is_being_process_on_station():
                    currently_picking = r
                
                if r.is_in_station_path():
                    inside_station.append(r)
                    if r.id not in self.robot_queue_order.get(station_id, []):
                        # print(f"adding robot queue {r} to {station_id}")
                        self.robot_queue_order[station_id].append(r.id)
                    print(f"{r} is_in_station_path {station_id}")

            for rid in self.robot_queue_order[station_id]:
                if rid not in [x.id for x in inside_station]:
                    # print(f"removing robot queue {r} from {station_id}")
                    self.robot_queue_order[station_id].remove(rid)
            print(f"current robot inside station {station_id}: {self.robot_queue_order[station_id]}")
            my_robot_queue_order = [None] * len(self.robot_queue_order[station_id])
            for n, rid in enumerate(self.robot_queue_order[station_id]):
                my_robot_queue_order[n] = [o for o in self.get_movable_objects() if o.object_type == "robot" and o.id == rid]
                my_robot_queue_order[n] = my_robot_queue_order[n][0] if my_robot_queue_order[n] else None
            if len(my_robot_queue_order) >= 3:
                first_queue = my_robot_queue_order[0]
                second_queue = my_robot_queue_order[1]
                third_queue = my_robot_queue_order[2]
            elif len(my_robot_queue_order) == 2:
                first_queue = my_robot_queue_order[0]
                second_queue = my_robot_queue_order[1]
                third_queue = None
            elif len(my_robot_queue_order) == 1:
                first_queue = my_robot_queue_order[0]
                second_queue = None
                third_queue = None
            else:
                first_queue = None
                second_queue = None
                third_queue = None
            for order in picker.orders:
                order_id = order.order_id
                unpicked_skus = order.get_unpicked_skus()
                df_dicts.append({
                    "station_id": station_id,
                    "order_id": order_id,
                    "unpicked_skus": self._normalize_sku_qty_dict(unpicked_skus),
                    # "robot_inside_station": self.robot_queue_order[station_id],
                    "pod_1": first_queue,
                    "pod_2": second_queue,
                    "pod_3": third_queue,
                    "occupied_1": first_queue.job.orders if (first_queue and first_queue.job) else None,
                    "occupied_2": second_queue.job.orders if (second_queue and second_queue.job) else None,
                    "occupied_3": third_queue.job.orders if (third_queue and third_queue.job) else None,
                    "next_bin_avail": None,
                    "pre_assign": self.preassign_dict.get(order_id, None)
                })
        df = pd.DataFrame(df_dicts)
        df = self.forcast_next_bin_avail(df)
        return df

    def get_advanced_table(self):
        picking_stations = [station for station in self.station_manager.stations if station.station_type == "picker"]
        if not self.robot_queue_order:
            self.robot_queue_order = {picker.station_id: [] for picker in picking_stations}
        # if not self.currently_picking:
        #     self.currently_picking = {picker.station_id: None for picker in picking_stations}
        # print(f"PICKING STATION {picking_stations[0].station_id}")
        # print(f"ORDER IDS {picking_stations[0].order_ids}")
        # for o in picking_stations[0].orders:
        #     print(f"ORDER {o.order_id} has")
        #     print(f"SKUS {o.skus}")
        #     print(f"GET_REMAINING_SKUS {o.get_remaining_skus()}")

        # print(f"PICKING STATION {picking_stations[0].station_id} {picking_stations[0].coordinate}")
        # print(f"INCOMING POD {picking_stations[0].incoming_pod}")
        # print(f"JOB_RUNNING")
        # robots_otw_picking_station = [o for o in self.get_movable_objects() if o.object_type == "robot" and o.job and o.job.pod.pod_id in picking_stations[0].incoming_pod]
        # for r in robots_otw_picking_station:
        #     print(f"sending {r.job.pod.pod_id} to {r.job.station_id} status {r.current_state}")
        # print(f"CURRENTLY PICKING")
        # currently_picking = [r for r in robots_otw_picking_station if r.is_being_process_on_station()]
        # for r in currently_picking:
        #     print(f"sending {r.job.pod.pod_id} to {r.job.station_id} status {r.current_state} {r.pos_x:.2f},{r.pos_y:.2f}")
        # print(f"JOB_QUEUE")
        # for rj in self.job_queue:
        #     print(rj)

        df_dicts = []
        for picker in picking_stations:
            station_id = picker.station_id
            # if self.currently_picking[station_id]:
            #     print(f">>> CURRENT STATUS in {station_id}<<<")
            #     my_robot = [o for o in self.get_movable_objects() if o.object_type == "robot" and o.id == self.currently_picking[station_id]]
            #     my_robot = my_robot[0] if my_robot else None
            #     print(f"{my_robot}")
            #     print(f"ID: {self.currently_picking[station_id]}")
            #     print(f"picking delay {my_robot.job.picking_delay}")
            #     print(f"state {my_robot.current_state}")
            robots_otw = [o for o in self.get_movable_objects() if 
                          o.object_type == "robot" 
                          and o.job 
                          and o.job.pod.pod_id in picker.incoming_pod]
            
            inside_station = []
            currently_picking = None
            for r in robots_otw:
                if r.is_being_process_on_station():
                    currently_picking = r
                    # self.currently_picking[station_id] = r.id
                    # print(f"{r} is_being_process {station_id}")
                    # print(f"ID: {r.id}")
                    # print(f"picking delay {r.job.picking_delay}")
                    # print(f"state {r.current_state}")
                    # print(f"location: {r.pos_x}, {r.pos_y}")
                    # print(f"station location: {picker.coordinate}")
                
                if r.is_in_station_path():
                    inside_station.append(r)
                    if r.id not in self.robot_queue_order.get(station_id, []):
                        # print(f"adding robot queue {r} to {station_id}")
                        self.robot_queue_order[station_id].append(r.id)
                    print(f"{r} is_in_station_path {station_id}")

            for rid in self.robot_queue_order[station_id]:
                if rid not in [x.id for x in inside_station]:
                    # print(f"removing robot queue {r} from {station_id}")
                    self.robot_queue_order[station_id].remove(rid)
            print(f"current robot inside station {station_id}: {self.robot_queue_order[station_id]}")
            my_robot_queue_order = [None] * len(self.robot_queue_order[station_id])
            for n, rid in enumerate(self.robot_queue_order[station_id]):
                my_robot_queue_order[n] = [o for o in self.get_movable_objects() if o.object_type == "robot" and o.id == rid]
                my_robot_queue_order[n] = my_robot_queue_order[n][0] if my_robot_queue_order[n] else None
            if len(my_robot_queue_order) >= 3:
                first_queue = my_robot_queue_order[0]
                second_queue = my_robot_queue_order[1]
                third_queue = my_robot_queue_order[2]
            elif len(my_robot_queue_order) == 2:
                first_queue = my_robot_queue_order[0]
                second_queue = my_robot_queue_order[1]
                third_queue = None
            elif len(my_robot_queue_order) == 1:
                first_queue = my_robot_queue_order[0]
                second_queue = None
                third_queue = None
            else:
                first_queue = None
                second_queue = None
                third_queue = None
            for order in picker.orders:
                order_id = order.order_id
                unpicked_skus = order.get_unpicked_skus()
                df_dicts.append({
                    "station_id": station_id,
                    "order_id": order_id,
                    "unpicked_skus": self._normalize_sku_qty_dict(unpicked_skus),
                    # "robot_inside_station": self.robot_queue_order[station_id],
                    "pod_1": first_queue,
                    "pod_2": second_queue,
                    "pod_3": third_queue,
                    "occupied_1": first_queue.job.orders if (first_queue and first_queue.job) else None,
                    "occupied_2": second_queue.job.orders if (second_queue and second_queue.job) else None,
                    "occupied_3": third_queue.job.orders if (third_queue and third_queue.job) else None,
                    "next_bin_avail": None,
                    "pre_assign": self.preassign_dict.get(order_id, None)
                })
        df = pd.DataFrame(df_dicts)
        df = self.forcast_next_bin_avail(df)
        df = self.pre_assign_order(df)
        print(df)
        return df

    def forcast_next_bin_avail(self, df):
        def parse_sku_qty_dict(value):
            if isinstance(value, dict):
                return self._normalize_sku_qty_dict(value)
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return {}
            if isinstance(value, str):
                try:
                    return self._normalize_sku_qty_dict(ast.literal_eval(value))
                except (ValueError, SyntaxError):
                    cleaned = re.sub(
                        r"(?:np\.)?(?:int64|int32|int16|int8)\((-?\d+)\)",
                        r"\1",
                        value,
                    )
                    cleaned = re.sub(
                        r"(?:np\.)?(?:float64|float32)\((-?\d+(?:\.\d+)?)\)",
                        r"\1",
                        cleaned,
                    )
                    try:
                        return self._normalize_sku_qty_dict(ast.literal_eval(cleaned))
                    except (ValueError, SyntaxError):
                        print(f"[WARN] Could not parse unpicked_skus value: {value}")
                        return {}
            return {}

        def is_fulfilled(row):
            required = row['unpicked_skus']
            # Flatten all occupied bins into a list
            all_occupied = []
            for col in ['occupied_1', 'occupied_2', 'occupied_3']:
                val = row.get(col)
                if isinstance(val, list):
                    all_occupied.extend(val)
            
            # Sum quantities by SKU
            available = defaultdict(int)
            for _, sku, qty in all_occupied:
                available[sku] += qty

            # Check if every required SKU has enough quantity
            for sku, req_qty in required.items():
                if available[sku] < req_qty:
                    return False
            return True
        df['unpicked_skus'] = df['unpicked_skus'].apply(parse_sku_qty_dict)
        df['next_bin_avail'] = df.apply(is_fulfilled, axis=1)
        return df
    
    def pre_assign_order(self, df):
        """
        For each row where 'next_bin_avail' is True and 'pre_assign' is None,
        call choose_order with (station_id, pod_1, pod_2, pod_3),
        and store the result in 'pre_assign'.
        """
        print("PREASSIGN IS CALLED !!!!!!!")
        mask = (df['next_bin_avail'] == True) & (df['pre_assign'].isna())  # noqa: E712
        print(mask)
        df.loc[mask, 'pre_assign'] = df[mask].apply(
            lambda row: self.choose_order(row['station_id'], row['order_id'], row['pod_1'], row['pod_2'], row['pod_3']),
            axis=1
        )
        return df

    # Example implementation
    def choose_order(self, station_id: str, order_id: int, pod_1: Pod, pod_2: Pod, pod_3: Pod):
        print("CHOOSE ORDER IS CALLED !!!!!")
        df = self.get_fulfilment_table(mode="F3", excludes=[pod_1, pod_2, pod_3])
        print("\n\nfulfillment table during pre-assignment")
        print(df)
        print("\n\n")
        if df.empty:
            print("[PRE-ASSIGN FULFULLMENT TABLE IS INVALID!]")
            print(f"for station {station_id}")
            print(f"for order_id {order_id}")
            unassigned_orders = [order for order in self.order_manager.unfinished_orders if (order.station_id is None and order.order_id not in self.order_manager.preassign_order_ids)]
            unassigned_order_ids = [order.order_id for order in unassigned_orders]
            print(f"unassigned_order_ids {unassigned_order_ids}")
            print(f"pod queue {pod_1.pod_id}, {pod_2.pod_id}, {pod_3.pod_id}")
        df.sort_values(by=station_id, ascending=False, inplace=True)
        val = df.index[0]
        self.order_manager.preassign_order_ids.append(val)
        self.preassign_dict[order_id] = int(val)
        print(f" VALUE {val} {df.iloc[0]}")
        return df.index[0]

    # ------------------------------------------------------------------ #
    #  Aisyahna's Similarity-Based Order Batching + POA                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _jaccard_similarity(skus_a: set, skus_b: set) -> float:
        """Sim(A,B) = (shared / |A|) * (shared / |B|)"""
        if not skus_a or not skus_b:
            return 0.0
        shared = len(skus_a & skus_b)
        return (shared / len(skus_a)) * (shared / len(skus_b))

    def aisyahna_poa(self):
        """Full similarity-based order batching and POA (Steps 2-9)."""

        # -- Step 2: Check if batching timer should fire --
        if self._tick - self._aisyahna_last_batch_tick < self.aisyahna_batch_interval:
            return  # not time yet
        self._aisyahna_last_batch_tick = self._tick

        # -- Identify available stations and free bins --
        picking_stations = [
            s for s in self.station_manager.stations if s.station_type == 'picker'
        ]
        free_bins = {}  # station_id -> int
        for st in picking_stations:
            fb = st.max_orders - len(st.order_ids)
            if fb > 0:
                free_bins[st.station_id] = fb

        if not free_bins:
            return  # no station has a free bin

        total_assignable = sum(free_bins.values())

        # -- Step 3: Take candidate orders from the pool --
        unassigned_orders = [
            o for o in self.order_manager.unfinished_orders
            if o.station_id is None
               and o.order_id not in self.order_manager.preassign_order_ids
        ]
        if not unassigned_orders:
            return

        candidates = unassigned_orders[:total_assignable]

        # -- Step 4: Calculate pairwise similarity --
        # Build SKU sets per order
        order_skus = {}
        for o in candidates:
            order_skus[o.order_id] = set(o.skus.keys())

        n = len(candidates)
        ids = [o.order_id for o in candidates]

        # Similarity matrix (dict of dict for flexible access)
        sim = {a: {} for a in ids}
        for i in range(n):
            for j in range(i, n):
                a, b = ids[i], ids[j]
                if i == j:
                    sim[a][b] = 1.0
                else:
                    s = self._jaccard_similarity(order_skus[a], order_skus[b])
                    sim[a][b] = s
                    sim[b][a] = s

        # -- Step 5: Group orders into batches --
        num_batches = len(free_bins)

        if num_batches >= n:
            # Fewer (or equal) candidates than stations — each order is its own batch
            batches = [[oid] for oid in ids]
        else:
            # Total similarity score per order (sum of similarities to all others)
            total_sim = {
                oid: sum(sim[oid][other] for other in ids if other != oid)
                for oid in ids
            }
            # Pick top-N seeds (highest total similarity)
            sorted_by_sim = sorted(total_sim.items(), key=lambda x: x[1], reverse=True)
            seeds = [oid for oid, _ in sorted_by_sim[:num_batches]]

            # Initialize batches with seeds
            batches = [[seed] for seed in seeds]

            # Assign remaining orders to the closest seed
            remaining = [oid for oid in ids if oid not in seeds]
            for oid in remaining:
                best_idx = 0
                best_score = -1.0
                for bi, batch in enumerate(batches):
                    seed = batch[0]  # first element is the seed
                    score = sim[oid][seed]
                    if score > best_score:
                        best_score = score
                        best_idx = bi
                batches[best_idx].append(oid)

        # -- Step 6: Compare each batch to each station --
        station_ids = list(free_bins.keys())

        # Collect current SKU sets at each station (from active orders)
        station_skus = {}
        for sid in station_ids:
            st = self.station_manager.get_station_by_id(sid)
            skus_at_station = set()
            for order in st.orders:
                skus_at_station.update(order.skus.keys())
            station_skus[sid] = skus_at_station

        # Collect collective SKU set per batch
        batch_skus = []
        for batch in batches:
            collective = set()
            for oid in batch:
                collective.update(order_skus[oid])
            batch_skus.append(collective)

        # Greedy assignment: for each batch, pick the best unassigned station
        assignment = {}  # station_id -> list of order_ids
        assigned_stations = set()
        assigned_batches = set()

        # Build score matrix: batch_idx -> station_id -> similarity
        score_matrix = []
        for bi, bskus in enumerate(batch_skus):
            scores = {}
            for sid in station_ids:
                scores[sid] = self._jaccard_similarity(bskus, station_skus[sid])
            score_matrix.append(scores)

        # Greedy: pick the highest-scoring (batch, station) pair repeatedly
        for _ in range(min(len(batches), len(station_ids))):
            best_score = -1.0
            best_bi = -1
            best_sid = None
            for bi in range(len(batches)):
                if bi in assigned_batches:
                    continue
                for sid in station_ids:
                    if sid in assigned_stations:
                        continue
                    if score_matrix[bi][sid] > best_score:
                        best_score = score_matrix[bi][sid]
                        best_bi = bi
                        best_sid = sid
            if best_bi == -1:
                break
            assignment[best_sid] = batches[best_bi]
            assigned_stations.add(best_sid)
            assigned_batches.add(best_bi)

        # Any remaining batches that didn't get a station — their orders stay in pool
        # Any remaining stations that didn't get a batch — they stay idle this cycle

        # -- Step 7: Trim each batch to fit station's free bins --
        final_selection = {}  # station_id -> list of order_ids to assign
        for sid, order_ids in assignment.items():
            capacity = free_bins[sid]
            if len(order_ids) <= capacity:
                final_selection[sid] = order_ids
            else:
                final_selection[sid] = order_ids[:capacity]
                # rest stay in pool automatically (they're just not assigned)

        if not final_selection:
            return

        # -- Step 8: Execute the assignment --
        print(f"[AISYAHNA POA] tick={self._tick:.1f} assigning: {final_selection}")
        self.put_order_to_picking_station(final_selection)

    # ------------------------------------------------------------------ #

    def xxx(self):
        picking_stations = [s for s in self.station_manager.stations if s.station_type == 'picker']
        # Step 1: Get current empty bin per station
        empty_bins = {
            station.station_id: station.max_orders - len(station.order_ids)
            for station in picking_stations
            if (station.max_orders - len(station.order_ids)) > 0
        }

        if not empty_bins:
            return
        
        order_ids = []
        current_picker = list(empty_bins.keys())[0]
        total_order_ids = empty_bins[current_picker]
        fulfilment_fs = self.get_fulfilment_table(mode="FS")
        advanced_df = self.get_advanced_table_only()
        while self.preassign_per_station[current_picker] and empty_bins[current_picker] > 0:
            order_ids.append(self.preassign_per_station[current_picker].popleft())
            empty_bins[current_picker] -= 1
        if empty_bins[current_picker] == 0:
            # assign everything in order_ids
            self.yyy(current_picker, order_ids)
            return
        
        # exclude the preassigned
        exclude_indices = set()
        for q in self.preassign_per_station.values():
            exclude_indices.update(q)
        fulfilment_fs = fulfilment_fs[~fulfilment_fs.index.isin(exclude_indices)]

        order_candidates = fulfilment_fs[current_picker].sort_values(
            ascending=False
        )
        # print("### ORDER CANDIDATES")
        # print(order_candidates)

        next_bin_counts = (
            advanced_df[advanced_df['next_bin_avail'] == True]  # noqa: E712
            .groupby('station_id')
            .size()
            .to_dict()
        )
        # print("### NEXT BIN COUNTS")
        # print(next_bin_counts)
        next_bin_counts = {k: v for k, v in next_bin_counts.items() if k != current_picker}
        # print(f"after filter {next_bin_counts}")
        if not next_bin_counts:
            order_ids.extend(
                order_candidates.index[:empty_bins[current_picker]]
            )
            # print("### ORDER TO BE ASSIGNED")
            # print(order_ids)
            # assign everything in order_ids
            self.yyy(current_picker, order_ids)
            return
        else:
            print(f"next_bin_counts {next_bin_counts}")
            # raise AssertionError(f"there is next_bin_counts current picker {current_picker} next_bin_counts {next_bin_counts}")
            fulfilment_f3 = self.get_fulfilment_table(mode="F3")
            diverted_indices = set()
            for idx, val in order_candidates.items():
                best_picker = fulfilment_f3.loc[idx].idxmax()
                best_value = fulfilment_f3.loc[idx].max()
                # if best_picker != current_picker and best_value > val:
                #     raise AssertionError(f"current picker {current_picker} score {val} best_picker {best_picker} score {best_value}")
                if (
                        best_value > val and 
                        best_picker != current_picker and
                        best_picker in next_bin_counts and
                        next_bin_counts[best_picker] > 0
                    ):
                    # preassign
                    print("\n\n\n YESSSSSSS WE HAVE PREASSIGN!!!!! \n\n\n")
                    print("original")
                    print(order_candidates)
                    # with open('preassign_record.txt', 'a') as f:
                    #     f.write(f"[tick {self._tick}]current {current_picker} order {idx} score {val} bestpicker {best_picker} score {best_value}\n")
                    if not self.fast_train:
                        insert_pre_assign(
                            self._tick,
                            current_picker,
                            idx,
                            val,
                            best_picker,
                            best_value
                        )
                    self.preassign_per_station[best_picker].append(idx)
                    next_bin_counts[best_picker] -= 1
                    diverted_indices.add(idx)
                    # raise AssertionError
                else:
                    order_ids.append(idx)
                
                if len(order_ids) >= total_order_ids:
                    # process
                    self.yyy(current_picker, order_ids)
                    return

            # Fallback: if future-bin preassignment consumed too many candidate
            # orders, fill the current picker with remaining unassigned orders
            # instead of crashing the simulation.
            for idx, _val in order_candidates.items():
                if len(order_ids) >= total_order_ids:
                    break
                if idx in diverted_indices or idx in order_ids:
                    continue
                order_ids.append(idx)

            if order_ids:
                self.yyy(current_picker, order_ids[:total_order_ids])
            return

    def yyy(self, station_id, order_ids):
        self.put_order_to_picking_station({station_id: order_ids})
        return
    
    def update_robot_job_for_new_orders(self, job: RobotJob):
        if not self.dynamic_job_update_enabled:
            return 0
        if job is None or job.is_finished:
            return 0

        try:
            station: Station = self.station_manager.get_station_by_id(job.station_id)
        except KeyError:
            return 0
        if station is None or not station.is_picker_station():
            return 0

        pod: Pod = job.pod
        if pod is None or not any(
            details.get("current_qty", 0) > 0
            for details in pod.skus.values()
        ):
            return 0

        added_tasks = 0
        existing_order_skus = {
            (order_id, sku)
            for order_id, sku, qty in job.orders
            if qty > 0
        }
        orders: list[Order] = station.get_orders_in_station() or []
        for order in orders:
            remaining_skus = order.get_remaining_skus()
            for sku, qty in remaining_skus.items():
                if (order.order_id, sku) in existing_order_skus:
                    continue
                if sku not in pod.skus:
                    continue

                available_qty = pod.get_quantity(sku)
                quantity_to_take = min(available_qty, qty)
                if quantity_to_take <= 0:
                    continue

                order.commit_quantity(sku, quantity_to_take)
                job.add_picking_task(order.order_id, sku, quantity_to_take)
                pod.pick_sku(sku, quantity_to_take)
                self.pod_manager.reduce_sku_data(sku, quantity_to_take)
                station.reduce_sku_from_station(sku, quantity_to_take)
                existing_order_skus.add((order.order_id, sku))
                added_tasks += 1

        return added_tasks
