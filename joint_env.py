"""
Gymnasium environment for Joint POA + PPS using PPO.

Single centralized agent (warehouse controller) makes both decisions
in one step:
    action = [order_index, station_index, pod_index]

Flow per step:
    1. Check decision trigger (unassigned orders > 0, free bins > 0)
    2. Build observation (stations, orders, pods)
    3. Agent outputs action
    4. Validate action (clamp invalid indices)
    5. Execute POA (assign order to station)
    6. Execute PPS (send pod to station)
    7. Calculate reward
    8. Check if more decisions needed — if yes, return immediately
       for next step; if no, advance simulation until trigger fires again

Reward:
    pile_on_rate_delta - alpha * avg_order_completion_time_delta
"""

from __future__ import annotations

import os
import math
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from model.inventory import Inventory
from model.pod import Pod
from model.station import Station
from model.order import Order
from model.robot_job import RobotJob
from model.tools.pod_location import get_pod_location
from model.tools.job_task import upsert_job_task

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_STATIONS = 3
MAX_ORDERS_OBS = 50       # max unassigned orders in observation
MAX_PODS_OBS = 60         # max pods in observation
TOP_K_SKUS = 50           # SKU feature vector dimension
MAX_BINS_PER_STATION = 8
ALPHA_OCT = 0.001         # weight for completion-time penalty in reward


class JointEnv(gym.Env):
    """
    Joint POA + PPS environment.

    One step = one (order, station, pod) assignment.
    The agent is called repeatedly while decisions are possible,
    then the simulation advances until the next trigger.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_episode_ticks: int = 3000,
        reward_alpha: float = ALPHA_OCT,
    ):
        super().__init__()

        self.max_episode_ticks = max_episode_ticks
        self.reward_alpha = reward_alpha

        # ---- Action space: [order_index, station_index, pod_index] ----
        self.action_space = spaces.MultiDiscrete([
            MAX_ORDERS_OBS,   # which order from pool
            NUM_STATIONS,     # which station
            MAX_PODS_OBS,     # which pod
        ])

        # ---- Observation space ----
        # Station features: per station
        #   - free_bins (1)
        #   - active_order_count (1)
        #   - SKU demand vector (TOP_K_SKUS)
        #   - incoming_pod_count (1)
        station_feat_dim = 1 + 1 + TOP_K_SKUS + 1  # = 53

        # Order pool features: per order slot
        #   - SKU requirement vector (TOP_K_SKUS)
        #   - total_items (1)
        #   - wait_time (1)
        order_feat_dim = TOP_K_SKUS + 1 + 1  # = 52

        # Pod features: per pod slot
        #   - SKU inventory vector (TOP_K_SKUS)
        #   - is_idle (1)
        #   - distance to each station (NUM_STATIONS)
        #   - match degree with each station (NUM_STATIONS)
        pod_feat_dim = TOP_K_SKUS + 1 + NUM_STATIONS + NUM_STATIONS  # = 57

        self.station_feat_dim = station_feat_dim
        self.order_feat_dim = order_feat_dim
        self.pod_feat_dim = pod_feat_dim

        self.observation_space = spaces.Dict({
            "station_features": spaces.Box(
                low=0.0, high=1.0,
                shape=(NUM_STATIONS, station_feat_dim),
                dtype=np.float32,
            ),
            "order_features": spaces.Box(
                low=0.0, high=1.0,
                shape=(MAX_ORDERS_OBS, order_feat_dim),
                dtype=np.float32,
            ),
            "pod_features": spaces.Box(
                low=0.0, high=1.0,
                shape=(MAX_PODS_OBS, pod_feat_dim),
                dtype=np.float32,
            ),
            "num_orders": spaces.Box(
                low=0, high=MAX_ORDERS_OBS, shape=(1,), dtype=np.int32,
            ),
            "num_pods": spaces.Box(
                low=0, high=MAX_PODS_OBS, shape=(1,), dtype=np.int32,
            ),
        })

        # Internal state
        self._warehouse: Optional[Inventory] = None
        self._sku_index: Dict[str, int] = {}
        self._step_count: int = 0

        # Cached lists (rebuilt each observation)
        self._cached_unassigned: List[Order] = []
        self._cached_pods: List[Pod] = []

        # Episode metrics
        self._episode_orders_completed: int = 0
        self._episode_total_completion_time: float = 0.0
        self._episode_pile_on_items: int = 0
        self._episode_pile_on_visits: int = 0
        self._episode_cumulative_path_cost: float = 0.0

        # Previous step metrics (for delta reward)
        self._prev_orders_completed: int = 0
        self._prev_pile_on_items: int = 0
        self._prev_pile_on_visits: int = 0
        self._prev_completion_time: float = 0.0

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)

        self._warehouse = self._create_warehouse()
        self._build_sku_index()

        # Reset metrics
        self._step_count = 0
        self._episode_orders_completed = 0
        self._episode_total_completion_time = 0.0
        self._episode_pile_on_items = 0
        self._episode_pile_on_visits = 0
        self._episode_cumulative_path_cost = 0.0
        self._prev_orders_completed = 0
        self._prev_pile_on_items = 0
        self._prev_pile_on_visits = 0
        self._prev_completion_time = 0.0

        # Advance simulation until first decision point
        self._advance_to_decision_point()

        obs = self._build_observation()
        info = self._build_info()
        return obs, info

    def step(
        self, action: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        assert self._warehouse is not None

        # -- Step 4 & 5: Validate and execute --
        self._execute_action(action)

        # -- Step 7: Reward --
        reward = self._compute_reward()

        self._step_count += 1

        # -- Step 9: Check if more decisions needed --
        if self._decision_possible():
            # More decisions to make immediately — don't advance sim
            obs = self._build_observation()
            info = self._build_info()
            return obs, reward, False, False, info

        # No more immediate decisions — advance simulation
        terminated, truncated = self._advance_to_decision_point()

        obs = self._build_observation()
        info = self._build_info()
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Warehouse creation (same pattern as pps_env.py)
    # ------------------------------------------------------------------
    def _create_warehouse(self) -> Inventory:
        """Create and initialize a fresh warehouse."""
        from datetime import datetime
        from netlogo import draw_layout
        from model.tools.pod_location import (
            initialize_pod_location_table, clear_pod_locations,
        )
        from model.tools.pod_travel import (
            initialize_pod_travel_table, clear_pod_travel,
        )
        from model.tools.job_task import (
            initialize_job_task_table, clear_job_task_table,
        )
        from model.tools.order_history import (
            initialize_order_history_table, clear_order_history,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        initialize_job_task_table(timestamp)
        initialize_order_history_table(timestamp)
        initialize_pod_location_table(timestamp)
        initialize_pod_travel_table(timestamp)
        clear_job_task_table()
        clear_order_history()
        clear_pod_locations()
        clear_pod_travel()

        # Regenerate orders each episode for randomization
        from model.order_generator import config_orders
        for f in ["generated_order.csv", "generated_database_order.csv",
                   "generated_backlog.csv"]:
            if os.path.exists(f):
                os.remove(f)
        config_orders(
            initial_order=100, total_requested_item=500,
            items_orders_class_configuration={"A": 0.5, "B": 0.3, "C": 0.2},
            quantity_range=[1, 12], order_cycle_time=500,
            order_period_time=9, order_start_arrival_time=0,
            date=1, sim_ver=1, dev_mode=False,
        )
        config_orders(
            initial_order=100, total_requested_item=500,
            items_orders_class_configuration={"A": 0.5, "B": 0.3, "C": 0.2},
            quantity_range=[1, 12], order_cycle_time=300,
            order_period_time=9, order_start_arrival_time=0,
            date=1, sim_ver=2, dev_mode=True,
        )

        if os.path.exists("assign_order.csv"):
            os.remove("assign_order.csv")
        import pandas as pd
        pd.DataFrame(
            columns=["pod_id", "item_id", "qty", "order_id",
                      "processed_time", "task_type"]
        ).to_csv("pod_info.csv", index=False)

        # Reset class-level mutable state
        from engine.universe import Universe
        from engine.landscape import Landscape
        Universe._objects = []
        Universe.landscape = None
        Universe.graph = None
        Universe.graph_pod = None
        Inventory.map = []
        Inventory.landscape = None
        Inventory.graph = None
        Inventory.graph_pod = None
        Inventory.stop_and_go = 0
        Inventory.total_energy = 0
        Inventory.total_pod = 0
        Inventory.total_turning = 0
        Inventory.total_robot_idle = 0
        Inventory.movement_channel = {}
        Landscape._map = []
        Landscape._objects = {}
        Landscape.total_objects = 0

        warehouse = Inventory()

        # Joint RL controls both POA and PPS
        warehouse.poa_podmatch = False
        warehouse.poa_first = False
        warehouse.poa_second = False
        warehouse.poa_aisyahna = False
        warehouse.pps_pileon = False
        warehouse.pps_demand = False
        warehouse.pps_rl = True     # disables PPS in process_orders
        warehouse.joint_rl = True   # disables all POA in process_orders

        draw_layout(warehouse)
        warehouse.generateResult()

        return warehouse

    # ------------------------------------------------------------------
    # SKU index
    # ------------------------------------------------------------------
    def _build_sku_index(self):
        """Build mapping of top-K most common SKUs to vector indices."""
        sku_counts = {}
        for pod in self._warehouse.pod_manager.pods:
            for sku, details in pod.skus.items():
                sku_counts[sku] = sku_counts.get(sku, 0) + details["current_qty"]
        sorted_skus = sorted(sku_counts.items(), key=lambda x: x[1], reverse=True)
        self._sku_index = {
            sku: i for i, (sku, _) in enumerate(sorted_skus[:TOP_K_SKUS])
        }

    # ------------------------------------------------------------------
    # Decision check (Steps 1 & 2)
    # ------------------------------------------------------------------
    def _get_unassigned_orders(self) -> List[Order]:
        """Get orders in the pool (not assigned to any station)."""
        wh = self._warehouse
        return [
            o for o in wh.order_manager.unfinished_orders
            if o.station_id is None
            and o.order_id not in wh.order_manager.preassign_order_ids
        ]

    def _get_free_bin_count(self) -> Dict[str, int]:
        """Get free bin count per station (only stations with free bins)."""
        result = {}
        for st in self._warehouse.station_manager.picking_stations:
            free = st.max_orders - len(st.order_ids)
            if free > 0:
                result[st.station_id] = free
        return result

    def _get_idle_pods(self) -> List[Pod]:
        """Get idle pods that have at least some SKU stock."""
        pm = self._warehouse.pod_manager
        pods = []
        for pod in pm.pods:
            if not pm.is_idle(pod.pod_id):
                continue
            has_stock = any(
                det["current_qty"] > 0 for det in pod.skus.values()
            )
            if has_stock:
                pods.append(pod)
        return pods[:MAX_PODS_OBS]

    def _decision_possible(self) -> bool:
        """Check if at least 1 unassigned order AND 1 free bin AND 1 idle pod."""
        unassigned = self._get_unassigned_orders()
        if not unassigned:
            return False
        free_bins = self._get_free_bin_count()
        if not free_bins:
            return False
        idle_pods = self._get_idle_pods()
        if not idle_pods:
            return False
        return True

    # ------------------------------------------------------------------
    # Observation building (Step 3)
    # ------------------------------------------------------------------
    def _get_station_positions(self) -> List[Tuple[float, float]]:
        positions = []
        for station in sorted(
            self._warehouse.station_manager.picking_stations,
            key=lambda s: s.station_id,
        ):
            positions.append((station.pos_x, station.pos_y))
        return positions

    def _build_observation(self) -> Dict[str, np.ndarray]:
        wh = self._warehouse
        stations = sorted(
            wh.station_manager.picking_stations,
            key=lambda s: s.station_id,
        )
        station_pos = self._get_station_positions()

        # Cache lists for action execution
        self._cached_unassigned = self._get_unassigned_orders()[:MAX_ORDERS_OBS]
        self._cached_pods = self._get_idle_pods()

        n_orders = len(self._cached_unassigned)
        n_pods = len(self._cached_pods)

        # Max distance for normalization (Manhattan diagonal)
        max_dist = 80.0

        # -- Station features --
        station_features = np.zeros(
            (NUM_STATIONS, self.station_feat_dim), dtype=np.float32
        )
        station_demands = {}  # sid -> {sku: qty} for pod match degree

        for si, st in enumerate(stations):
            free = st.max_orders - len(st.order_ids)
            station_features[si, 0] = free / MAX_BINS_PER_STATION  # free_bins normalized
            station_features[si, 1] = len(st.order_ids) / MAX_BINS_PER_STATION  # active orders normalized

            # SKU demand vector
            demand = defaultdict(int)
            for order in st.orders:
                for sku, qty in order.get_remaining_skus().items():
                    demand[sku] += qty
            station_demands[st.station_id] = dict(demand)

            total_demand = sum(demand.values()) if demand else 1
            for sku, qty in demand.items():
                if sku in self._sku_index:
                    idx = self._sku_index[sku]
                    station_features[si, 2 + idx] = min(qty / max(total_demand, 1), 1.0)

            # Incoming pod count
            station_features[si, 2 + TOP_K_SKUS] = min(
                len(st.incoming_pod) / 11.0, 1.0
            )

        # -- Order pool features --
        order_features = np.zeros(
            (MAX_ORDERS_OBS, self.order_feat_dim), dtype=np.float32
        )
        for oi, order in enumerate(self._cached_unassigned):
            # SKU requirement vector
            remaining = order.get_remaining_skus()
            total_items = sum(remaining.values()) if remaining else 0
            for sku, qty in remaining.items():
                if sku in self._sku_index:
                    idx = self._sku_index[sku]
                    order_features[oi, idx] = min(qty / 12.0, 1.0)  # normalize by max qty

            # Total items needed
            order_features[oi, TOP_K_SKUS] = min(total_items / 50.0, 1.0)

            # Wait time (ticks since arrival)
            wait = max(0, wh._tick - order.order_arrival)
            order_features[oi, TOP_K_SKUS + 1] = min(wait / 3000.0, 1.0)

        # -- Pod features --
        pod_features = np.zeros(
            (MAX_PODS_OBS, self.pod_feat_dim), dtype=np.float32
        )
        for pi, pod in enumerate(self._cached_pods):
            # SKU inventory vector
            for sku, det in pod.skus.items():
                if sku in self._sku_index and det["current_qty"] > 0:
                    idx = self._sku_index[sku]
                    pod_features[pi, idx] = min(det["current_qty"] / 100.0, 1.0)

            # Is idle (always 1 since we filter for idle pods)
            pod_features[pi, TOP_K_SKUS] = 1.0

            # Distance to each station
            for si, (sx, sy) in enumerate(station_pos):
                dist = abs(pod.pos_x - sx) + abs(pod.pos_y - sy)
                pod_features[pi, TOP_K_SKUS + 1 + si] = 1.0 - min(dist / max_dist, 1.0)

            # Match degree with each station's demand
            for si, st in enumerate(stations):
                demand = station_demands.get(st.station_id, {})
                if not demand:
                    pod_features[pi, TOP_K_SKUS + 1 + NUM_STATIONS + si] = 0.0
                    continue
                total_demand = sum(demand.values())
                matched = 0
                for sku, req in demand.items():
                    if sku in pod.skus:
                        matched += min(pod.skus[sku]["current_qty"], req)
                pod_features[pi, TOP_K_SKUS + 1 + NUM_STATIONS + si] = min(
                    matched / max(total_demand, 1), 1.0
                )

        return {
            "station_features": station_features,
            "order_features": order_features,
            "pod_features": pod_features,
            "num_orders": np.array([n_orders], dtype=np.int32),
            "num_pods": np.array([n_pods], dtype=np.int32),
        }

    # ------------------------------------------------------------------
    # Action execution (Steps 4-6)
    # ------------------------------------------------------------------
    def _execute_action(self, action: np.ndarray):
        """Validate and execute one (order, station, pod) assignment."""
        wh = self._warehouse

        order_idx = int(action[0])
        station_idx = int(action[1])
        pod_idx = int(action[2])

        # --- Step 5: Validate ---

        # Validate order index
        n_orders = len(self._cached_unassigned)
        if n_orders == 0:
            return  # nothing to assign
        order_idx = order_idx % n_orders
        order = self._cached_unassigned[order_idx]

        # Validate station index
        stations = sorted(
            wh.station_manager.picking_stations,
            key=lambda s: s.station_id,
        )
        station_idx = station_idx % NUM_STATIONS
        station = stations[station_idx]

        # If station is full, reassign to station with most free bins
        if len(station.order_ids) >= station.max_orders:
            best_station = None
            best_free = 0
            for st in stations:
                free = st.max_orders - len(st.order_ids)
                if free > best_free:
                    best_free = free
                    best_station = st
            if best_station is None:
                return  # all stations full
            station = best_station

        # Validate pod index
        n_pods = len(self._cached_pods)
        if n_pods == 0:
            return  # no idle pods
        pod_idx = pod_idx % n_pods
        pod = self._cached_pods[pod_idx]

        # If pod is no longer idle, find nearest idle pod to station
        if not wh.pod_manager.is_idle(pod.pod_id):
            best_pod = None
            best_dist = float('inf')
            for p in self._cached_pods:
                if wh.pod_manager.is_idle(p.pod_id):
                    dist = abs(p.pos_x - station.pos_x) + abs(p.pos_y - station.pos_y)
                    if dist < best_dist:
                        best_dist = dist
                        best_pod = p
            if best_pod is None:
                return  # no idle pods at all
            pod = best_pod

        # --- Step 6a: Execute POA ---
        order.assign_station(station.station_id)
        station.add_order(order.order_id, order)

        # Update assign_order.csv
        import pandas as pd
        assign_df = pd.read_csv('assign_order.csv')
        assign_df.loc[
            assign_df['order_id'] == order.order_id, 'assigned_station'
        ] = station.station_id
        assign_df.loc[
            assign_df['order_id'] == order.order_id, 'status'
        ] = -1
        assign_df.to_csv('assign_order.csv', index=False)

        from model.tools.order_history import upsert_order_history
        upsert_order_history(
            order.order_id,
            assigned_station=station.station_id,
            order_assigned_time=wh._tick,
        )

        # Start processing timer if not started
        if order.process_start_time <= 0:
            order.start_processing(int(wh._tick))

        # --- Step 6b: Execute PPS ---
        # Build SKU demand map for the station
        sku_to_quantity = defaultdict(int)
        sku_to_order_map = defaultdict(list)
        for o in station.orders:
            for sku, qty in o.get_remaining_skus().items():
                sku_to_quantity[sku] += qty
                sku_to_order_map[sku].append((o.order_id, qty))

        if not sku_to_quantity:
            return

        # Check pod has at least 1 matching SKU — if not, find the best matching idle pod
        has_match = any(
            sku in pod.skus and pod.skus[sku]["current_qty"] > 0
            for sku in sku_to_quantity
        )
        if not has_match:
            # Safety net: substitute with the best matching idle pod
            best_pod = None
            best_score = 0
            for p in self._cached_pods:
                if not wh.pod_manager.is_idle(p.pod_id):
                    continue
                score = sum(
                    min(p.skus[sku]["current_qty"], req)
                    for sku, req in sku_to_quantity.items()
                    if sku in p.skus and p.skus[sku]["current_qty"] > 0
                )
                if score > best_score:
                    best_score = score
                    best_pod = p
            if best_pod is None or best_score == 0:
                return  # genuinely no pod can help
            pod = best_pod

        # Create picking job
        job = wh.add_picking_task_after_pps(
            station, pod, sku_to_order_map, sku_to_quantity
        )
        if len(job.orders) > 0:
            wh.job_queue.append(job)
            for triplet in job.orders:
                upsert_job_task(
                    pod_id=str(job.pod.pod_id),
                    order_id=str(triplet[0]),
                    sku=str(triplet[1]),
                    qty=str(triplet[2]),
                    assigned_station=station.station_id,
                    pod_assigned_time=wh._tick,
                    status="queue",
                )
            # Track pile-on
            self._episode_pile_on_items += len(job.orders)
            self._episode_pile_on_visits += 1

    # ------------------------------------------------------------------
    # Simulation advancement (Step 9 → Step 1)
    # ------------------------------------------------------------------
    def _advance_to_decision_point(self) -> Tuple[bool, bool]:
        """
        Advance simulation ticks until a decision is possible:
        unassigned order > 0 AND free bin > 0 AND idle pod > 0.
        Returns (terminated, truncated).
        """
        wh = self._warehouse
        max_idle_ticks = 500

        for _ in range(max_idle_ticks):
            if wh._tick >= self.max_episode_ticks:
                return False, True  # truncated

            # Check if all orders done
            total_orders = len(wh.order_manager.orders)
            if total_orders > 0 and len(wh.order_manager.unfinished_orders) == 0:
                return True, False  # terminated

            # Run one tick
            wh.tick()

            # Update metrics
            self._update_episode_metrics()
            self._episode_cumulative_path_cost = wh.total_energy

            # Check if decision is possible
            if self._decision_possible():
                return False, False

        # Safety: if spun too long, return anyway
        return False, False

    # ------------------------------------------------------------------
    # Metrics (Step 7 & 8)
    # ------------------------------------------------------------------
    def _update_episode_metrics(self):
        wh = self._warehouse
        finished_count = len(wh.order_manager.orders) - len(
            wh.order_manager.unfinished_orders
        )
        self._episode_orders_completed = finished_count

        total_ct = 0.0
        for order in wh.order_manager.orders:
            if order.order_complete_time > 0 and order.process_start_time > 0:
                total_ct += order.order_complete_time - order.process_start_time
        self._episode_total_completion_time = total_ct

    def _compute_reward(self) -> float:
        """reward = pile_on_rate_delta - alpha * avg_completion_time_delta"""
        # Pile-on delta
        pile_on_delta = self._episode_pile_on_items - self._prev_pile_on_items
        pile_on_visits_delta = self._episode_pile_on_visits - self._prev_pile_on_visits
        if pile_on_visits_delta > 0:
            pile_on_rate = pile_on_delta / pile_on_visits_delta
        else:
            pile_on_rate = 0.0

        # Avg completion time delta
        new_completed = self._episode_orders_completed - self._prev_orders_completed
        ct_delta = self._episode_total_completion_time - self._prev_completion_time
        if new_completed > 0:
            avg_ct = ct_delta / new_completed
        else:
            avg_ct = 0.0

        # Update previous
        self._prev_orders_completed = self._episode_orders_completed
        self._prev_pile_on_items = self._episode_pile_on_items
        self._prev_pile_on_visits = self._episode_pile_on_visits
        self._prev_completion_time = self._episode_total_completion_time

        reward = pile_on_rate - self.reward_alpha * avg_ct
        return float(reward)

    def _build_info(self) -> Dict[str, Any]:
        avg_ct = (
            self._episode_total_completion_time
            / max(self._episode_orders_completed, 1)
        )
        pile_on = (
            self._episode_pile_on_items
            / max(self._episode_pile_on_visits, 1)
        )
        return {
            "orders_completed": self._episode_orders_completed,
            "avg_order_completion_time": avg_ct,
            "pile_on_rate": pile_on,
            "pile_on_items": self._episode_pile_on_items,
            "pile_on_visits": self._episode_pile_on_visits,
            "cumulative_path_cost": self._episode_cumulative_path_cost,
            "throughput": self._episode_orders_completed,
            "tick": self._warehouse._tick if self._warehouse else 0,
            "step": self._step_count,
        }
