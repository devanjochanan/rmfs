"""
Gymnasium environment for PPS (Pick Pod Selection) using PPO.

Agent: each idle pod decides whether to stay unassigned or assign itself
       to one of the 3 picking stations.
Actions (per pod):
    0 = unassigned (do nothing)
    1 = assign to picker-1
    2 = assign to picker-2
    3 = assign to picker-3
Observation:
    Per pod:
        - SKU quantity vector over Top-K SKUs
        - Manhattan distance to each station
        - Match degree with each station's demand
    Per station (global):
        - Future-state SKU demand = current SKU demand - incoming pods' SKUs
          that will be picked at that station
Reward:
    pile_on  -  alpha * avg_order_completion_time
"""

from __future__ import annotations

import os
import sys
import math
import copy
from contextlib import contextmanager
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
NUM_STATIONS = 3           # picker-0, picker-1, picker-2
NUM_ACTIONS = NUM_STATIONS + 1   # 0=unassigned, 1..3=station
TOP_K_SKUS = 50           # dimensionality of SKU feature vector per pod
ALPHA_OCT = 0.001          # weight for order-completion-time penalty in reward
MAX_PODS_OBS = 60          # max pods we observe at once (padded if fewer)


# ---------------------------------------------------------------------------
# Silence simulation stdout during RL training (big speedup on Windows).
# Toggle via env var: set PPS_ENV_VERBOSE=1 to restore prints.
# ---------------------------------------------------------------------------
_SILENT_SIM = os.environ.get("PPS_ENV_VERBOSE", "0") != "1"
_DEVNULL = open(os.devnull, "w") if _SILENT_SIM else None


@contextmanager
def _silent_sim():
    """Redirect stdout to /dev/null while the simulation runs."""
    if not _SILENT_SIM:
        yield
        return
    saved = sys.stdout
    sys.stdout = _DEVNULL
    try:
        yield
    finally:
        sys.stdout = saved


class PPSEnv(gym.Env):
    """
    One *step* of this environment corresponds to one PPS decision round
    inside the warehouse simulation.

    The simulation runs its own tick loop internally between PPS calls.
    Each reset() re-initialises the warehouse.  Each step():
        1. Receives actions for all candidate pods.
        2. Executes the pod-station assignments inside the simulation.
        3. Advances the simulation until the next PPS decision point.
        4. Returns new observation + reward.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_episode_ticks: int = 32400,   # 9 hours * 3600 / 0.25
        use_heuristic_fallback: bool = True,
        reward_alpha: float = ALPHA_OCT,
    ):
        super().__init__()

        self.max_episode_ticks = max_episode_ticks
        self.use_heuristic_fallback = use_heuristic_fallback
        self.reward_alpha = reward_alpha

        # ---- spaces (will be refined in reset once we know pod count) ----
        self.max_pods = MAX_PODS_OBS
        # Per-pod feature: TOP_K_SKUS (sku qty) + 3 (dist to stations) + 3 (match degree)
        self.pod_feature_dim = TOP_K_SKUS + NUM_STATIONS + NUM_STATIONS
        self.observation_space = spaces.Dict({
            "pod_features": spaces.Box(
                low=0.0, high=1.0,
                shape=(self.max_pods, self.pod_feature_dim),
                dtype=np.float32,
            ),
            "station_features": spaces.Box(
                low=0.0, high=1.0,
                shape=(NUM_STATIONS, TOP_K_SKUS),
                dtype=np.float32,
            ),
            "num_candidates": spaces.Box(
                low=0, high=self.max_pods, shape=(1,), dtype=np.int32,
            ),
            "action_mask": spaces.Box(
                low=0, high=1,
                shape=(self.max_pods, NUM_ACTIONS),
                dtype=np.int8,
            ),
        })
        # Each pod picks one of 4 actions
        self.action_space = spaces.MultiDiscrete([NUM_ACTIONS] * self.max_pods)

        # Internal state
        self._warehouse: Optional[Inventory] = None
        self._sku_index: Dict[str, int] = {}  # sku_id -> index in TOP_K vector
        self._episode_orders_completed: int = 0
        self._episode_total_completion_time: float = 0.0
        self._episode_pile_on_items: int = 0
        self._episode_pile_on_visits: int = 0
        self._episode_cumulative_path_cost: float = 0.0
        self._prev_orders_completed: int = 0
        self._prev_pile_on_items: int = 0
        self._prev_pile_on_visits: int = 0
        self._prev_completion_time: float = 0.0
        self._step_count: int = 0

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

        with _silent_sim():
            # Build a fresh warehouse
            self._warehouse = self._create_warehouse()

            # Build SKU index from the pod manager
            self._build_sku_index()

            # Reset episode metrics
            self._episode_orders_completed = 0
            self._episode_total_completion_time = 0.0
            self._episode_pile_on_items = 0
            self._episode_pile_on_visits = 0
            self._episode_cumulative_path_cost = 0.0
            self._prev_orders_completed = 0
            self._prev_pile_on_items = 0
            self._prev_pile_on_visits = 0
            self._prev_completion_time = 0.0
            self._step_count = 0

            # Advance simulation until first PPS decision point
            self._advance_to_next_pps_point()

            obs = self._build_observation()
            info = self._build_info()
        return obs, info

    def step(
        self, action: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        assert self._warehouse is not None

        with _silent_sim():
            # 1. Execute RL pod assignments
            self._execute_assignments(action)

            # 2. Advance simulation to next PPS decision point
            terminated, truncated = self._advance_to_next_pps_point()

            # 3. Compute reward
            reward = self._compute_reward()

            # 4. Build observation
            obs = self._build_observation()
            info = self._build_info()

            self._step_count += 1

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Warehouse creation
    # ------------------------------------------------------------------
    def _create_warehouse(self) -> Inventory:
        """Create and initialize a fresh warehouse, same as netlogo.py setup()."""
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
        from model.tools.pre_assign import (
            initialize_pre_assign_table, clear_pre_assign_table,
        )

        # Initialize DB tables (same as netlogo.py setup())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        initialize_job_task_table(timestamp)
        initialize_order_history_table(timestamp)
        initialize_pod_location_table(timestamp)
        initialize_pod_travel_table(timestamp)
        initialize_pre_assign_table(timestamp)
        clear_job_task_table()
        clear_order_history()
        clear_pod_locations()
        clear_pod_travel()
        clear_pre_assign_table()

        # Regenerate orders each episode for randomization
        from model.order_generator import config_orders
        for f in ["generated_order.csv", "generated_database_order.csv", "generated_backlog.csv"]:
            if os.path.exists(f):
                os.remove(f)
        config_orders(
            initial_order=100,
            total_requested_item=500,
            items_orders_class_configuration={"A": 0.5, "B": 0.3, "C": 0.2},
            quantity_range=[1, 12],
            order_cycle_time=500,
            order_period_time=9,
            order_start_arrival_time=0,
            date=1,
            sim_ver=1,
            dev_mode=False,
        )
        config_orders(
            initial_order=100,
            total_requested_item=500,
            items_orders_class_configuration={"A": 0.5, "B": 0.3, "C": 0.2},
            quantity_range=[1, 12],
            order_cycle_time=300,
            order_period_time=9,
            order_start_arrival_time=0,
            date=1,
            sim_ver=2,
            dev_mode=True,
        )

        # Remove stale CSVs
        if os.path.exists("assign_order.csv"):
            os.remove("assign_order.csv")
        # Recreate pod_info.csv with headers (finish_picking_task reads it)
        import pandas as pd
        pd.DataFrame(columns=["pod_id", "item_id", "qty", "order_id", "processed_time", "task_type"])\
            .to_csv("pod_info.csv", index=False)

        # Reset class-level mutable state that persists across instances
        # (Universe, Inventory, Landscape all use class-level lists/dicts)
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

        # POA: Rika's Future-aware POA (no order batching)
        warehouse.poa_podmatch = False
        warehouse.poa_first = False
        warehouse.poa_second = True
        warehouse.poa_aisyahna = False

        # PPS: disable heuristics, let RL control pod selection
        warehouse.pps_pileon = False
        warehouse.pps_demand = False
        warehouse.pps_rl = True

        draw_layout(warehouse)
        warehouse.generateResult()

        return warehouse

    # ------------------------------------------------------------------
    # SKU index for feature encoding
    # ------------------------------------------------------------------
    def _build_sku_index(self):
        """Build mapping of top-K most common SKUs to vector indices."""
        sku_counts = {}
        for pod in self._warehouse.pod_manager.pods:
            for sku, details in pod.skus.items():
                sku_counts[sku] = sku_counts.get(sku, 0) + details["current_qty"]
        # Sort by total qty descending, take top K
        sorted_skus = sorted(sku_counts.items(), key=lambda x: x[1], reverse=True)
        self._sku_index = {
            sku: i for i, (sku, _) in enumerate(sorted_skus[:TOP_K_SKUS])
        }

    # ------------------------------------------------------------------
    # Observation building
    # ------------------------------------------------------------------
    def _get_candidate_pods(self) -> List[Pod]:
        """Return idle pods that have at least 1 SKU matching any station demand."""
        pm = self._warehouse.pod_manager
        sm = self._warehouse.station_manager

        # Gather all station demands
        station_demands = {}
        for station in sm.picking_stations:
            demand = defaultdict(int)
            for order in station.orders:
                for sku, qty in order.get_remaining_skus().items():
                    demand[sku] += qty
            station_demands[station.station_id] = demand

        # Filter idle pods with at least 1 matching SKU
        all_demand_skus = set()
        for d in station_demands.values():
            all_demand_skus.update(d.keys())

        candidates = []
        for pod in pm.pods:
            if not pm.is_idle(pod.pod_id):
                continue
            pod_skus = set(
                sku for sku, det in pod.skus.items() if det["current_qty"] > 0
            )
            if pod_skus & all_demand_skus:
                candidates.append(pod)

        return candidates[:self.max_pods]

    def _get_station_positions(self) -> List[Tuple[float, float]]:
        """Return (x,y) positions for each picking station."""
        positions = []
        for station in sorted(
            self._warehouse.station_manager.picking_stations,
            key=lambda s: s.station_id,
        ):
            positions.append((station.pos_x, station.pos_y))
        return positions

    def _get_station_demands(self) -> Dict[str, Dict[str, int]]:
        """Aggregate remaining SKU demand per station."""
        demands = {}
        for station in self._warehouse.station_manager.picking_stations:
            d = defaultdict(int)
            for order in station.orders:
                for sku, qty in order.get_remaining_skus().items():
                    d[sku] += qty
            demands[station.station_id] = dict(d)
        return demands

    def _get_station_current_demands(self) -> Dict[str, Dict[str, int]]:
        """Aggregate gross SKU demand per station (total - delivered)."""
        demands = {}
        for station in self._warehouse.station_manager.picking_stations:
            d = defaultdict(int)
            for order in station.orders:
                for sku, qty in order.get_unpicked_skus().items():
                    d[sku] += qty
            demands[station.station_id] = dict(d)
        return demands

    def _get_incoming_pod_commits(self) -> Dict[str, Dict[str, int]]:
        """SKU qty committed by incoming (not yet delivered) pods per station."""
        wh = self._warehouse
        commits: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for job in wh.job_queue:
            if job is None or job.is_finished:
                continue
            for _o_id, sku, qty in job.orders:
                commits[job.station_id][sku] += qty
        for o in wh.get_movable_objects():
            if o.object_type != "robot":
                continue
            job = o.job
            if job is None or job.is_finished:
                continue
            for _o_id, sku, qty in job.orders:
                commits[job.station_id][sku] += qty
        return commits

    def _get_future_station_demands(self) -> Dict[str, Dict[str, int]]:
        """Future-state SKU demand = current demand - incoming-pod commits."""
        current = self._get_station_current_demands()
        incoming = self._get_incoming_pod_commits()
        future: Dict[str, Dict[str, int]] = {}
        for sid, cur_d in current.items():
            inc_d = incoming.get(sid, {})
            fd: Dict[str, int] = {}
            for sku, qty in cur_d.items():
                remaining = qty - inc_d.get(sku, 0)
                if remaining > 0:
                    fd[sku] = remaining
            future[sid] = fd
        return future

    def _build_observation(self) -> Dict[str, np.ndarray]:
        candidates = self._get_candidate_pods()
        station_pos = self._get_station_positions()
        station_demands = self._get_station_demands()
        future_demands = self._get_future_station_demands()
        station_ids = sorted(station_demands.keys())

        n = len(candidates)
        features = np.zeros((self.max_pods, self.pod_feature_dim), dtype=np.float32)
        mask = np.zeros((self.max_pods, NUM_ACTIONS), dtype=np.int8)
        station_feats = np.zeros((NUM_STATIONS, TOP_K_SKUS), dtype=np.float32)

        for j, sid in enumerate(station_ids[:NUM_STATIONS]):
            fd = future_demands.get(sid, {})
            for sku, qty in fd.items():
                idx = self._sku_index.get(sku)
                if idx is None:
                    continue
                station_feats[j, idx] = min(qty / 100.0, 1.0)

        # Max distance for normalization (diagonal of grid)
        max_dist = 49.0 + 31.0  # Manhattan max

        for i, pod in enumerate(candidates):
            # 1. SKU quantity features (normalized by 100)
            for sku, det in pod.skus.items():
                if sku in self._sku_index:
                    idx = self._sku_index[sku]
                    features[i, idx] = min(det["current_qty"] / 100.0, 1.0)

            # 2. Distance to each station (normalized)
            for j, (sx, sy) in enumerate(station_pos):
                dist = abs(pod.pos_x - sx) + abs(pod.pos_y - sy)
                features[i, TOP_K_SKUS + j] = 1.0 - min(dist / max_dist, 1.0)

            # 3. Match degree with each station's demand
            for j, sid in enumerate(station_ids):
                demand = station_demands.get(sid, {})
                if not demand:
                    features[i, TOP_K_SKUS + NUM_STATIONS + j] = 0.0
                    continue
                total_demand = sum(demand.values())
                matched = 0
                for sku, req in demand.items():
                    if sku in pod.skus:
                        matched += min(pod.skus[sku]["current_qty"], req)
                features[i, TOP_K_SKUS + NUM_STATIONS + j] = (
                    min(matched / max(total_demand, 1), 1.0)
                )

            # Action mask: action 0 always valid, station actions valid if station has orders
            mask[i, 0] = 1  # unassigned always valid
            for j, sid in enumerate(station_ids):
                station = self._warehouse.station_manager.get_station_by_id(sid)
                if len(station.incoming_pod) < station.max_robots:
                    mask[i, j + 1] = 1
                else:
                    mask[i, j + 1] = 0

        # Pad remaining with action 0 only
        for i in range(n, self.max_pods):
            mask[i, 0] = 1

        return {
            "pod_features": features,
            "station_features": station_feats,
            "num_candidates": np.array([n], dtype=np.int32),
            "action_mask": mask,
        }

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------
    def _execute_assignments(self, actions: np.ndarray):
        """Execute pod-to-station assignments from RL actions."""
        candidates = self._get_candidate_pods()
        station_ids = sorted(
            s.station_id
            for s in self._warehouse.station_manager.picking_stations
        )
        n = len(candidates)

        for i in range(min(n, len(actions))):
            act = int(actions[i])
            if act == 0:
                continue  # unassigned
            if act < 1 or act > NUM_STATIONS:
                continue

            target_station_id = station_ids[act - 1]
            pod = candidates[i]
            station = self._warehouse.station_manager.get_station_by_id(
                target_station_id
            )

            # Safety: skip if pod no longer idle or station full
            if not self._warehouse.pod_manager.is_idle(pod.pod_id):
                continue
            if len(station.incoming_pod) >= station.max_robots:
                continue
            if not station.orders:
                continue

            # Build sku_to_quantity and sku_to_order_map for this station
            sku_to_quantity = defaultdict(int)
            sku_to_order_map = defaultdict(list)
            for order in station.orders:
                for sku, qty in order.get_remaining_skus().items():
                    sku_to_quantity[sku] += qty
                    sku_to_order_map[sku].append((order.order_id, qty))

            if not sku_to_quantity:
                continue

            # Check pod has at least 1 matching SKU
            has_match = any(
                sku in pod.skus and pod.skus[sku]["current_qty"] > 0
                for sku in sku_to_quantity
            )
            if not has_match:
                continue

            # Execute assignment using existing method
            job = self._warehouse.add_picking_task_after_pps(
                station, pod, sku_to_order_map, sku_to_quantity
            )
            if len(job.orders) > 0:
                self._warehouse.job_queue.append(job)
                for triplet in job.orders:
                    upsert_job_task(
                        pod_id=str(job.pod.pod_id),
                        order_id=str(triplet[0]),
                        sku=str(triplet[1]),
                        qty=str(triplet[2]),
                        assigned_station=station.station_id,
                        pod_assigned_time=self._warehouse._tick,
                        status="queue",
                    )
                # Track pile-on
                self._episode_pile_on_items += len(job.orders)
                self._episode_pile_on_visits += 1

    # ------------------------------------------------------------------
    # Simulation advancement
    # ------------------------------------------------------------------
    def _advance_to_next_pps_point(self) -> Tuple[bool, bool]:
        """
        Advance warehouse simulation ticks until the next PPS decision
        is needed (i.e., at least one station has demand and idle pods exist).
        Returns (terminated, truncated).
        """
        wh = self._warehouse
        max_idle_ticks = 200  # safety: don't spin forever

        for _ in range(max_idle_ticks):
            if wh._tick >= self.max_episode_ticks:
                return False, True  # truncated

            # Check if all orders done
            total_orders = len(wh.order_manager.orders)
            finished = total_orders - len(wh.order_manager.unfinished_orders)
            if total_orders > 0 and len(wh.order_manager.unfinished_orders) == 0:
                return True, False  # terminated (all orders done)

            # Run one tick of simulation
            wh.tick()

            # Track metrics from completed orders
            self._update_episode_metrics()

            # Track cumulative path cost (robot energy)
            self._episode_cumulative_path_cost = wh.total_energy

            # Check if PPS decision is needed
            if self._pps_decision_needed():
                return False, False

        # If we spun too long, just return
        return False, False

    def _pps_decision_needed(self) -> bool:
        """Check if any station has unfulfilled demand and idle pods exist."""
        wh = self._warehouse
        has_demand = False
        for station in wh.station_manager.picking_stations:
            for order in station.orders:
                if order.get_remaining_skus():
                    has_demand = True
                    break
            if has_demand:
                break

        if not has_demand:
            return False

        # Check if idle pods exist with matching SKUs
        candidates = self._get_candidate_pods()
        return len(candidates) > 0

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def _update_episode_metrics(self):
        """Update episode-level metrics from finished orders."""
        wh = self._warehouse
        finished_count = len(wh.order_manager.orders) - len(
            wh.order_manager.unfinished_orders
        )
        self._episode_orders_completed = finished_count

        # Sum completion times from finished orders
        total_ct = 0.0
        for order in wh.order_manager.orders:
            if order.order_complete_time > 0 and order.process_start_time > 0:
                total_ct += order.order_complete_time - order.process_start_time
        self._episode_total_completion_time = total_ct

    def _compute_reward(self) -> float:
        """
        Reward = delta_pile_on - alpha * delta_avg_order_completion_time

        Where delta = change since last step.
        """
        # Pile-on delta
        pile_on_delta = (
            self._episode_pile_on_items - self._prev_pile_on_items
        )
        pile_on_visits_delta = (
            self._episode_pile_on_visits - self._prev_pile_on_visits
        )
        if pile_on_visits_delta > 0:
            pile_on_rate = pile_on_delta / pile_on_visits_delta
        else:
            pile_on_rate = 0.0

        # Avg completion time delta
        new_completed = (
            self._episode_orders_completed - self._prev_orders_completed
        )
        ct_delta = (
            self._episode_total_completion_time - self._prev_completion_time
        )
        if new_completed > 0:
            avg_ct = ct_delta / new_completed
        else:
            avg_ct = 0.0

        # Update prev
        self._prev_orders_completed = self._episode_orders_completed
        self._prev_pile_on_items = self._episode_pile_on_items
        self._prev_pile_on_visits = self._episode_pile_on_visits
        self._prev_completion_time = self._episode_total_completion_time

        reward = pile_on_rate - self.reward_alpha * avg_ct
        return float(reward)

    def _build_info(self) -> Dict[str, Any]:
        """Build info dict with metrics for TensorBoard logging."""
        avg_ct = (
            self._episode_total_completion_time / max(self._episode_orders_completed, 1)
        )
        pile_on = (
            self._episode_pile_on_items / max(self._episode_pile_on_visits, 1)
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
