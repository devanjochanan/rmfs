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
        - SKU quantity vector over all configured SKU types
        - Manhattan distance to each station
        - Match degree with each station's demand
        - One-hot traffic zone where the pod is located
    Per station (global):
        - Future-state SKU demand = current SKU demand - incoming pods' SKUs
          that will be picked at that station
    Global traffic:
        - Robot count in each of the 5 traffic zones
Reward:
    negative assigned-order flow-time cost delta:
    completed order time sum + unfinished assigned order age sum

During training, pod SKU allocation is regenerated every episode by default.
The regenerated allocation still uses the ABC SKU class distribution from the
existing PodGenerator settings, while the physical layout stays fixed.
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
TOP_K_SKUS = 500          # dimensionality of SKU feature vector per pod
PICKED_QTY_WEIGHT = 0.0    # kept for CLI compatibility; inactive in reward
ALPHA_OCT = 1.0            # weight for order-completion-time penalty in reward
POD_VISIT_PENALTY = 0.0    # kept for CLI compatibility; inactive in reward
MAX_PODS_OBS = 60          # max pods we observe at once (padded if fewer)
SIM_TICK_TO_SECOND = 0.15  # keep PPSEnv timing aligned with NetLogo
RANDOMIZE_POD_SKUS_EACH_EPISODE = (
    os.environ.get("PPS_RANDOMIZE_PODS_EACH_EPISODE", "1").strip().lower()
    not in {"0", "false", "no", "off"}
)
FAST_TRAIN_MODE = (
    os.environ.get("RMFS_FAST_TRAIN", "0").strip().lower()
    in {"1", "true", "yes", "on"}
)
NUM_TRAFFIC_ZONES = 5
MAX_ZONE_ROBOT_COUNT = 100.0
TRAFFIC_ZONES = (
    ((5, 43), (0, 5)),
    ((5, 43), (6, 11)),
    ((5, 43), (12, 18)),
    ((5, 43), (19, 24)),
    ((5, 43), (25, 30)),
)


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


def _noop(*args, **kwargs):
    return None


def _activate_fast_training_io():
    """Disable reporting-only CSV/SQLite hooks during PPO training."""
    global upsert_job_task

    upsert_job_task = _noop

    import model.inventory as inventory_module
    import model.robot as robot_module
    import model.robot_job as robot_job_module
    import model.tools.job_task as job_task_module
    import model.tools.order_history as order_history_module
    import model.tools.pod_travel as pod_travel_module
    import model.tools.pre_assign as pre_assign_module

    inventory_module.upsert_order_history = _noop
    inventory_module.upsert_job_task = _noop
    inventory_module.update_job_task = _noop
    inventory_module.initialize_pre_assign_table = _noop
    inventory_module.clear_pre_assign_table = _noop
    inventory_module.insert_pre_assign = _noop

    robot_module.upsert_pod_travel = _noop

    job_task_module.initialize_job_task_table = _noop
    job_task_module.clear_job_task_table = _noop
    job_task_module.upsert_job_task = _noop
    job_task_module.update_job_task = _noop

    order_history_module.initialize_order_history_table = _noop
    order_history_module.clear_order_history = _noop
    order_history_module.upsert_order_history = _noop

    pod_travel_module.initialize_pod_travel_table = _noop
    pod_travel_module.clear_pod_travel = _noop
    pod_travel_module.upsert_pod_travel = _noop

    pre_assign_module.initialize_pre_assign_table = _noop
    pre_assign_module.clear_pre_assign_table = _noop
    pre_assign_module.insert_pre_assign = _noop


if FAST_TRAIN_MODE:
    _activate_fast_training_io()


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
        max_episode_ticks: int = 32400,   # 9 hours of simulation time
        use_heuristic_fallback: bool = True,
        reward_picked_qty_weight: float = PICKED_QTY_WEIGHT,
        reward_alpha: float = ALPHA_OCT,
        reward_visit_penalty: float = POD_VISIT_PENALTY,
    ):
        super().__init__()

        self.max_episode_ticks = max_episode_ticks
        self.use_heuristic_fallback = use_heuristic_fallback
        self.reward_picked_qty_weight = reward_picked_qty_weight
        self.reward_alpha = reward_alpha
        self.reward_visit_penalty = reward_visit_penalty

        # ---- spaces (will be refined in reset once we know pod count) ----
        self.max_pods = MAX_PODS_OBS
        # Per-pod feature: SKU qty + station distances + match degree + pod zone one-hot.
        self.pod_feature_dim = (
            TOP_K_SKUS + NUM_STATIONS + NUM_STATIONS + NUM_TRAFFIC_ZONES
        )
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
            "zone_robot_counts": spaces.Box(
                low=0.0, high=MAX_ZONE_ROBOT_COUNT,
                shape=(NUM_TRAFFIC_ZONES,),
                dtype=np.float32,
            ),
        })
        # Each pod picks one of 4 actions
        self.action_space = spaces.MultiDiscrete([NUM_ACTIONS] * self.max_pods)

        # Internal state
        self._warehouse: Optional[Inventory] = None
        self._sku_index: Dict[str, int] = {}  # sku_id -> stable feature index
        self._episode_orders_completed: int = 0
        self._episode_total_completion_time: float = 0.0
        self._episode_completed_orders_time_sum: float = 0.0
        self._episode_unfinished_orders_age_sum: float = 0.0
        self._episode_total_flow_time_cost: float = 0.0
        self._episode_pile_on_items: int = 0
        self._episode_pile_on_visits: int = 0
        self._episode_cumulative_path_cost: float = 0.0
        self._prev_orders_completed: int = 0
        self._prev_pile_on_items: int = 0
        self._prev_pile_on_visits: int = 0
        self._prev_completion_time: float = 0.0
        self._prev_flow_time_cost: float = 0.0
        self._last_reward_picked_qty: float = 0.0
        self._last_reward_pod_visits: float = 0.0
        self._last_reward_avg_completion_time: float = 0.0
        self._last_reward_flow_time_cost_delta: float = 0.0
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
            self._episode_completed_orders_time_sum = 0.0
            self._episode_unfinished_orders_age_sum = 0.0
            self._episode_total_flow_time_cost = 0.0
            self._episode_pile_on_items = 0
            self._episode_pile_on_visits = 0
            self._episode_cumulative_path_cost = 0.0
            self._prev_orders_completed = 0
            self._prev_pile_on_items = 0
            self._prev_pile_on_visits = 0
            self._prev_completion_time = 0.0
            self._prev_flow_time_cost = 0.0
            self._last_reward_picked_qty = 0.0
            self._last_reward_pod_visits = 0.0
            self._last_reward_avg_completion_time = 0.0
            self._last_reward_flow_time_cost_delta = 0.0
            self._step_count = 0

            # Advance simulation until first PPS decision point
            self._advance_to_next_pps_point()
            self._prev_flow_time_cost = self._episode_total_flow_time_cost

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

        # Regenerate pod-SKU allocation each training episode while keeping
        # the same ABC class distribution configured in netlogo.assign_skus_to_pods().
        if RANDOMIZE_POD_SKUS_EACH_EPISODE:
            for f in ["pods.csv", "skus_data.csv", "sorted_skus_data.csv"]:
                if os.path.exists(f):
                    os.remove(f)

        # Remove stale CSVs
        if os.path.exists("assign_order.csv"):
            os.remove("assign_order.csv")
        if not FAST_TRAIN_MODE:
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
        warehouse.fast_train = FAST_TRAIN_MODE
        warehouse.tick_to_second = SIM_TICK_TO_SECOND

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
        """Build a stable SKU-id-to-feature-index mapping.

        The previous top-K mapping sorted SKUs by current pod quantity. That is
        acceptable for a small feature sample, but with all SKU features each
        column should represent the same SKU across episodes and in NetLogo.
        """
        sku_ids = []
        for csv_file, column in (("items.csv", "item_id"), ("skus_data.csv", "item_id")):
            if not os.path.exists(csv_file):
                continue
            try:
                import pandas as pd
                df = pd.read_csv(csv_file, usecols=[column])
                sku_ids = sorted(df[column].dropna().astype(int).unique().tolist())
                if sku_ids:
                    break
            except Exception:
                sku_ids = []

        if not sku_ids:
            sku_set = set()
            for pod in self._warehouse.pod_manager.pods:
                for sku in pod.skus.keys():
                    try:
                        sku_set.add(int(sku))
                    except (TypeError, ValueError):
                        sku_set.add(sku)
            sku_ids = sorted(sku_set)

        self._sku_index = {sku: i for i, sku in enumerate(sku_ids[:TOP_K_SKUS])}

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

    @staticmethod
    def _get_traffic_zone_index(x: float, y: float) -> Optional[int]:
        """Return zero-based traffic zone index for a coordinate, or None."""
        for idx, ((min_x, max_x), (min_y, max_y)) in enumerate(TRAFFIC_ZONES):
            if min_x <= x <= max_x and min_y <= y <= max_y:
                return idx
        return None

    def _get_zone_robot_counts(self) -> np.ndarray:
        counts = np.zeros(NUM_TRAFFIC_ZONES, dtype=np.float32)
        wh = self._warehouse
        if wh is None:
            return counts

        for obj in wh.get_movable_objects():
            if getattr(obj, "object_type", None) != "robot":
                continue
            zone_idx = self._get_traffic_zone_index(obj.pos_x, obj.pos_y)
            if zone_idx is not None:
                counts[zone_idx] += 1.0
        return counts

    def _build_observation(self) -> Dict[str, np.ndarray]:
        candidates = self._get_candidate_pods()
        station_pos = self._get_station_positions()
        station_demands = self._get_station_demands()
        future_demands = self._get_future_station_demands()
        station_ids = sorted(station_demands.keys())

        n = len(candidates)
        features = np.zeros((self.max_pods, self.pod_feature_dim), dtype=np.float32)
        station_feats = np.zeros((NUM_STATIONS, TOP_K_SKUS), dtype=np.float32)
        zone_robot_counts = self._get_zone_robot_counts()

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

            # 4. Pod zone location as one-hot features.
            zone_idx = self._get_traffic_zone_index(pod.pos_x, pod.pos_y)
            if zone_idx is not None:
                zone_offset = TOP_K_SKUS + NUM_STATIONS + NUM_STATIONS
                features[i, zone_offset + zone_idx] = 1.0

        return {
            "pod_features": features,
            "station_features": station_feats,
            "num_candidates": np.array([n], dtype=np.int32),
            "zone_robot_counts": zone_robot_counts,
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
                # Track actual picked quantity, not number of order-SKU triplets.
                picked_qty = sum(qty for _, _, qty in job.orders)
                self._episode_pile_on_items += picked_qty
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

        def finish_step(terminated: bool, truncated: bool) -> Tuple[bool, bool]:
            self._update_episode_metrics()
            self._episode_cumulative_path_cost = wh.total_energy
            return terminated, truncated

        for _ in range(max_idle_ticks):
            if wh._tick >= self.max_episode_ticks:
                return finish_step(False, True)  # truncated

            # Check if all orders done
            total_orders = len(wh.order_manager.orders)
            if total_orders > 0 and len(wh.order_manager.unfinished_orders) == 0:
                return finish_step(True, False)  # terminated (all orders done)

            # Run one tick of simulation
            wh.tick()

            # Check if PPS decision is needed
            if self._pps_decision_needed():
                return finish_step(False, False)

        # If we spun too long, just return
        return finish_step(False, False)

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

        completed_orders_time_sum = 0.0
        for order in wh.order_manager.orders:
            if order.order_complete_time > 0 and order.process_start_time > 0:
                completed_orders_time_sum += (
                    order.order_complete_time - order.process_start_time
                )

        unfinished_orders_age_sum = 0.0
        current_time = int(wh._tick)
        for order in wh.order_manager.unfinished_orders:
            if order.process_start_time > 0:
                unfinished_orders_age_sum += max(
                    current_time - order.process_start_time,
                    0.0,
                )

        self._episode_completed_orders_time_sum = float(completed_orders_time_sum)
        self._episode_unfinished_orders_age_sum = float(unfinished_orders_age_sum)
        self._episode_total_completion_time = float(completed_orders_time_sum)
        self._episode_total_flow_time_cost = float(
            completed_orders_time_sum + unfinished_orders_age_sum
        )

    def _compute_reward(self) -> float:
        """
        Reward = -alpha * assigned-order flow-time cost delta.

        Flow-time cost uses the current order completion-time definition:
        order time starts when the order is assigned to a workstation/picker.
        Completed orders contribute their final processing time, while
        unfinished assigned orders contribute their current age.
        """
        picked_qty_delta = (
            self._episode_pile_on_items - self._prev_pile_on_items
        )
        pod_visits_delta = (
            self._episode_pile_on_visits - self._prev_pile_on_visits
        )
        flow_time_cost_delta = (
            self._episode_total_flow_time_cost - self._prev_flow_time_cost
        )

        self._last_reward_picked_qty = float(picked_qty_delta)
        self._last_reward_pod_visits = float(pod_visits_delta)
        self._last_reward_avg_completion_time = 0.0
        self._last_reward_flow_time_cost_delta = float(flow_time_cost_delta)

        # Update prev
        self._prev_orders_completed = self._episode_orders_completed
        self._prev_pile_on_items = self._episode_pile_on_items
        self._prev_pile_on_visits = self._episode_pile_on_visits
        self._prev_completion_time = self._episode_total_completion_time
        self._prev_flow_time_cost = self._episode_total_flow_time_cost

        reward = -self.reward_alpha * flow_time_cost_delta
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
            "completed_orders_time_sum": self._episode_completed_orders_time_sum,
            "unfinished_orders_age_sum": self._episode_unfinished_orders_age_sum,
            "total_flow_time_cost": self._episode_total_flow_time_cost,
            "pile_on_rate": pile_on,
            "pile_on_items": self._episode_pile_on_items,
            "picked_quantity": self._episode_pile_on_items,
            "pile_on_visits": self._episode_pile_on_visits,
            "reward_picked_qty_delta": self._last_reward_picked_qty,
            "reward_pod_visits_delta": self._last_reward_pod_visits,
            "reward_avg_completion_time_delta": (
                self._last_reward_avg_completion_time
            ),
            "reward_flow_time_cost_delta": self._last_reward_flow_time_cost_delta,
            "reward_picked_qty_weight": self.reward_picked_qty_weight,
            "reward_visit_penalty": self.reward_visit_penalty,
            "reward_alpha": self.reward_alpha,
            "cumulative_path_cost": self._episode_cumulative_path_cost,
            "throughput": self._episode_orders_completed,
            "tick": self._warehouse._tick if self._warehouse else 0,
            "step": self._step_count,
        }
