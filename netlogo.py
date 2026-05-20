import csv
import pickle
import os
import traceback
from collections import defaultdict
from typing import List
import random
import warnings

import networkx as nx
import pandas as pd
from pandas import DataFrame
import numpy as np

from pandas import DataFrame
from sklearn.cluster import KMeans

from engine.netlogo_coordinate import NetLogoCoordinate
from engine.object import Object
from model.intersection import Intersection
from model.inventory import Inventory
from model.order import Order
from model.order_generator import *
from model.pod import Pod
from model.pod_manager import PodManager
from model.robot import Robot
from model.station import Station
from model.layout import Layout
from model.pod_generator import PodGenerator
# DB
from model.tools.pod_location import clear_pod_locations, initialize_pod_location_table, upsert_pod_location
from model.tools.pod_travel import clear_pod_travel, initialize_pod_travel_table
from model.tools.job_task import clear_job_task_table, initialize_job_task_table, upsert_job_task
from model.tools.order_history import clear_order_history, initialize_order_history_table

from pip._internal import main as pipmain

warnings.simplefilter(action='ignore', category=FutureWarning)

ACTIVATE_NEAREST = True

PPS_RL_NUM_STATIONS = 3
PPS_RL_TOP_K_SKUS = 500
PPS_RL_MAX_PODS = 60
PPS_RL_NUM_TRAFFIC_ZONES = 5
PPS_RL_MAX_ZONE_ROBOT_COUNT = 100.0
PPS_RL_TRAFFIC_ZONES = (
    ((5, 43), (0, 5)),
    ((5, 43), (6, 11)),
    ((5, 43), (12, 18)),
    ((5, 43), (19, 24)),
    ((5, 43), (25, 30)),
)
PPS_RL_POD_FEATURE_DIM = (
    PPS_RL_TOP_K_SKUS
    + PPS_RL_NUM_STATIONS
    + PPS_RL_NUM_STATIONS
    + PPS_RL_NUM_TRAFFIC_ZONES
)
PPS_RL_MODEL_PATH = os.environ.get(
    "PPS_RL_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "saved_models", "pps_rl_best.zip"),
)

_PPS_RL_MODEL = None
_PPS_RL_LOAD_ATTEMPTED = False
_PPS_RL_ACTIVE_LOGGED = False
_PPS_MODE = os.environ.get("PPS_MODE", "ppo").strip().lower()


def _pps_rl_enabled():
    value = os.environ.get("PPS_RL_ENABLED", "1").strip().lower()
    return _PPS_MODE == "ppo" and value not in {"0", "false", "no", "off"}


def _normalize_pps_mode(mode):
    mode = str(mode).strip().lower().replace("-", "_").replace(" ", "_")
    if mode in {"ppo", "rl", "pps_rl", "ppo_pps"}:
        return "ppo"
    if mode in {"rika", "rika_pps", "heuristic", "pile_on", "pileon", "baseline"}:
        return "heuristic"
    if mode in {"demand", "demand_pps"}:
        return "demand"
    return "ppo"


def _pps_rl_model_candidates():
    paths = [PPS_RL_MODEL_PATH]
    if not PPS_RL_MODEL_PATH.endswith(".zip"):
        paths.append(PPS_RL_MODEL_PATH + ".zip")
    return paths


def _ensure_numpy_pickle_compat():
    """Allow NumPy 2.x SB3 archives to load in older NumPy 1.x environments."""
    try:
        import importlib
        import sys
        import numpy.core as numpy_core
    except ImportError:
        return

    sys.modules.setdefault("numpy._core", numpy_core)
    for submodule in (
        "multiarray",
        "umath",
        "numeric",
        "fromnumeric",
        "_multiarray_umath",
        "_methods",
        "overrides",
    ):
        try:
            module = importlib.import_module(f"numpy.core.{submodule}")
        except Exception:
            continue
        sys.modules.setdefault(f"numpy._core.{submodule}", module)


def _pps_rl_model_matches_current_observation(model):
    spaces = getattr(getattr(model, "observation_space", None), "spaces", None)
    if not spaces:
        return False

    required = {
        "pod_features",
        "station_features",
        "num_candidates",
        "zone_robot_counts",
    }
    if not required.issubset(spaces.keys()):
        return False

    return (
        spaces["pod_features"].shape == (PPS_RL_MAX_PODS, PPS_RL_POD_FEATURE_DIM)
        and spaces["station_features"].shape == (
            PPS_RL_NUM_STATIONS,
            PPS_RL_TOP_K_SKUS,
        )
        and spaces["num_candidates"].shape == (1,)
        and spaces["zone_robot_counts"].shape == (PPS_RL_NUM_TRAFFIC_ZONES,)
    )


def _load_pps_rl_model():
    global _PPS_RL_MODEL, _PPS_RL_LOAD_ATTEMPTED

    if _PPS_RL_MODEL is not None:
        return _PPS_RL_MODEL
    if _PPS_RL_LOAD_ATTEMPTED or not _pps_rl_enabled():
        return None

    _PPS_RL_LOAD_ATTEMPTED = True
    model_path = next((path for path in _pps_rl_model_candidates() if os.path.exists(path)), None)
    if model_path is None:
        print(f"[PPS_RL] Model not found at {PPS_RL_MODEL_PATH}. Using heuristic PPS.")
        return None

    try:
        from stable_baselines3 import PPO
        _ensure_numpy_pickle_compat()
        model = PPO.load(model_path, device="cpu")
        if not _pps_rl_model_matches_current_observation(model):
            print(
                "[PPS_RL] Loaded model uses the old observation shape. "
                "Retrain PPO after the traffic-zone feature update."
            )
            return None
        _PPS_RL_MODEL = model
        print(f"[PPS_RL] Loaded PPO model from {model_path}")
        return _PPS_RL_MODEL
    except Exception:
        print("[PPS_RL] Failed to load PPO model. Using heuristic PPS.")
        traceback.print_exc()
        return None


def _build_pps_rl_sku_index(universe):
    sku_ids = []
    for csv_file, column in (("items.csv", "item_id"), ("skus_data.csv", "item_id")):
        if not os.path.exists(csv_file):
            continue
        try:
            df = pd.read_csv(csv_file, usecols=[column])
            sku_ids = sorted(df[column].dropna().astype(int).unique().tolist())
            if sku_ids:
                break
        except Exception:
            sku_ids = []

    if not sku_ids:
        sku_set = set()
        for pod in universe.pod_manager.pods:
            for sku in pod.skus.keys():
                try:
                    sku_set.add(int(sku))
                except (TypeError, ValueError):
                    sku_set.add(sku)
        sku_ids = sorted(sku_set)

    return {sku: i for i, sku in enumerate(sku_ids[:PPS_RL_TOP_K_SKUS])}


def _configure_pps_rl_strategy(universe):
    """Enable PPO PPS when the model is available; otherwise keep heuristic PPS."""
    global _PPS_RL_ACTIVE_LOGGED

    if _PPS_MODE == "heuristic":
        universe.pps_rl = False
        universe.pps_pileon = True
        universe.pps_demand = False
        return False

    if _PPS_MODE == "demand":
        universe.pps_rl = False
        universe.pps_pileon = False
        universe.pps_demand = True
        return False

    if _load_pps_rl_model() is None:
        universe.pps_rl = False
        universe.pps_pileon = True
        universe.pps_demand = False
        return False

    universe.pps_pileon = False
    universe.pps_demand = False
    universe.pps_rl = True

    if not hasattr(universe, "pps_rl_sku_index"):
        universe.pps_rl_sku_index = _build_pps_rl_sku_index(universe)
    if not hasattr(universe, "pps_picked_quantity"):
        universe.pps_picked_quantity = 0
    if not hasattr(universe, "pps_pod_visits"):
        universe.pps_pod_visits = 0

    if not _PPS_RL_ACTIVE_LOGGED:
        print("[PPS_RL] NetLogo simulation is using PPO for pick-pod selection.")
        _PPS_RL_ACTIVE_LOGGED = True
    return True


def set_pps_mode(mode):
    """Switch PPS mode for NetLogo: 'ppo', 'heuristic'/'rika', or 'demand'."""
    global _PPS_MODE, _PPS_RL_LOAD_ATTEMPTED, _PPS_RL_ACTIVE_LOGGED

    _PPS_MODE = _normalize_pps_mode(mode)
    if _PPS_MODE == "ppo" and _PPS_RL_MODEL is None:
        _PPS_RL_LOAD_ATTEMPTED = False
    _PPS_RL_ACTIVE_LOGGED = False

    if os.path.exists("netlogo.state"):
        try:
            with open("netlogo.state", "rb") as file:
                universe = pickle.load(file)
            for obj in universe._objects:
                obj.setUniverse(universe)
            _configure_pps_rl_strategy(universe)
            with open("netlogo.state", "wb") as file:
                pickle.dump(universe, file)
        except Exception:
            traceback.print_exc()

    print(f"[PPS_MODE] Current PPS mode: {_PPS_MODE}")
    return _PPS_MODE


def _pps_rl_station_demands(universe, use_committed=True):
    demands = {}
    for station in universe.station_manager.picking_stations:
        station_demand = defaultdict(int)
        for order in station.orders:
            order_skus = order.get_remaining_skus() if use_committed else order.get_unpicked_skus()
            for sku, qty in order_skus.items():
                station_demand[sku] += qty
        demands[station.station_id] = dict(station_demand)
    return demands


def _pps_rl_incoming_pod_commits(universe):
    commits = defaultdict(lambda: defaultdict(int))
    for job in universe.job_queue:
        if job is None or job.is_finished:
            continue
        for _order_id, sku, qty in job.orders:
            commits[job.station_id][sku] += qty

    for obj in universe.get_movable_objects():
        if obj.object_type != "robot":
            continue
        job = obj.job
        if job is None or job.is_finished:
            continue
        for _order_id, sku, qty in job.orders:
            commits[job.station_id][sku] += qty
    return commits


def _pps_rl_future_station_demands(universe):
    current = _pps_rl_station_demands(universe, use_committed=False)
    incoming = _pps_rl_incoming_pod_commits(universe)
    future = {}
    for station_id, cur_demand in current.items():
        inc_demand = incoming.get(station_id, {})
        remaining = {}
        for sku, qty in cur_demand.items():
            qty_left = qty - inc_demand.get(sku, 0)
            if qty_left > 0:
                remaining[sku] = qty_left
        future[station_id] = remaining
    return future


def _pps_rl_candidate_pods(universe):
    station_demands = _pps_rl_station_demands(universe)
    demand_skus = set()
    for demand in station_demands.values():
        demand_skus.update(demand.keys())

    if not demand_skus:
        return []

    candidates = []
    for pod in universe.pod_manager.pods:
        if not universe.pod_manager.is_idle(pod.pod_id):
            continue
        pod_skus = {
            sku for sku, details in pod.skus.items()
            if details["current_qty"] > 0
        }
        if pod_skus & demand_skus:
            candidates.append(pod)
    return candidates[:PPS_RL_MAX_PODS]


def _pps_rl_decision_needed(universe):
    for station in universe.station_manager.picking_stations:
        for order in station.orders:
            if order.get_remaining_skus():
                return len(_pps_rl_candidate_pods(universe)) > 0
    return False


def _pps_rl_traffic_zone_index(x, y):
    for idx, ((min_x, max_x), (min_y, max_y)) in enumerate(PPS_RL_TRAFFIC_ZONES):
        if min_x <= x <= max_x and min_y <= y <= max_y:
            return idx
    return None


def _pps_rl_zone_robot_counts(universe):
    counts = np.zeros(PPS_RL_NUM_TRAFFIC_ZONES, dtype=np.float32)
    for obj in universe.get_movable_objects():
        if getattr(obj, "object_type", None) != "robot":
            continue
        zone_idx = _pps_rl_traffic_zone_index(obj.pos_x, obj.pos_y)
        if zone_idx is not None:
            counts[zone_idx] += 1.0
    return counts


def _build_pps_rl_observation(universe):
    if not hasattr(universe, "pps_rl_sku_index"):
        universe.pps_rl_sku_index = _build_pps_rl_sku_index(universe)

    sku_index = universe.pps_rl_sku_index
    candidates = _pps_rl_candidate_pods(universe)
    station_demands = _pps_rl_station_demands(universe)
    future_demands = _pps_rl_future_station_demands(universe)
    stations = sorted(universe.station_manager.picking_stations, key=lambda s: s.station_id)
    station_ids = [station.station_id for station in stations]
    station_pos = [(station.pos_x, station.pos_y) for station in stations]

    pod_features = np.zeros((PPS_RL_MAX_PODS, PPS_RL_POD_FEATURE_DIM), dtype=np.float32)
    station_features = np.zeros((PPS_RL_NUM_STATIONS, PPS_RL_TOP_K_SKUS), dtype=np.float32)
    zone_robot_counts = _pps_rl_zone_robot_counts(universe)

    for j, station_id in enumerate(station_ids[:PPS_RL_NUM_STATIONS]):
        for sku, qty in future_demands.get(station_id, {}).items():
            idx = sku_index.get(sku)
            if idx is not None:
                station_features[j, idx] = min(qty / 100.0, 1.0)

    max_dist = 49.0 + 31.0
    for i, pod in enumerate(candidates):
        for sku, details in pod.skus.items():
            idx = sku_index.get(sku)
            if idx is not None:
                pod_features[i, idx] = min(details["current_qty"] / 100.0, 1.0)

        for j, (station_x, station_y) in enumerate(station_pos[:PPS_RL_NUM_STATIONS]):
            dist = abs(pod.pos_x - station_x) + abs(pod.pos_y - station_y)
            pod_features[i, PPS_RL_TOP_K_SKUS + j] = 1.0 - min(dist / max_dist, 1.0)

        for j, station_id in enumerate(station_ids[:PPS_RL_NUM_STATIONS]):
            demand = station_demands.get(station_id, {})
            if not demand:
                continue

            matched = 0
            total_demand = sum(demand.values())
            for sku, req_qty in demand.items():
                if sku in pod.skus:
                    matched += min(pod.skus[sku]["current_qty"], req_qty)
            pod_features[i, PPS_RL_TOP_K_SKUS + PPS_RL_NUM_STATIONS + j] = (
                min(matched / max(total_demand, 1), 1.0)
            )

        zone_idx = _pps_rl_traffic_zone_index(pod.pos_x, pod.pos_y)
        if zone_idx is not None:
            zone_offset = (
                PPS_RL_TOP_K_SKUS
                + PPS_RL_NUM_STATIONS
                + PPS_RL_NUM_STATIONS
            )
            pod_features[i, zone_offset + zone_idx] = 1.0

    return {
        "pod_features": pod_features,
        "station_features": station_features,
        "num_candidates": np.array([len(candidates)], dtype=np.int32),
        "zone_robot_counts": zone_robot_counts,
    }


def _execute_pps_rl_actions(universe, actions):
    candidates = _pps_rl_candidate_pods(universe)
    station_ids = sorted(
        station.station_id
        for station in universe.station_manager.picking_stations
    )
    flat_actions = np.asarray(actions).reshape(-1)
    assignments = 0

    for i in range(min(len(candidates), len(flat_actions))):
        action = int(flat_actions[i])
        if action == 0 or action < 1 or action > PPS_RL_NUM_STATIONS:
            continue

        pod = candidates[i]
        station = universe.station_manager.get_station_by_id(station_ids[action - 1])

        if not universe.pod_manager.is_idle(pod.pod_id):
            continue
        if len(station.incoming_pod) >= station.max_robots:
            continue
        if not station.orders:
            continue

        sku_to_quantity = defaultdict(int)
        sku_to_order_map = defaultdict(list)
        for order in station.orders:
            for sku, qty in order.get_remaining_skus().items():
                sku_to_quantity[sku] += qty
                sku_to_order_map[sku].append((order.order_id, qty))

        if not sku_to_quantity:
            continue
        has_match = any(
            sku in pod.skus and pod.skus[sku]["current_qty"] > 0
            for sku in sku_to_quantity
        )
        if not has_match:
            continue

        job = universe.add_picking_task_after_pps(
            station,
            pod,
            sku_to_order_map,
            sku_to_quantity,
        )
        if len(job.orders) == 0:
            continue

        universe.job_queue.append(job)
        assignments += 1
        for order_id, sku, qty in job.orders:
            upsert_job_task(
                pod_id=str(job.pod.pod_id),
                order_id=str(order_id),
                sku=str(sku),
                qty=str(qty),
                assigned_station=station.station_id,
                pod_assigned_time=universe._tick,
                status="queue",
            )

    return assignments


def _apply_pps_rl_policy(universe):
    if not getattr(universe, "pps_rl", False):
        return 0
    if not _pps_rl_decision_needed(universe):
        return 0

    model = _load_pps_rl_model()
    if model is None:
        return 0

    observation = _build_pps_rl_observation(universe)
    actions, _state = model.predict(observation, deterministic=True)
    return _execute_pps_rl_actions(universe, actions)


def _get_throughput(universe):
    """Completed orders so far, matching the PPS training throughput metric."""
    total_orders = len(universe.order_manager.orders)
    unfinished_orders = len(universe.order_manager.unfinished_orders)
    return max(total_orders - unfinished_orders, 0)


def _get_avg_order_completion_time(universe):
    completed_times = []
    for order in universe.order_manager.orders:
        if order.order_complete_time >= 0 and order.process_start_time >= 0:
            completed_times.append(order.order_complete_time - order.process_start_time)
    if not completed_times:
        return 0
    return sum(completed_times) / len(completed_times)


def _get_pod_visits(universe):
    return getattr(universe, "pps_pod_visits", getattr(universe, "pps_rl_pod_visits", 0))


def _get_picked_quantity(universe):
    return getattr(universe, "pps_picked_quantity", getattr(universe, "pps_rl_picked_quantity", 0))


def _get_pile_on_rate(universe):
    pod_visits = _get_pod_visits(universe)
    if pod_visits <= 0:
        return 0
    return _get_picked_quantity(universe) / pod_visits


class DirectedGraph:
    key = ''

    def __init__(self):
        """Initialize an instance with a directed graph."""
        self.graph = nx.DiGraph()

    @staticmethod
    def node_valid(node):
        """Check if a node is valid based on custom logic.

        Args:
            node (str): The node in format 'x,y'.

        Returns:
            bool: True if the node is valid, False otherwise.
        """
        x, y = map(int, node.split(","))
        return x >= 2 and y >= 0

    def add_node(self, node):
        """Add a node to the graph if it's valid.

        Args:
            node (str): The node to add.
        """
        if self.node_valid(node):
            self.graph.add_node(node)

    def add_edge(self, start, end, weight):
        """Add an edge between two nodes with a weight if both nodes are valid.

        Args:
            start (str): The start node.
            end (str): The end node.
            weight (float): The weight of the edge.
        """
        if self.node_valid(start) and self.node_valid(end):
            self.graph.add_edge(start, end, weight=weight)

    @staticmethod
    def get_heading(p1: NetLogoCoordinate, p2: NetLogoCoordinate):
        if p1.x == p2.x:
            if p1.y > p2.y:
                return 180
            else:
                return 0
        elif p1.y == p2.y:
            if p1.x > p2.x:
                return 270
            else:
                return 90

    def dijkstra_modified(self, start, end, penalties, zone_boundary, avoid=None):
        """Find the shortest path between two nodes using Dijkstra's algorithm, avoiding specified nodes.

        Args:
            start (str): The start node.
            end (str): The end node.
            avoid (list, optional): Nodes to avoid in the path.

        Returns:
            list or None: The path from start to end if one exists, otherwise None.
        """
        # Create a copy of the graph so we can modify it without affecting the original
        G = self.graph.copy()

        # Increase the weight of the edges leading to and from the nodes to avoid
        if avoid:
            for node in avoid:
                if node not in G:
                    continue
                for neighbor in list(G.neighbors(node)) + list(G.predecessors(node)):
                    # Increase the weight significantly to discourage using these paths
                    if G.has_edge(neighbor, node):
                        G[neighbor][node]['weight'] += 10000
                    if G.has_edge(node, neighbor):
                        G[node][neighbor]['weight'] += 10000

        # Increase the weight of edges in every zone based on the penalty
        for index, zone in enumerate(zone_boundary):
            for row in range(zone[1][0], zone[0][0]):
                for col in range(zone[0][1], zone[1][1]):
                    coordinate_str = f"{row},{col}"
                    for neighbor in list(G.neighbors(coordinate_str)) + list(G.predecessors(coordinate_str)):
                        if G.has_edge(neighbor, coordinate_str):
                            G[neighbor][coordinate_str]['weight'] = penalties[index]
                        if G.has_edge(coordinate_str, neighbor):
                            G[coordinate_str][neighbor]['weight'] = penalties[index]

        try:
            # Use Dijkstra's algorithm to find the shortest path
            path = nx.shortest_path(G, source=start, target=end, weight='weight', method='bellman-ford')
            return path
        except nx.NetworkXNoPath:
            return None

    def dijkstra(self, start, end, avoid=None):
        """Find the shortest path between two nodes using Dijkstra's algorithm, avoiding specified nodes.

        Args:
            start (str): The start node.
            end (str): The end node.
            avoid (list, optional): Nodes to avoid in the path.

        Returns:
            list or None: The path from start to end if one exists, otherwise None.
        """
        # Create a copy of the graph so we can modify it without affecting the original
        G = self.graph.copy()

        # Increase the weight of the edges leading to and from the nodes to avoid
        if avoid:
            for node in avoid:
                if node not in G:
                    continue
                for neighbor in list(G.neighbors(node)) + list(G.predecessors(node)):
                    # Increase the weight significantly to discourage using these paths
                    if G.has_edge(neighbor, node):
                        G[neighbor][node]['weight'] += 1000
                    if G.has_edge(node, neighbor):
                        G[node][neighbor]['weight'] += 1000

        try:
            # Use Dijkstra's algorithm to find the shortest path
            path = nx.shortest_path(G, source=start, target=end, weight='weight', method='bellman-ford')
            return path
        except nx.NetworkXNoPath:
            return None


intersections: List[Intersection] = []

stations = [
    [2, 33],
    [2, 27],
    [2, 21],
    [2, 15],
    [2, 9],
    [2, 3],
]


# def initStation(universe: Inventory):
#     # Iterate over each station defined in the 'stations' list
#     # Assuming 'stations' is a list of tuples/lists where each item contains the x and y coordinates of a station
#     for s in stations:
#         # Create a new Station object
#         station = Station(1, "picker")

#         # Set the x and y positions from the station data
#         station.pos_x = s[0]
#         station.pos_y = s[1]

#         # Set the coordinates for the station using a helper function or class
#         # NetLogoCoordinate may be a function or class designed to handle coordinate transformations or representations
#         station.coordinate = NetLogoCoordinate(s[0], s[1])

#         # Add the station object to the universe's list of objects
#         # This could be for general object management within the universe
#         universe.addObject(station)

#         # Specifically add the station object to the universe's list of stations
#         # This could be for easy access to stations or station-specific management
#         universe.station_manager.add_station(station)


def initRobots(universe: Inventory):

    num_robot = 20  # Number of robots

    robots = []
    x_range = (5, 43)
    y_range = (0, 30)

    # Initialize a set to keep track of used coordinates
    used_coordinates = set()

    # Generate the robots with random unique x and y coordinates
    while len(robots) < num_robot:
        x = random.randint(x_range[0], x_range[1])
        y = random.randint(y_range[0], y_range[1])
        if (x, y) not in used_coordinates:
            robot = {
                'velocity': 0,
                'heading': 0,
                'x': x,
                'y': y
            }
            robots.append(robot)
            used_coordinates.add((x, y))

    # Iterate through each robot in the list to initialize and add to the universe
    for r in robots:
        # Create a new Robot instance
        robot = Robot(universe)

        # Set the robot's attributes based on the dictionary values
        robot.velocity = r['velocity']
        robot.heading = r['heading']
        robot.pos_x = r['x']
        robot.pos_y = r['y']

        # Optionally, set the robot's coordinates using a specific coordinate system
        robot.coordinate = NetLogoCoordinate(robot.pos_x, robot.pos_y)

        # Add the robot to the universe, which likely involves adding it to some internal list or map
        universe.addObject(robot)


def draw_layout(universe):
    # Check if generated_pod.csv exists in the current directory
    if os.path.exists('generated_pod.csv'):
        print("Generated pod already exist, delete generated_pod.csv if you want to change")
        draw_layout_from_generated_file(universe)
    else:
        layout = Layout()
        # This one to generate new configuration
        layout.generate()
        draw_layout_from_generated_file(universe)


def draw_layout_from_generated_file(universe: Inventory):
    draw_storage_from_generated_file(universe)

    # Config Orders
    assign_skus_to_pods(universe.pod_manager)
    config_orders(
        initial_order=100,
        total_requested_item=500,  # Number of SKU in warehouse
        # total_requested_item=1000,
        items_orders_class_configuration={"A": 0.5, "B": 0.3, "C": 0.2}, # data 13
        # items_orders_class_configuration={"A": 0.6, "B": 0.2, "C": 0.2}, # data 10 , 11 , 12
        # items_orders_class_configuration={"A": 0.7, "B": 0.2, "C": 0.1},  # data 1 - 8 Item class configuration in warehouse
        # items_orders_class_configuration={"A": 0.3, "B": 0.3, "C": 0.5}, # original
        quantity_range=[1, 12],  # Quantity range of number of SKU in each order
        order_cycle_time=500,  # Number of order per hour #previous data 1 - 12 use 500 
        order_period_time=9,  # the total hours
        order_start_arrival_time=0,  # Start time of order arrival
        date=1,
        sim_ver=1,
        dev_mode=False)
    # Config Backlog Orders
    config_orders(
        initial_order=100,  # Initial order in backlog
        total_requested_item=500,  # Number of SKU in warehouse
        # total_requested_item=1000,
        # items_orders_class_configuration={"A": 0.7, "B": 0.2, "C": 0.1},  # data 1 -8 # Item class configuration in warehouse
        items_orders_class_configuration={"A": 0.5, "B": 0.3, "C": 0.2}, # data 13
        # items_orders_class_configuration={"A": 0.6, "B": 0.2, "C": 0.2}, #data 10 , 11 , 12
        # items_orders_class_configuration={"A": 0.3, "B": 0.3, "C": 0.5}, # original
        quantity_range=[1, 12],  # Quantity range of number of SKU in each order
        order_cycle_time=300,  # Number of order per hour
        order_period_time=9,
        order_start_arrival_time=0,
        date=1,
        sim_ver=2,
        dev_mode=True)
    initRobots(universe)
    # Assign backlog clustering
    assign_backlog_orders(universe)

    pod = list(universe.pod_manager.coordinate_to_pods.values())[0]
    destinations = [
        [pod.pos_x, pod.pos_y, 0]
    ]


def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    return intersection / union


def compute_jaccard_similarity(data):
    similarity_dict = {}
    grouped = data.groupby('order_id')['item_id'].apply(set)
    for order_dum, items in grouped.items():
        similarities = []
        for other_order_dum, other_items in grouped.items():
            if order_dum == other_order_dum:
                similarities.append(1.0)  # similarity with itself is 1
            else:
                similarity = jaccard_similarity(items, other_items)
                similarities.append(similarity)
        similarity_dict[order_dum] = similarities
    return grouped, similarity_dict


def cluster_backlog_orders(jaccard_similarities, total_station, station_capacity_df):
    jaccard_similarities_list = [similarities for similarities in jaccard_similarities.values()]
    # print(jaccard_similarities_list)
    cluster_labels = [-1] * len(jaccard_similarities_list)
    station_remaining_capacity = station_capacity_df['capacity_left'].tolist()

    # K-Means clustering
    kmeans = KMeans(n_clusters=total_station)
    kmeans.fit(jaccard_similarities_list)

    cluster_labels1 = kmeans.labels_

    cluster_distances = []

    # calculate distances for each order
    for i, label in enumerate(cluster_labels1):
        centroid = kmeans.cluster_centers_[label]
        distance = np.linalg.norm(jaccard_similarities_list[i] - centroid)
        cluster_distances.append((i, label, distance))

    cluster_distances.sort(key=lambda x: x[2])

    # assign each backlog order to a cluster
    for order_idx, label, distance in cluster_distances:
        station_id = station_capacity_df.iloc[label]['id_station']
        if station_remaining_capacity[label] > 0:
            cluster_labels[order_idx] = station_id
            station_remaining_capacity[label] -= 1
        else:
            cluster_labels[order_idx] = None

    print("cluster label:")
    print(cluster_labels)

    return cluster_labels


def assign_cluster_labels(universe: Inventory, data_backlog_order_df, full_order, cluster_labels, station_capacity_df):
    order_dum_to_cluster = dict(zip(full_order.index, cluster_labels))
    temp = float('inf')
    new_order = None

    orders_df = pd.read_csv('generated_order.csv')

    file_path = 'assign_order.csv'
    if os.path.exists(file_path):
        assign_order_df = pd.read_csv(file_path)
        # pass
    else:
        assign_order_df = orders_df.copy()
        assign_order_df['assigned_station'] = pd.Series(dtype='object')
        assign_order_df['assigned_pod'] = pd.Series(dtype='object')
        assign_order_df['status'] = -3
        assign_order_df['order_processed'] = pd.Series(dtype='object')
        assign_order_df['order_finished'] = pd.Series(dtype='object')
        assign_order_df.to_csv('assign_order.csv', index=False)

    unique_orders = set()
    order_sku_map = {}
    new_order = None
    for index, row in data_backlog_order_df.iterrows():
        order_dum = row['order_id']
        station_id = order_dum_to_cluster[order_dum]

        if station_id is not None and order_dum not in unique_orders:
            unique_orders.add(order_dum)
            new_order = Order(order_dum, 0)
            # print("order: ", new_order.order_id)
            # print("station: ", station_id)

            assign_order_df.loc[assign_order_df['order_id'] == new_order.order_id, 'assigned_station'] = station_id
            assign_order_df.loc[assign_order_df['order_id'] == new_order.order_id, 'status'] = -1
            assign_order_df.loc[assign_order_df['order_id'] == new_order.order_id, 'order_processed'] = int(
                universe.tick_to_second)
            assign_order_df.to_csv('assign_order.csv', index=False)
            new_order.assign_station(station_id)
            station = universe.station_manager.get_station_by_id(station_id)
            universe.order_manager.add_order(new_order)
            order_sku_map[order_dum] = 0

        if order_dum in unique_orders:
            order = universe.order_manager.get_order_by_id(order_dum)
            order.add_sku(row['item_id'], row['item_quantity'])
            order_sku_map[order_dum] += 1
        if order_dum in order_sku_map:
            order = universe.order_manager.get_order_by_id(order_dum)
            expected_sku_count = data_backlog_order_df[data_backlog_order_df['order_id'] == order_dum].shape[0]
            if order_sku_map[order_dum] == expected_sku_count:
                station.add_order(order_dum, order)

    return station_capacity_df


def assign_backlog_orders(universe: Inventory):
    # open file order
    order_path = "generated_order.csv"
    data_order_df = pd.read_csv(order_path)

    # filter order_id < 0
    unassigned_backlog_order = data_order_df.loc[(data_order_df['order_id'] < 0)].sort_values(by=['order_id']).reset_index(
        drop=True)

    columns = ['id_station', 'capacity_left']
    station_id_cap_df = pd.DataFrame(columns=columns)

    for station in universe.station_manager.stations:
        id = station.station_id
        cap = station.max_orders - len(station.order_ids)

        new_row = pd.DataFrame({'id_station': [id], 'capacity_left': [cap]})
        # station_id_cap_df = station_id_cap_df.append({'id_station': id, 'capacity_left': cap}, ignore_index=True)
        station_id_cap_df = pd.concat([station_id_cap_df, new_row], ignore_index=True)
    is_picker = station_id_cap_df['id_station'].str.startswith('picker')

    station_id_cap_df = station_id_cap_df[is_picker]
    station_id_cap_df.reset_index(drop=True, inplace=True)

    if len(unassigned_backlog_order) > 0:
        total_station = len(station_id_cap_df)

        full_order, jaccard_similarities = compute_jaccard_similarity(unassigned_backlog_order)

        cluster_labels = cluster_backlog_orders(jaccard_similarities, total_station, station_id_cap_df)

        station_id_cap_df = assign_cluster_labels(universe, unassigned_backlog_order, full_order, cluster_labels,
                                                  station_id_cap_df)


def draw_storage_from_generated_file(universe: Inventory):
    station_picker_counter = 1
    station_replenish_counter = 1
    pods_horizontal_length = 5
    pods_vertical_length = 2
    pod_counter = 0
    graph = DirectedGraph()
    graph_pod = DirectedGraph()
    graph_pod.key = 'pod'
    universe.graph = graph
    universe.graph_pod = graph_pod
    data = pd.read_csv("generated_pod.csv", header=None)
    total_rows = len(data)
    total_cols = 0
    for y, row in data.iterrows():
        # Invert Y only to draw
        for x, value in row.items():
            obj = Object()
            obj.object_type = 'way-direction'
            obj_key = f"{x},{y}"
            obj.pos_x = x
            obj.pos_y = y

            obj_left_coordinate = f"{x - 1},{y}"
            obj_right_coordinate = f"{x + 1},{y}"
            obj_above_coordinate = f"{x},{y - 1}"
            obj_below_coordinate = f"{x},{y + 1}"

            obj_left_value = data.iloc[y, x - 1] if x > 0 else None
            obj_right_value = data.iloc[y, x + 1] if x < len(row) - 1 else None
            obj_above_value = data.iloc[y - 1, x] if y > 0 else None
            obj_below_value = data.iloc[y + 1, x] if y < total_rows - 1 else None

            weight = 1
            turning_weight = 5
            intersection_weight = 4
            if x <= 7:
                weight = 3

            if value == 0 or value == 1 or value == 2:
                add_all_direction_paths(graph, obj_key, weight)

                if value == 0:
                    obj.shape = 'empty-space'
                    if ACTIVATE_NEAREST:
                        universe.storage_manager.createStorage(x, y)
                elif value == 1:
                    obj = Pod(pod_counter)
                    if ACTIVATE_NEAREST:
                        storage = universe.storage_manager.createStorage(x, y)
                    pod_counter += 1
                    # obj.coordinate = NetLogoCoordinate(x, y)
                    obj.pos_x = x
                    obj.pos_y = y
                    upsert_pod_location(obj.pod_id, obj.pos_x, obj.pos_y)
                    
                    if ACTIVATE_NEAREST:
                        universe.storage_manager.addPodToStorage(obj, storage)
                    graph_pod.add_node(obj_key)
                    universe.pod_manager.add_pod(obj)
                elif value == 2:
                    obj.shape = 'square 2'

                if obj_left_value != 1:
                    graph_pod.add_edge(obj_key, obj_left_coordinate, weight=100)
                if obj_right_value != 1:
                    graph_pod.add_edge(obj_key, obj_right_coordinate, weight=100)
                if obj_above_value != 1:
                    graph_pod.add_edge(obj_key, obj_above_coordinate, weight=100)
                if obj_below_value != 1:
                    graph_pod.add_edge(obj_key, obj_below_coordinate, weight=100)
            elif value == 3:
                obj.shape = 'empty-space'

                intersection = Intersection(NetLogoCoordinate(x, y))
                approaching_path_coordinates = []

                if obj_right_value in [4, 6, 7]:
                    right_x = x + 1
                    while data.iloc[y, right_x] in [4, 6, 7]:
                        approaching_path_coordinates.append((right_x, y))
                        right_x += 1

                    if data.iloc[y, right_x] == 3:
                        intersection.add_connected_intersection_id(right_x, y)
                if obj_left_value in [5, 6, 7]:
                    left_x = x - 1
                    while data.iloc[y, left_x] in [5, 6, 7]:
                        approaching_path_coordinates.append((left_x, y))
                        left_x -= 1

                    if data.iloc[y, left_x] == 3:
                        intersection.add_connected_intersection_id(left_x, y)
                if obj_below_value == 6:
                    below_y = y + 1
                    while data.iloc[below_y, x] == 6:
                        approaching_path_coordinates.append((x, below_y))
                        below_y += 1

                    if data.iloc[below_y, x] == 3:
                        intersection.add_connected_intersection_id(x, below_y)
                if obj_above_value == 7:
                    above_y = y - 1
                    while data.iloc[above_y, x] == 7:
                        approaching_path_coordinates.append((x, above_y))
                        above_y -= 1

                    if data.iloc[above_y, x] == 3:
                        intersection.add_connected_intersection_id(x, above_y)

                for each_approaching_coordinate in approaching_path_coordinates:
                    intersection.approaching_path_coordinates.append(each_approaching_coordinate)

                if obj.pos_x == 15:
                    intersection.use_reinforcement_learning = True
                    if obj.pos_y == 0:
                        intersection.set_RL_model_name("BOTTOM")
                    elif obj.pos_y == 30:
                        intersection.set_RL_model_name("TOP")
                    else:
                        intersection.set_RL_model_name("MIDDLE")

                universe.intersection_manager.add_intersection(intersection)

                if obj_left_value == 4 or obj_right_value == 4:
                    graph.add_edge(obj_key, obj_left_coordinate, weight=intersection_weight)
                    graph_pod.add_edge(obj_key, obj_left_coordinate, weight=intersection_weight)
                elif obj_left_value == 5 or obj_right_value == 5:
                    graph.add_edge(obj_key, obj_right_coordinate, weight=intersection_weight)
                    graph_pod.add_edge(obj_key, obj_right_coordinate, weight=intersection_weight)

                if obj_above_value == 6 or obj_above_value == 6:
                    graph.add_edge(obj_key, obj_above_coordinate, weight=intersection_weight)
                    graph_pod.add_edge(obj_key, obj_above_coordinate, weight=intersection_weight)
                elif obj_below_value == 7 or obj_below_value == 7:
                    graph.add_edge(obj_key, obj_below_coordinate, weight=intersection_weight)
                    graph_pod.add_edge(obj_key, obj_below_coordinate, weight=intersection_weight)

                if obj_left_value == 6 or obj_left_value == 7:
                    graph.add_edge(obj_key, obj_left_coordinate, weight=intersection_weight)
                    graph_pod.add_edge(obj_key, obj_left_coordinate, weight=intersection_weight)
                elif obj_right_value == 6 or obj_right_value == 7:
                    graph.add_edge(obj_key, obj_right_coordinate, weight=intersection_weight)
                    graph_pod.add_edge(obj_key, obj_right_coordinate, weight=intersection_weight)
            elif value == 4:
                obj.shape = 'arrow-left'
                graph.add_edge(obj_key, obj_left_coordinate, weight=weight)
                graph_pod.add_edge(obj_key, obj_left_coordinate, weight=weight)

                graph.add_edge(obj_key, obj_above_coordinate, weight=turning_weight)
                graph_pod.add_edge(obj_key, obj_above_coordinate, weight=100)
                graph.add_edge(obj_key, obj_below_coordinate, weight=turning_weight)
                graph_pod.add_edge(obj_key, obj_below_coordinate, weight=100)
            elif value == 5:
                obj.shape = 'arrow-right'
                graph.add_edge(obj_key, obj_right_coordinate, weight=weight)
                graph_pod.add_edge(obj_key, obj_right_coordinate, weight=weight)

                graph.add_edge(obj_key, obj_above_coordinate, weight=turning_weight)
                graph_pod.add_edge(obj_key, obj_above_coordinate, weight=100)
                graph.add_edge(obj_key, obj_below_coordinate, weight=turning_weight)
                graph_pod.add_edge(obj_key, obj_below_coordinate, weight=100)
            elif value == 6:
                obj.shape = 'arrow-up'
                graph.add_edge(obj_key, obj_above_coordinate, weight=weight)
                graph_pod.add_edge(obj_key, obj_above_coordinate, weight=weight)

                graph.add_edge(obj_key, obj_left_coordinate, weight=turning_weight)
                graph_pod.add_edge(obj_key, obj_left_coordinate, weight=100)
                graph.add_edge(obj_key, obj_right_coordinate, weight=turning_weight)
                graph_pod.add_edge(obj_key, obj_right_coordinate, weight=100)
            elif value == 7:
                obj.shape = 'arrow-down'
                graph.add_edge(obj_key, obj_below_coordinate, weight=weight)
                graph_pod.add_edge(obj_key, obj_below_coordinate, weight=weight)

                graph.add_edge(obj_key, obj_left_coordinate, weight=weight)
                graph_pod.add_edge(obj_key, obj_left_coordinate, weight=100)
                graph.add_edge(obj_key, obj_right_coordinate, weight=weight)
                graph_pod.add_edge(obj_key, obj_right_coordinate, weight=100)
            elif value == 11 or value == 21:
                obj.shape = 'person-red'
            elif value == 12 or value == 23:
                graph_pod.add_edge(obj_key, obj_right_coordinate, weight=weight)
                obj.shape = 'rail'
            elif value == 13 or value == 22:
                graph_pod.add_edge(obj_key, obj_left_coordinate, weight=weight)
                obj.shape = 'rail'
            elif value == 14 or value == 24:
                if obj_left_value == 11:
                    obj = Station(station_picker_counter, "picker")
                    station_picker_counter += 1
                    obj.pos_x = x
                    obj.pos_y = y
                    obj.coordinate = NetLogoCoordinate(x, y)
                    obj.short_path = construct_station_path(data, x, y, station_type='picking')
                    obj.long_path = construct_station_path(data, x, y, station_type='picking', short_path=False)
                    universe.station_manager.add_station(obj)
                elif obj_right_value == 21:
                    obj = Station(station_replenish_counter, "replenishment")
                    station_replenish_counter += 1
                    obj.pos_x = x
                    obj.pos_y = y
                    obj.coordinate = NetLogoCoordinate(x, y)
                    obj.short_path = construct_station_path(data, x, y, station_type='replenishment')
                    obj.long_path = construct_station_path(data, x, y, station_type='replenishment', short_path=False)
                    universe.station_manager.add_station(obj)

                obj.shape = 'rail-triangle'
                if value == 14:
                    obj.heading = 270
                elif value == 24:
                    obj.heading = 90
                graph_pod.add_edge(obj_key, obj_above_coordinate, weight=weight)
            elif value == 16:
                obj.shape = 'rail-corner'
                obj.heading = 270
                graph_pod.add_edge(obj_key, obj_right_coordinate, weight=weight)
            elif value == 17:
                obj.shape = 'rail-corner'
                graph_pod.add_edge(obj_key, obj_above_coordinate, weight=weight)
            elif value == 18:
                obj.shape = 'rail-corner'
                obj.heading = 180
                graph_pod.add_edge(obj_key, obj_left_coordinate, weight=weight)
            elif value == 19:
                obj.shape = 'rail-corner'
                obj.heading = 90
                graph_pod.add_edge(obj_key, obj_above_coordinate, weight=weight)
            elif value == 26:
                obj.shape = 'rail-corner'
                obj.heading = 180
                graph_pod.add_edge(obj_key, obj_left_coordinate, weight=weight)
            elif value == 27:
                obj.shape = 'rail-corner'
                obj.heading = 90
                graph_pod.add_edge(obj_key, obj_below_coordinate, weight=weight)
            elif value == 28:
                obj.shape = 'rail-corner'
                obj.heading = 270
                graph_pod.add_edge(obj_key, obj_right_coordinate, weight=weight)
            elif value == 29:
                obj.shape = 'rail-corner'
                obj.heading = 0
                graph_pod.add_edge(obj_key, obj_above_coordinate, weight=weight)
            elif value == 99:
                obj.shape = 'empty-space'
            else:
                continue

            if obj_left_coordinate == 13:
                graph_pod.add_edge(obj_key, obj_left_coordinate, weight=weight)

            obj.pos_x = x
            obj.pos_y = y
            total_cols += 1
            universe.addObject(obj)

    universe.set_warehouse_size([total_rows, total_cols])


def construct_station_path(data: DataFrame, start_x, start_y, station_type: str, short_path=True):
    station_path: List[NetLogoCoordinate] = [NetLogoCoordinate(start_x, start_y)]

    if station_type not in ['picking', 'replenishment']:
        raise ValueError("station_type must be either 'picking' or 'replenishment'")

    x_increment = 1 if station_type == 'picking' else -1
    if not short_path:
        station_path.insert(0, NetLogoCoordinate(start_x + 1 * x_increment, start_y))
        station_path.insert(0, NetLogoCoordinate(start_x + 2 * x_increment, start_y))
        station_path.insert(0, NetLogoCoordinate(start_x + 2 * x_increment, start_y + 1))
        station_path.insert(0, NetLogoCoordinate(start_x + 1 * x_increment, start_y + 1))

    # go to bottom
    y, x = start_y + 1, start_x
    while data.iloc[y, x] in (14, 17, 24, 27):
        station_path.insert(0, NetLogoCoordinate(x, y))

        if data.iloc[y, x] in (17, 27):
            x += x_increment
            while data.iloc[y, x] in (13, 23):
                station_path.insert(0, NetLogoCoordinate(x, y))
                x += x_increment

        y += 1

    return station_path


def add_all_direction_paths(graph, obj_key, weight):
    x, y = map(int, obj_key.split(','))
    directions = {
        'left': (x - 1, y),
        'right': (x + 1, y),
        'up': (x, y + 1),
        'down': (x, y - 1)
    }

    for dir_key, (nx, ny) in directions.items():
        neighbor_key = f"{nx},{ny}"
        graph.add_edge(obj_key, neighbor_key, weight=weight)


def assign_skus_to_pods(pod_manager):
    # Check if pods.csv exists in the current directory
    if os.path.exists('pods.csv'):
        assign_skus_to_pods_from_file(pod_manager)
    else:
        # Fungsi generate pods.csv
        # PodGenerator(pod_manager).generate()
        PodGenerator(pod_types=[0], pod_num=[420], total_sku=500,
                    #   items_class_conf={"A": 0.07, "B": 0.28, "C": 0.65}, 
                      items_class_conf={"A": 0.1, "B": 0.3, "C": 0.6},
                      items_pods_inventory_levels={"A": 0.4, "B": 0.5, "C": 0.6}, #intial inventory , how much of each class's total inventory should be place in pods
                      items_warehouse_inventory_levels={"A": 0.3, "B": 0.4, "C": 0.5}, #replenishment threshold
                      items_pods_class_conf={"A": 0.7, "B": 0.1, "C": 0.2}, 
                    #   items_warehouse_inventory_levels={"A": 0.4, "B": 0.5, "C": 0.6}, #original
                    #   items_pods_class_conf={"A": 0.6, "B": 0.3, "C": 0.1}, #original 
                    #   items_pods_class_conf={"A": 0.7, "B": 0.2, "C": 0.1}, #data 1 - 8 used this config
                    #   items_pods_class_conf={"A": 0.4, "B": 0.4, "C": 0.2}, # data 10 
                    #   items_pods_class_conf={"A": 0.5, "B": 0.3, "C": 0.2}, # data 11 
                    #   items_pods_class_conf={"A": 0.7, "B": 0.2, "C": 0.1}, # data 12 
               
                      pod_manager=pod_manager,
                      dev_mode=False).generate()
        assign_skus_to_pods_from_file(pod_manager)


def assign_skus_to_pods_from_file(pod_manager: PodManager):

    with open('pods.csv', mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            pod_id = int(row['pod_id'])
            sku = int(row['item'])
            limit_qty = int(row['max_qty'])
            current_qty = int(row['qty'])
            threshold = row['item_pod_inventory_level']
            global_threshold_inv_level = row['item_warehouse_inventory_level']
            weight = float(row['item_weight'])

            # Find the pod by id
            pod: Pod = pod_manager.get_pod_by_id(pod_id)
            pod.add_sku(sku, limit_qty=limit_qty, current_qty=current_qty, threshold=threshold, weight=weight)
            pod_manager.add_sku_to_pod(sku, pod)

            # Add SKU Data of level
            pod_manager.add_sku_data(sku, current_qty, limit_qty, global_threshold_inv_level)

    csv_file = 'skus_data.csv'
    if os.path.exists(csv_file):
        os.remove(csv_file)
    skus_data = pod_manager.get_all_skus_data()

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['item_id', 'current_global_qty', 'max_global_qty', 'global_inv_level'])
        for key, value in skus_data.items():
            writer.writerow([key, value['current_global_qty'], value['max_global_qty'], value['global_inv_level']])

    pod_info = pd.DataFrame(columns=["pod_id", "item_id", "qty", "order_id", "processed_time", "task_type"])
    pod_info.to_csv("pod_info.csv", index=False)

    print(f"Data has been saved to {csv_file}")
    df = pd.read_csv(csv_file)
    df_sorted = df.sort_values(by='item_id')
    sorted_csv_file = 'sorted_skus_data.csv'
    df_sorted.to_csv(sorted_csv_file, index=False)


def setup():
    try:
        # Initiate DB
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        initialize_job_task_table(timestamp)
        initialize_order_history_table(timestamp)
        initialize_pod_location_table(timestamp)
        initialize_pod_travel_table(timestamp)

        clear_job_task_table()
        clear_order_history()
        clear_pod_locations()
        clear_pod_travel()
        # Initialize the simulation universe
        assignment_path = "assign_order.csv"
        if os.path.exists(assignment_path):
            os.remove(assignment_path)

        pod_info = "pod_info.csv"
        if os.path.exists(pod_info):
            os.remove(pod_info)
        universe = Inventory()

        # Populate the universe with objects and connections
        draw_layout(universe)

        # Set simulation parameters
        universe.tick_to_second = 0.15
        _configure_pps_rl_strategy(universe)

        # Generate initial results
        next_result = universe.generateResult()

        # Save the universe state for future ticks
        with open('netlogo.state', 'wb') as config_dictionary_file:
            pickle.dump(universe, config_dictionary_file)

        # Return only the first element (object positions) as NetLogo setup doesn't need station info
        return next_result[0]

    except Exception as e:
        # Print complete stack trace
        traceback.print_exc()
        return "An error occurred. See the details above."


def tick():
    try:
        # Load the simulation state
        with open('netlogo.state', 'rb') as file:
            universe: Inventory = pickle.load(file)

        # Update each object with the current universe context
        for _n in universe._objects:
            _n.setUniverse(universe)
        _configure_pps_rl_strategy(universe)

        # Perform a simulation tick
        next_result = universe.tick()
        _apply_pps_rl_policy(universe)
        if universe._tick > 28800:
            return IndexError

        # Save updated state
        with open('netlogo.state', 'wb') as config_dictionary_file:
            pickle.dump(universe, config_dictionary_file)

        # Return all required information for NetLogo
        # next_result[0] contains object positions
        # next_result[1] contains station orders
        return [next_result[0], universe.total_energy, len(universe.job_queue), universe.stop_and_go,
                universe.total_turning, next_result[1], _get_throughput(universe),
                _get_avg_order_completion_time(universe), _get_pod_visits(universe),
                _get_pile_on_rate(universe), _get_picked_quantity(universe)]

    except Exception as e:
        # Print complete stack trace
        traceback.print_exc()
        return "An error occurred. See the details above."
    
def console_tick():
    try:
        # Load the simulation state
        with open('netlogo.state', 'rb') as file:
            universe: Inventory = pickle.load(file)

        # Update each object with the current universe context
        for _n in universe._objects:
            _n.setUniverse(universe)
        _configure_pps_rl_strategy(universe)
        while True:
            # Perform a simulation tick
            next_result = universe.tick()
            _apply_pps_rl_policy(universe)
            if universe._tick > 28800:
                return IndexError

        # Save updated state
        with open('netlogo.state', 'wb') as config_dictionary_file:
            pickle.dump(universe, config_dictionary_file)

        # Return all required information for NetLogo
        # next_result[0] contains object positions
        # next_result[1] contains station orders
        return [next_result[0], universe.total_energy, len(universe.job_queue), universe.stop_and_go,
                universe.total_turning, next_result[1], _get_throughput(universe),
                _get_avg_order_completion_time(universe), _get_pod_visits(universe),
                _get_pile_on_rate(universe), _get_picked_quantity(universe)]

    except Exception as e:
        # Print complete stack trace
        traceback.print_exc()
        return "An error occurred. See the details above."


def setup_py():
    def install_package(package_name):
        """Install a Python package using pip."""
        pipmain(['install', package_name])

    # List of packages to install
    packages = ["networkx", "matplotlib"]

    # Install each package
    for package in packages:
        install_package(package)
