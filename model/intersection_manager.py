from typing import List, Optional

from engine.deep_q_network import DeepQNetwork
from engine.util import *
from model.intersection import Intersection


class IntersectionManager:
    def __init__(self, start_date_string):
        self.intersections: List[Intersection] = []
        self.coordinate_to_intersection = {}
        self.intersection_id_to_intersection = {}
        self.q_models = {}
        self.previous_state = {}
        self.previous_action = {}
        self.start_date_string = start_date_string

    def add_intersection(self, intersection: Intersection):
        self.intersections.append(intersection)
        coordinate = intersection.intersection_coordinate
        self.coordinate_to_intersection[(coordinate.x, coordinate.y)] = intersection
        self.intersection_id_to_intersection[intersection.intersection_id] = intersection

    def get_intersection_by_coordinate(self, x, y):
        return self.coordinate_to_intersection.get((x, y), None)

    def get_connected_intersections(self, current_intersection: Intersection) -> List[Intersection]:
        connected_intersections = []
        connected_intersection_ids = current_intersection.connected_intersection_ids

        for intersection_id in connected_intersection_ids:
            intersection = self.find_intersection_by_id(intersection_id)
            if intersection is not None:
                connected_intersections.append(intersection)

        return connected_intersections

    def get_state(self, current_intersection: Intersection, tick):
        state = [
            current_intersection.get_allowed_direction_code(),
            current_intersection.duration_since_last_change(tick),
            len(current_intersection.horizontal_robots),
            len(current_intersection.get_robots_by_state_horizontal("delivering_pod")),
            len(current_intersection.get_robots_by_state_horizontal("returning_pod")),
            len(current_intersection.get_robots_by_state_horizontal("taking_pod")),
            len(current_intersection.vertical_robots),
            len(current_intersection.get_robots_by_state_vertical("delivering_pod")),
            len(current_intersection.get_robots_by_state_vertical("returning_pod")),
            len(current_intersection.get_robots_by_state_vertical("taking_pod")),
        ]
        connected_intersections = self.get_connected_intersections(current_intersection)
        for intersection in connected_intersections:
            state.append(intersection.get_allowed_direction_code())
            state.append(intersection.robot_count())
        return state

    def handle_model(self, intersection: Intersection, tick):
        state = self.get_state(intersection, tick)
        self.previous_state[intersection.intersection_id] = state
        if intersection.RL_model_name not in self.q_models:
            self.q_models[intersection.RL_model_name] = self.create_new_model(intersection, state)
        model = self.q_models[intersection.RL_model_name]
        action = model.act(state)

        self.insert_robot_intersection_information_to_csv(intersection, action, tick)

        self.previous_action[intersection.intersection_id] = action
        new_direction = intersection.get_allowed_direction_by_code(action)
        self.update_allowed_direction(intersection.intersection_id, new_direction, tick)

    def insert_robot_intersection_information_to_csv(self, intersection, action, tick):
        previous_allowed_direction = intersection.allowed_direction
        new_allowed_direction = intersection.get_allowed_direction_by_code(action)
        if previous_allowed_direction == new_allowed_direction:
            return

        previous_allowed_direction = previous_allowed_direction if previous_allowed_direction is not None else "None"
        new_allowed_direction = new_allowed_direction if new_allowed_direction is not None else "None"

        header = ["intersection_id", "previous_action", "action_decided", "tick_changed", "duration_since_last_change"]
        data = [
            intersection.intersection_id,
            previous_allowed_direction,
            new_allowed_direction,
            tick,
            intersection.duration_since_last_change(tick)
        ]

        write_to_csv("allowed_direction_changes.csv", header, data, self.start_date_string)

    @staticmethod
    def create_new_model(intersection: Intersection, state):
        state_size = len(state)
        return DeepQNetwork(state_size=state_size,
                            action_size=3,
                            model_name=intersection.RL_model_name)

    def update_allowed_direction_using_q_model(self, tick):
        for intersection in self.intersections:
            connected_intersections = self.get_connected_intersections(intersection)

            if intersection.use_reinforcement_learning:
                # Check if the current intersection or any connected intersection has at least one robot
                robots_present = (intersection.robot_count() > 0 or
                                  any(connected.robot_count() > 0 for connected in connected_intersections))

                if robots_present:
                    self.handle_model(intersection, tick)

    def update_model_after_execution(self, tick):
        for intersection in self.intersections:
            if intersection.use_reinforcement_learning and intersection.RL_model_name in self.q_models:
                self.remember_and_replay(intersection, self.calculate_reward(intersection, tick),
                                         self.is_episode_done(intersection, tick), tick)

    def remember_and_replay(self, intersection: Intersection, reward, done, tick):
        model = self.q_models[intersection.RL_model_name]
        if intersection.intersection_id in self.previous_state and intersection.intersection_id in self.previous_action:
            next_state = self.get_state(intersection, tick)
            model.remember(self.previous_state[intersection.intersection_id],
                           self.previous_action[intersection.intersection_id], reward, next_state, done)
            if done:
                model.replay(64)

            self.reset_previous_state_and_action(intersection)

        if tick % 1000 == 0 and tick != 0:
            print("SAVING_MODEL")
            intersection.reset_totals()
            model.save_model(intersection.RL_model_name, tick)

    def reset_previous_state_and_action(self, intersection: Intersection):
        if intersection.RL_model_name in self.previous_state:
            del self.previous_state[intersection.intersection_id]
        if intersection.RL_model_name in self.previous_action:
            del self.previous_action[intersection.intersection_id]

    @staticmethod
    def is_episode_done(intersection: Intersection, tick):
        if intersection.robot_count() == 0:
            return True
        elif int(tick) % 1000 == 0:
            return True
        else:
            return False

    def calculate_reward(self, intersection: Intersection, tick):
        reward = 0

        for each_robot in intersection.previous_vertical_robots:
            reward += self.calculate_reward_for_passing_robot(each_robot, intersection, "vertical", 2)

        for each_robot in intersection.previous_horizontal_robots:
            reward += self.calculate_reward_for_passing_robot(each_robot, intersection, "horizontal", 1)

        intersection.clear_previous_robots()

        for each_robot in intersection.vertical_robots.values():
            reward += self.calculate_reward_for_current_robot(each_robot, intersection, "vertical", 2, tick)

        for each_robot in intersection.horizontal_robots.values():
            reward += self.calculate_reward_for_current_robot(each_robot, intersection, "horizontal", 1, tick)

        if intersection.allowed_direction is not None and intersection.robot_count() == 0:
            reward += -0.1

        return reward

    def calculate_reward_for_current_robot(self, robot, intersection, direction, multiplier, current_tick):
        robot_state_multiplier = self.get_state_multiplier(robot)

        total_waiting_time_current_robot = current_tick - robot.current_intersection_start_time
        average_waiting_time = intersection.calculate_average_waiting_time(direction)

        total_stop_n_go_current_robot = robot.current_intersection_stop_and_go
        average_stop_n_go = intersection.calculate_average_stop_n_go(direction)

        reward = 0
        if total_waiting_time_current_robot > average_waiting_time:
            wait_diff = total_waiting_time_current_robot - average_waiting_time
            reward += -0.1 * wait_diff * robot_state_multiplier * multiplier

        if total_stop_n_go_current_robot > average_stop_n_go:
            stop_go_diff = total_stop_n_go_current_robot - average_stop_n_go
            reward += -0.1 * stop_go_diff * robot_state_multiplier * multiplier

        return reward

    def calculate_reward_for_passing_robot(self, robot, intersection, direction, multiplier):
        robot_state_multiplier = self.get_state_multiplier(robot)

        previous_average_wait = intersection.calculate_average_waiting_time(direction)
        previous_average_stop_n_go = intersection.calculate_average_stop_n_go(direction)

        intersection.track_robot_intersection_data(robot, direction)

        current_average_wait = intersection.calculate_average_waiting_time(direction)
        current_average_stop_n_go = intersection.calculate_average_stop_n_go(direction)

        reward = 0
        if current_average_wait < previous_average_wait:
            wait_diff = previous_average_wait - current_average_wait
            reward += 0.3 * wait_diff * robot_state_multiplier * multiplier

        if current_average_stop_n_go < previous_average_stop_n_go:
            stop_go_diff = previous_average_stop_n_go - current_average_stop_n_go
            reward += 0.3 * stop_go_diff * robot_state_multiplier * multiplier

        # reward for passing the intersection
        reward += 1 * robot_state_multiplier * multiplier

        return reward

    @staticmethod
    def get_state_multiplier(robot):
        if robot.current_state == 'delivering_pod':
            return 1.5
        elif robot.current_state == 'returning_pod':
            return 1
        elif robot.current_state == 'taking_pod':
            return 0.75
        else:
            return 1

    def update_allowed_direction(self, intersection_id, direction, tick):
        intersection: Intersection = self.find_intersection_by_id(intersection_id)
        intersection.change_traffic_light(direction, tick)

    def find_intersection_by_path_coordinate(self, x: int, y: int) -> Optional[str]:
        for intersection in self.intersections:
            if (x, y) in intersection.approaching_path_coordinates:
                return intersection.intersection_id
        return None

    def find_intersection_by_id(self, intersection_id):
        return self.intersection_id_to_intersection.get(intersection_id, None)

    def print_info(self, x, y):
        intersection = self.coordinate_to_intersection.get((x, y))
        if intersection is not None:
            intersection.print_info()
