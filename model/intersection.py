from engine.netlogo_coordinate import NetLogoCoordinate


class Intersection:
    def __init__(self, intersection_coordinate: NetLogoCoordinate, use_reinforcement_learning=False):
        self.intersection_id = f"{intersection_coordinate.x}-{intersection_coordinate.y}"
        self.intersection_coordinate = intersection_coordinate
        self.approaching_path_coordinates = [(intersection_coordinate.x, intersection_coordinate.y)]
        self.allowed_direction = None
        self.vertical_robots = {}
        self.horizontal_robots = {}
        self.previous_vertical_robots = []
        self.previous_horizontal_robots = []
        self.last_changed_tick = 0
        self.use_reinforcement_learning = use_reinforcement_learning
        self.RL_model_name = None
        self.connected_intersection_ids = []
        self.total_stop_n_go_horizontal = 0
        self.total_stop_n_go_vertical = 0
        self.total_waiting_time_horizontal = 0
        self.total_waiting_time_vertical = 0
        self.total_robots_passed_horizontal = 0
        self.total_robots_passed_vertical = 0

    def duration_since_last_change(self, tick):
        return tick - self.last_changed_tick

    def add_robot(self, robot):
        if robot.pos_x == self.intersection_coordinate.x:
            self.horizontal_robots[robot.robotName()] = robot
        elif robot.pos_y == self.intersection_coordinate.y:
            self.vertical_robots[robot.robotName()] = robot

    def remove_robot(self, robot):
        if robot.robotName() in self.horizontal_robots:
            del self.horizontal_robots[robot.robotName()]
            self.previous_horizontal_robots.append(robot)
        elif robot.robotName() in self.vertical_robots:
            del self.vertical_robots[robot.robotName()]
            self.previous_vertical_robots.append(robot)

    def get_robots_by_state_horizontal(self, state):
        return [robot for robot in self.horizontal_robots.values() if robot.current_state == state]

    def get_robots_by_state_vertical(self, state):
        return [robot for robot in self.vertical_robots.values() if robot.current_state == state]

    def clear_previous_robots(self):
        self.previous_horizontal_robots.clear()
        self.previous_vertical_robots.clear()

    def track_robot_intersection_data(self, robot, direction):
        waiting_time = robot.current_intersection_finish_time - robot.current_intersection_start_time
        if direction == 'horizontal':
            self.total_stop_n_go_horizontal += robot.current_intersection_stop_and_go
            self.total_waiting_time_horizontal += waiting_time
            self.total_robots_passed_horizontal += 1
        elif direction == 'vertical':
            self.total_stop_n_go_vertical += robot.current_intersection_stop_and_go
            self.total_waiting_time_vertical += waiting_time
            self.total_robots_passed_vertical += 1

    def calculate_average_stop_n_go(self, direction):
        if direction == 'horizontal':
            return int(self.total_stop_n_go_horizontal / self.total_robots_passed_horizontal) \
                if self.total_robots_passed_horizontal > 0 else 0
        elif direction == 'vertical':
            return int(self.total_stop_n_go_vertical / self.total_robots_passed_vertical) \
                if self.total_robots_passed_vertical > 0 else 0

    def calculate_average_waiting_time(self, direction):
        if direction == 'horizontal':
            return int(self.total_waiting_time_horizontal / self.total_robots_passed_horizontal) \
                if self.total_robots_passed_horizontal > 0 else 0
        elif direction == 'vertical':
            return int(self.total_waiting_time_vertical / self.total_robots_passed_vertical) \
                if self.total_robots_passed_vertical > 0 else 0

    def calculate_total_waiting_time_current_robots(self, direction, tick):
        total_waiting_time = 0

        if direction == 'horizontal':
            for robot in self.horizontal_robots.values():
                if robot.current_intersection_start_time is not None:
                    total_waiting_time += tick - robot.current_intersection_start_time
        elif direction == 'vertical':
            for robot in self.vertical_robots.values():
                if robot.current_intersection_start_time is not None:
                    total_waiting_time += tick - robot.current_intersection_start_time

        return total_waiting_time

    def reset_totals(self):
        self.total_stop_n_go_horizontal = 0
        self.total_stop_n_go_vertical = 0
        self.total_waiting_time_horizontal = 0
        self.total_waiting_time_vertical = 0
        self.total_robots_passed_horizontal = 0
        self.total_robots_passed_vertical = 0

    def robot_count(self):
        return len(self.horizontal_robots) + len(self.vertical_robots)

    def update_robot(self, robot):
        robot_name = robot.robotName()
        if robot_name in self.horizontal_robots:
            self.horizontal_robots[robot_name] = robot
        elif robot_name in self.vertical_robots:
            self.vertical_robots[robot_name] = robot

    def calculate_average_waiting_time_per_direction(self, tick):
        total_waiting_time_horizontal = 0
        total_waiting_time_vertical = 0

        for robot in self.horizontal_robots.values():
            if robot.current_intersection_start_time is not None:
                total_waiting_time_horizontal += tick - robot.current_intersection_start_time

        for robot in self.vertical_robots.values():
            if robot.current_intersection_start_time is not None:
                total_waiting_time_vertical += tick - robot.current_intersection_start_time

        average_waiting_time_horizontal = total_waiting_time_horizontal / len(
            self.horizontal_robots) if self.horizontal_robots else 0
        average_waiting_time_vertical = total_waiting_time_vertical / len(
            self.vertical_robots) if self.vertical_robots else 0

        return average_waiting_time_horizontal, average_waiting_time_vertical

    def change_traffic_light(self, direction, tick):
        if self.allowed_direction == direction:
            return
        self.allowed_direction = direction
        self.last_changed_tick = tick
        print(f"Intersection: {self.intersection_id} Changed allowed direction to {direction} at tick {tick} for intersection {self.intersection_id}")

    def is_allowed_to_move(self, robot_heading):
        if self.allowed_direction is None:
            return True
        if robot_heading in (0, 180):
            return self.allowed_direction == 'Vertical'
        elif robot_heading in (90, 270):
            return self.allowed_direction == 'Horizontal'

    def get_allowed_direction_code(self):
        if self.allowed_direction is None:
            return 0
        elif self.allowed_direction == 'Vertical':
            return 1
        elif self.allowed_direction == 'Horizontal':
            return 2

    @staticmethod
    def get_allowed_direction_by_code(code):
        if code == 0:
            return None
        elif code == 1:
            return 'Vertical'
        elif code == 2:
            return 'Horizontal'

    def print_info(self):
        print("Current Allowed Direction:", self.allowed_direction)
        print("Last Updated Tick:", self.last_changed_tick)

    def add_connected_intersection_id(self, x, y):
        intersection_id = f"{x}-{y}"
        self.connected_intersection_ids.append(intersection_id)

    def should_save_robot_info(self):
        if self.intersection_coordinate.x == 15:
            return True
        else:
            return False

    def set_RL_model_name(self, model_name):
        self.RL_model_name = f"IntersectionModel_{model_name}"
