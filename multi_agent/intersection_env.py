from typing import Dict, Tuple

from gym.envs.registration import register
import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv, MultiAgentWrapper
from highway_env.road.lane import LineType, StraightLane, CircularLane, AbstractLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.controller import ControlledVehicle


class IntersectionEnv(AbstractEnv):

    ACTIONS: Dict[int, str] = {
        0: 'SLOWER',
        1: 'IDLE',
        2: 'FASTER'
    }
    ACTIONS_INDEXES = {v: k for k, v in ACTIONS.items()}

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 15,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20],
                },
                "absolute": True,
                "flatten": False,
                "observe_intentions": False
            },
            "action": {
                "type": "DiscreteMetaAction",
                "longitudinal": True,
                "lateral": False,
                "target_speeds": [0, 4.5, 9]
            },
            "duration": 13,  # [s]
            "destination": "o1",
            "controlled_vehicles": 1,
            "initial_vehicle_count": 10,
            "spawn_probability": 0.6,
            "screen_width": 600,
            "screen_height": 600,
            "centering_position": [0.5, 0.6],
            "scaling": 5.5 * 1.3,
            "collision_reward": -5,
            "high_speed_reward": 1,
            "arrived_reward": 1,
            "reward_speed_range": [7.0, 9.0],
            "normalize_reward": False,
            "offroad_terminal": False
        })
        return config

    def _reward(self, action: int) -> float:
        # Cooperative multi-agent reward
        return sum(self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles) \
               / len(self.controlled_vehicles)

    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        scaled_speed = utils.lmap(vehicle.speed, self.config["reward_speed_range"], [0, 1])
        reward = self.config["collision_reward"] * vehicle.crashed \
                 + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)

        reward = self.config["arrived_reward"] if self.has_arrived(vehicle) else reward
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward, [self.config["collision_reward"], self.config["arrived_reward"]], [0, 1])
        reward = 0 if not vehicle.on_road else reward
        return reward

    def _is_terminal(self) -> bool:
        dones = [self._agent_is_win(vehicle) for vehicle in self.controlled_vehicles]
        terminal = [self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles]
        return dones, terminal

    def _agent_is_terminal(self, vehicle: Vehicle) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return vehicle.crashed \
            or self.time >= self.config["duration"] \
            or self.has_arrived(vehicle) \
            or (self.config["offroad_terminal"] and not vehicle.on_road)

    def _agent_is_win(self, vehicle: Vehicle) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return not vehicle.crashed \
            and (self.has_arrived(vehicle) or self.time >= self.config["duration"]) \
            and not(self.config["offroad_terminal"] and not vehicle.on_road)

    def get_state(self):
        environment_state = []
        observation_features = self.config["observation"]["observation_config"]['features']
        observation_range = self.config["observation"]["observation_config"]['features_range']

        for idx in range(len(self.road.vehicles)):
          vehicle_data = self.road.vehicles[idx].to_dict()
          vehicle_state = []
          for feature_key in observation_features:
            vehicle_state.append(vehicle_data[feature_key])
          environment_state.append(vehicle_state.copy())

        for idx in range(len(self.road.vehicles), self.config["controlled_vehicles"] + self.config["initial_vehicle_count"]):
           environment_state.append([0.0] * len(observation_features))
        return environment_state



    def _info(self, obs: np.ndarray, action: int) -> dict:
        info = {
            "speed": [vehicle.speed for vehicle in self.controlled_vehicles],
            "crashed": [vehicle.crashed for vehicle in self.controlled_vehicles],
            "action": action,
        }
        info["agents_rewards"] = tuple(self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles)
        info["agents_dones"] = tuple(self._agent_is_win(vehicle) for vehicle in self.controlled_vehicles)
        info["agents_terminated"] = tuple(self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles)
        return info

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles(self.config["initial_vehicle_count"])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, done, terminal, info = super().step(action)
        self._clear_vehicles()
        self._spawn_vehicle(spawn_probability=self.config["spawn_probability"])
        return obs, reward, done, terminal, info

    def _make_road(self) -> None:
      """
      Construct a 4-way intersection with priority rules for traffic simulation.
    
      :return: None
      """
      width_lane = AbstractLane.DEFAULT_WIDTH
      radius_right_turn = width_lane + 5  # [m]
      radius_left_turn = radius_right_turn + width_lane  # [m]
      distance_outer = radius_right_turn + width_lane / 2
      length_access = 100  # Total access length [m] (sum of both sides)

      road_network = RoadNetwork()
      type_none, type_continuous, type_striped = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
      for index_corner in range(4):
        angle_corner = np.radians(90 * index_corner)
        is_horizontal = index_corner % 2 == 0
        lane_priority = 3 if is_horizontal else 1
        rotation_matrix = np.array([
            [np.cos(angle_corner), -np.sin(angle_corner)],
            [np.sin(angle_corner), np.cos(angle_corner)]
        ])
        # Incoming
        start_incoming = rotation_matrix @ np.array([width_lane / 2, length_access + distance_outer])
        end_incoming = rotation_matrix @ np.array([width_lane / 2, distance_outer])
        road_network.add_lane(
            "o" + str(index_corner), "ir" + str(index_corner),
            StraightLane(start_incoming, end_incoming, line_types=[type_striped, type_continuous], priority=lane_priority, speed_limit=10)
        )
        
        # Right turns
        center_right = rotation_matrix @ np.array([distance_outer, distance_outer])
        road_network.add_lane(
            "ir" + str(index_corner), "il" + str((index_corner - 1) % 4),
            CircularLane(center_right, radius_right_turn, angle_corner + np.radians(180), angle_corner + np.radians(270),
                         line_types=[type_none, type_continuous], priority=lane_priority, speed_limit=10)
        )
            # Left turns
        center_left = rotation_matrix @ np.array([-radius_left_turn + width_lane / 2, radius_left_turn - width_lane / 2])
        road_network.add_lane(
            "ir" + str(index_corner), "il" + str((index_corner + 1) % 4),
            CircularLane(center_left, radius_left_turn, angle_corner + np.radians(0), angle_corner + np.radians(-90),
                         clockwise=False, line_types=[type_none, type_none], priority=lane_priority - 1, speed_limit=10)
        )
                    # Straight lanes
        start_straight = rotation_matrix @ np.array([width_lane / 2, distance_outer])
        end_straight = rotation_matrix @ np.array([width_lane / 2, -distance_outer])
        road_network.add_lane(
            "ir" + str(index_corner), "il" + str((index_corner + 2) % 4),
            StraightLane(start_straight, end_straight, line_types=[type_striped, type_none], priority=lane_priority, speed_limit=10)
        )

        # Exit lanes
        start_exit = rotation_matrix @ np.flip([width_lane / 2, length_access + distance_outer], axis=0)
        end_exit = rotation_matrix @ np.flip([width_lane / 2, distance_outer], axis=0)
        road_network.add_lane(
            "il" + str((index_corner - 1) % 4), "o" + str((index_corner - 1) % 4),
            StraightLane(end_exit, start_exit, line_types=[type_none, type_continuous], priority=lane_priority, speed_limit=10)
        )


        self.road = RegulatedRoad(network=road_network, np_random=self.np_random, record_history=self.config["show_trajectories"])
        

    def _make_vehicles(self, n_vehicles: int = 10) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane

        :return: the ego-vehicle
        """
        # Configure vehicles
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 7  # Low jam distance
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -3

        # Random vehicles
        simulation_steps = 3
        for t in range(n_vehicles - 1):
            self._spawn_vehicle(np.linspace(0, 80, n_vehicles)[t])
        for _ in range(simulation_steps):
            [(self.road.act(), self.road.step(1 / self.config["simulation_frequency"])) for _ in range(self.config["simulation_frequency"])]

        # Challenger vehicle
        self._spawn_vehicle(60, spawn_probability=1, go_straight=True, position_deviation=0.1, speed_deviation=0)

        # Controlled vehicles
        self.controlled_vehicles = []
        for ego_id in range(0, self.config["controlled_vehicles"]):
            ego_lane = self.road.network.get_lane(("o{}".format(ego_id % 4), "ir{}".format(ego_id % 4), 0))
            destination = self.config["destination"] or "o" + str(self.np_random.randint(1, 4))
            ego_vehicle = self.action_type.vehicle_class(
                             self.road,
                             ego_lane.position(60 + 5 * np.random.randn(1), 0),
                             speed=ego_lane.speed_limit,
                             heading=ego_lane.heading_at(60))
            try:
                ego_vehicle.plan_route_to(destination)
                ego_vehicle.speed_index = ego_vehicle.speed_to_index(ego_lane.speed_limit)
                ego_vehicle.target_speed = ego_vehicle.index_to_speed(ego_vehicle.speed_index)
            except AttributeError:
                pass

            self.road.vehicles.append(ego_vehicle)
            self.controlled_vehicles.append(ego_vehicle)
            for v in self.road.vehicles:  # Prevent early collisions
                if v is not ego_vehicle and np.linalg.norm(v.position - ego_vehicle.position) < 20:
                    self.road.vehicles.remove(v)

    def _spawn_vehicle(self,
                       longitudinal: float = 0,
                       position_deviation: float = 1.,
                       speed_deviation: float = 1.,
                       spawn_probability: float = 0.6,
                       go_straight: bool = False) -> None:
        if np.random.rand() > spawn_probability:
            return

        route = self.np_random.choice(range(4), size=2, replace=False)
        route[1] = (route[0] + 2) % 4 if go_straight else route[1]
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = vehicle_type.make_on_lane(self.road, ("o" + str(route[0]), "ir" + str(route[0]), 0),
                                            longitudinal=longitudinal + 5 + np.random.randn() * position_deviation,
                                            speed=8 + np.random.randn() * speed_deviation)
        for v in self.road.vehicles:
            if np.linalg.norm(v.position - vehicle.position) < 15:
                return
        vehicle.plan_route_to("o" + str(route[1]))
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        return vehicle

    def _clear_vehicles(self) -> None:
        is_leaving = lambda vehicle: "il" in vehicle.lane_index[0] and "o" in vehicle.lane_index[1] \
                                     and vehicle.lane.local_coordinates(vehicle.position)[0] \
                                     >= vehicle.lane.length - 4 * vehicle.LENGTH
        self.road.vehicles = [vehicle for vehicle in self.road.vehicles if
                              vehicle in self.controlled_vehicles or not (is_leaving(vehicle) or vehicle.route is None)]

    def has_arrived(self, vehicle: Vehicle, exit_distance: float = 25) -> bool:
        return "il" in vehicle.lane_index[0] \
               and "o" in vehicle.lane_index[1] \
               and vehicle.lane.local_coordinates(vehicle.position)[0] >= exit_distance

    def _cost(self, action: int) -> float:
        """The constraint signal is the occurrence of collisions."""
        return float(self.vehicle.crashed)


class MultiAgentIntersectionEnv(IntersectionEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "action": {
                 "type": "MultiAgentAction",
                 "action_config": {
                     "type": "DiscreteMetaAction",
                     "lateral": False,
                     "longitudinal": True
                 }
            },
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics"
                }
            },
            "controlled_vehicles": 3
        })
        return config

class ContinuousIntersectionEnv(IntersectionEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 5,
                "features": ["presence", "x", "y", "vx", "vy", "long_off", "lat_off", "ang_off"],
            },
            "action": {
                "type": "ContinuousAction",
                "steering_range": [-np.pi / 3, np.pi / 3],
                "longitudinal": True,
                "lateral": True,
                "dynamical": True
            },
        })
        return config

TupleMultiAgentIntersectionEnv = MultiAgentWrapper(MultiAgentIntersectionEnv)


register(
    id='intersection-v0',
    entry_point='highway_env.envs:IntersectionEnv',
)

register(
    id='intersection-v1',
    entry_point='highway_env.envs:ContinuousIntersectionEnv',
)

register(
    id='intersection-multi-agent-v0',
    entry_point='highway_env.envs:MultiAgentIntersectionEnv',
)

register(
    id='intersection-multi-agent-v1',
    entry_point='highway_env.envs:TupleMultiAgentIntersectionEnv',
)
