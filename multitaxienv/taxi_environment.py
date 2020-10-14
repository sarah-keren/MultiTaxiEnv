# -*- coding: utf-8 -*-

# TODO-2 - Run test with an arbitrary agent
# TODO-3 - Update notebook

import sys

import gym
from contextlib import closing
from io import StringIO
from gym import utils
from gym.utils import seeding
from gym.envs.toy_text import discrete
import numpy as np
import itertools
import random
from .config import multitaxifuel_rewards_w_idle, base_available_actions, all_action_names

MAP = [
    "+---------+",
    "|X: |F: :X|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|X| :G|X: |",
    "+---------+",
]


class TaxiEnv(gym.Env):
    """
    The Taxi Problem
    from "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
    by Tom Dietterich
    Description:
    There are four designated locations in the grid world indicated by R(ed), G(reen), Y(ellow), and B(lue).
    When the episode starts, the taxi starts off at a random square and the passenger is at a random location.
    The taxi drives to the passenger's location, picks up the passenger, drives to the passenger's destination
    (another one of the four specified locations), and then drops off the passenger. Once the passenger is dropped off, the episode ends.

    Observations:
    A list (taxis, fuels, pass_start, destinations, pass_locs):
        taxis: a list of coordinates of each taxi
        fuels: a list of fuels for each taxi
        pass_start: a list of startinig coordinates for each passenger (current position or last available)
        destinations: a list of destination coordiniates for each passenger
        pass_locs: a list of locations of each passenger. -1 means delivered, 0 means not picked up, and positive number means the passenger
                    is in the corresponding taxi number

    Passenger start: coordinates of each of these
    - -1: In a taxi
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)

    Passenger location:
    - -1: delivered
    - 0: not in taxi
    - x: in taxi x (x is integer)

    Destinations: coordinates of each of these
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)

    Fuel:
     - 0 to 10: start with 10

    Actions:
    Actions are given as a list, each element referring to one taxi's action. Each taxi has 7 actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: pickup passenger
    - 5: dropoff passenger
    - 6: turn engine on
    - 7: turn engine off
    - 8: standby
    - 9: refuel fuel tank


    Rewards:
    There is a reward of -1 for each action and an additional reward of +20 for delivering the passenger.
    There is a reward of -10 for executing actions "pickup", "dropoff", and "refuel" illegally.

    Rendering:
    - blue: passenger
    - magenta: destination
    - yellow: empty taxi
    - green: full taxi
    - other letters (R, G, Y and B): locations for passengers and destinations
    Main class to be charactarized with hyper-parameters.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, num_taxis: int = 2, num_passengers: int = 2, max_fuel: list = [np.inf, np.inf],
                 map: list = MAP, taxis_capacity: list = [1, 1], collision_limit: bool = False,
                 fuel_type_list: list = ['F', 'F'], option_to_stand_by: bool = True):
        """
        TODO -  later version make number of passengers dynamic, even in runtime
        
        Args:
            num_taxis: number of taxis occupy the environment
            num_passengers: number of passengers to deliver
            max_fuel: list of the maximum fuel of each taxi
            map: coordinated map of the environment
            taxis_capacity: list of max_capacity of each taxi
            collision_limit: is the environment tries prevent (active) collisions (True) of not (ignores) (False)
            fuel_type_list: fuel type of each taxi
            option_to_stand_by: does a taxi can standby
        """
        self.desc = np.asarray(map, dtype='c')

        self.num_rows = num_rows = len(self.desc) - 2
        self.num_columns = num_columns = len(self.desc[0][1:-1:2])

        self.passengers_locations = []
        self.fuel_station1 = None
        self.fuel_station2 = None
        self.fuel_stations = []

        for i, row in enumerate(self.desc[1:-1]):
            for j, char in enumerate(row[1:-1:2]):
                loc = [i, j]
                if char == b'X':
                    self.passengers_locations.append(loc)
                elif char == b'F':
                    self.fuel_station1 = loc
                    self.fuel_stations.append(loc)
                elif char == b'G':
                    self.fuel_station2 = loc
                    self.fuel_stations.append(loc)

        self.coordinates = [[i, j] for i in range(num_rows) for j in range(num_columns)]

        self.max_fuel = max_fuel

        self.num_taxis = num_taxis
        self.taxis_capacity = taxis_capacity
        self.fuel_type_list = fuel_type_list

        self.collission_limit = collision_limit
        self.collided = np.zeros(num_taxis)
        self.option_to_standby = option_to_stand_by

        # A list to indicate wether the engine of taxi i is on (1) or off (0), all taxis start as on.
        self.engine_status_list = list(np.ones(num_taxis).astype(bool))
        self.num_passengers = num_passengers

        self.num_actions = len(self.get_available_actions_dictionary()[0])
        self.action_space = gym.spaces.MultiDiscrete([self.num_actions for _ in range(self.num_taxis)])
        self.lastaction = None

        self.seed()
        self.state = None
        self.dones = []

    def seed(self, seed=None) -> list:
        """
        Setting a seed for the random sample state generation.
        Args:
            seed: seed to use

        Returns: list[seed]

        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self) -> list:
        """
        Reset the environment's state:
            - taxis coordinates.
            - refuel all taxis
            - random get destinations.
            - random locate passengers.
            - preserve other defenitions of the environment (collision, capacity...)
            - all engines turn on.
        Args:

        Returns: The reset state.

        """
        taxis_locations = random.sample(self.coordinates, self.num_taxis)
        fuels = [self.max_fuel[i] for i in range(self.num_taxis)]
        passengers_start_location = [start for start in random.choices(self.passengers_locations, k=self.num_passengers)]
        passengers_destinations = [random.choice([x for x in self.passengers_locations if x != start]) for start in passengers_start_location]
        # Status of each passenger: deliverd, in_taxi, waiting
        passengers_status = [0 for _ in range(self.num_passengers)]
        self.state = [taxis_locations, fuels, passengers_start_location, passengers_destinations, passengers_status]

        self.lastaction = None
        # Turning all engines on
        self.engine_status_list = list(np.ones(self.num_taxis))
        return self.state

    def get_available_actions_dictionary(self) -> [list, dict]:
        """                    self.passengers_locations.append(loc)

        TODO: Later versions - maybe return an action-dictionary for each taxi individually.

        Generate a dictionary of all possible actions,
        based on the hyper-parameters passed to __init__
        Returns: dictionary of action_number : action_name and available action_indexes

        """

        action_names = all_action_names
        base_dictionary = {} # total dictionary
        for index, action in enumerate(action_names):
            base_dictionary[index] = action

        action_index_dictionary_available_list = base_available_actions

        if self.option_to_standby:
            action_index_dictionary_available_list += ['turn_engine_on', 'turn_engine_off', 'standby']

        if not self.max_fuel[0] == np.inf:
            action_index_dictionary_available_list.append('refuel')

        action_index_dictionary = dict((value, key) for key, value in base_dictionary.items())
        available_actions_indexes = [action_index_dictionary[action] for action in action_index_dictionary_available_list]
        index_action_dictionary = dict((key, value) for key, value in base_dictionary.items())

        return available_actions_indexes, index_action_dictionary

    def step(self, actions: list) -> tuple:
        """
        Taking list of actions in the environment's current state.
        actions[i] is the action of taxi i.
        Args:
            actions: list[str] - list of actions to take.

        Returns: list of next_state, reward_collected, is_done
        """
        max_row = self.num_rows - 1
        max_col = self.num_columns - 1
        rewards = []
        for taxi, action in enumerate(actions):
            if self.collided[taxi] == 1:
                continue
            taxis, fuels, pass_start, destinations, pass_loc = self.state
            taxi_loc = taxis[taxi]
            row, col = taxi_loc
            fuel = fuels[taxi]
            is_engine_on = self.engine_status_list[taxi]
            _, index_action_dictionary = self.get_available_actions_dictionary()

            reward = multitaxifuel_rewards_w_idle['step']  # default reward when there is no pickup/dropoff
            moved = False

            if not is_engine_on:
                if index_action_dictionary[action] == 'standby':  # standby while engine is off
                    reward = multitaxifuel_rewards_w_idle['standby_engine_off']
                elif index_action_dictionary[action] == 'turn_engine_on':  # turn engine on
                    reward = multitaxifuel_rewards_w_idle['turn_engine_on']
                    self.engine_status_list[taxi] = 1

            elif is_engine_on:
                col_to = col
                row_to = row
                # movement
                if index_action_dictionary[action] == 'south':  # south
                    if row != max_row:
                        moved = True
                    row_to = min(row + 1, max_row)
                elif index_action_dictionary[action] == 'north':  # north
                    if row != 0:
                        moved = True
                    row_to = max(row - 1, 0)
                if index_action_dictionary[action] == 'east' and self.desc[1 + row, 2 * col + 2] == b":":  # east
                    if col != max_col:
                        moved = True
                    col_to = min(col + 1, max_col)
                elif index_action_dictionary[action] == 'west' and self.desc[1 + row, 2 * col] == b":":  # west
                    if col != 0:
                        moved = True
                    col_to = max(col - 1, 0)

                #check for collision definitions and determine movement dest
                if not self.collission_limit: #ignore
                    col, row = col_to, row_to
                elif self.collission_limit:
                    if self.collided[taxi] == 0:
                        col, row = col_to, row_to
                        if len([i for i in range(len(taxis)) if taxis[i] == taxis[taxi]]) > 1: #there is a collision
                            if self.option_to_standby:
                                moved = False
                            else:
                                self.collided[[i for i in range(len(taxis)) if taxis[i] == taxis[taxi]]] = 1
                        else: # there wasn't a collision
                            col, row = col_to, row_to
                    else:
                        continue

                # pickup/dropoff
                elif index_action_dictionary[action] == 'pickup':  # pickup
                    successful_pickup = False
                    for i, loc in enumerate(pass_loc):
                        if loc == 0 and taxi_loc == pass_start[i] and taxi + 1 not in pass_loc:
                            pass_loc = taxi + 1
                            successful_pickup = True
                            reward = multitaxifuel_rewards_w_idle['pickup']
                            break  # Picks up first passenger, modify this if capacity increases
                    if not successful_pickup:  # passenger not at location
                        reward = multitaxifuel_rewards_w_idle['bad_pickup']
                elif index_action_dictionary[action] == 'dropoff':  # dropoff
                    successful_dropoff = False
                    for i, loc in enumerate(pass_loc):  # at destination
                        if loc == taxi + 1 and taxi_loc == destinations[i]:
                            pass_loc = -1
                            reward = multitaxifuel_rewards_w_idle['final_dropoff']
                            successful_dropoff = True
                        elif loc == taxi + 1:  # drops off passenger
                            self.passengers_locations[i] = 0
                            pass_loc = taxi_loc
                            successful_dropoff = True
                            reward = multitaxifuel_rewards_w_idle['intermediate_dropoff']
                    if not successful_dropoff:  # not carrying a passenger
                        reward = multitaxifuel_rewards_w_idle['bad_dropoff']
                    # Turning engine off
                    elif index_action_dictionary[action] == 'turn_engine_off':
                        reward = multitaxifuel_rewards_w_idle['turn_engine_off']
                        self.engine_status_list[taxi] = False
                    elif index_action_dictionary[action] == 'standby':  # standing by engine is on
                        reward = multitaxifuel_rewards_w_idle['standby_engine_on']
                    # taxi refuel
                    elif index_action_dictionary[action] == 'refuel':
                        if taxi_loc in self.fuel_stations:
                            if self.desc[taxi_loc] == self.fuel_type_list[taxi]:
                                fuel = self.max_fuel
                        else:
                            reward = multitaxifuel_rewards_w_idle['bad_refuel']

            # fuel consumption
            if moved:
                if fuel == 0:
                    reward = multitaxifuel_rewards_w_idle['no_fuel']
                else:
                    fuel = max(0, fuel - 1)
                    taxis[taxi] = [row, col]
                    fuels[taxi] = fuel
            if not moved and action < 4:
                reward = multitaxifuel_rewards_w_idle['hit_wall']


            # check for done: we are finished if all the passengers are at their destinations
            done = all(loc == -1 for loc in pass_loc)
            self.dones.append(done)

            # check if all taxis collided
            done = all(self.collided == 1)
            self.dones.append(done)

            rewards.append(reward)
            self.state = [taxis, fuels, pass_start, destinations, pass_loc]
            self.lastaction = actions

        return self.state, rewards, any(self.dones), {}

    def render(self, mode='human', diag=True):
        # renders the state of the environment

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]
        taxis, fuels, pass_start, destinations, pass_locs = self.state

        colors = ['yellow', 'red', 'white', 'green', 'cyan', 'crimson', 'gray', 'magenta'] * 5
        colored = [False for taxi in taxis]

        def ul(x):
            return "_" if x == " " else x

        for i, loc in enumerate(pass_locs):
            if loc > 0:
                taxi_row, taxi_col = taxis[loc - 1]
                out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                    out[1 + taxi_row][2 * taxi_col + 1], colors[loc - 1], highlight=True, bold=True)
                colored[loc - 1] = True
            else:  # passenger in taxi
                pi, pj = pass_start[i]
                out[1 + pi][2 * pj + 1] = utils.colorize(out[1 + pi][2 * pj + 1], 'blue', bold=True)

        for i, taxi in enumerate(taxis):
            if not colored[i]:
                if self.collided[i] == 0:
                    taxi_row, taxi_col = taxi
                    out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                        ul(out[1 + taxi_row][2 * taxi_col + 1]), colors[i], highlight=True)
                else:
                    taxi_row, taxi_col = taxi
                    out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                        ul(out[1 + taxi_row][2 * taxi_col + 1]), 'white', highlight=True)

        for dest in destinations:
            di, dj = dest
            out[1 + di][2 * dj + 1] = utils.colorize(out[1 + di][2 * dj + 1], 'magenta')
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")

        if self.lastaction is not None:
            moves = all_action_names
            output = [moves[i] for i in self.lastaction]
            outfile.write("  ({})\n".format(' ,'.join(output)))
        for i, taxi in enumerate(taxis):
            outfile.write("Taxi{}: Fuel: {}, Location: ({},{})\n".format(i + 1, fuels[i], taxi[0], taxi[1]))
        for i, loc in enumerate(pass_locs):
            start = tuple(pass_start[i])
            end = tuple(destinations[i])
            if loc < 0:
                outfile.write("Passenger{}: Location: Arrived!, Destination: {}\n".format(i + 1, end))
            if loc == 0:
                outfile.write("Passenger{}: Location: {}, Destination: {}\n".format(i + 1, start, end))
            else:
                outfile.write("Passenger{}: Location: Taxi{}, Destination: {}\n".format(i + 1, loc, end))

        # No need to return anything for human
        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()

    def partial_observations(self, state):
        def flatten(x):
            return [item for sub in x for item in sub]

        observations = []
        taxis, fuels, pass_start, dest, pass_loc = state
        pass_info = flatten(pass_start) + flatten(dest) + pass_loc

        for i in range(len(taxis)):
            obs = taxis[i] + [fuels[i]] + pass_info
            obs = np.reshape(obs, [1, len(obs)])
            observations.append(obs)
        return observations

    def get_observation(self, state, agent_index):
        def flatten(x):
            return [item for sub in x for item in sub]

        taxis, fuels, pass_start, dest, pass_loc = state
        pass_info = flatten(pass_start) + flatten(dest) + pass_loc

        obs = taxis[agent_index] + [fuels[agent_index]] + pass_info
        obs = np.reshape(obs, [1, len(obs)])
        return obs
