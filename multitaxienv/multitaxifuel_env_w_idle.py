# -*- coding: utf-8 -*-

import sys
import gym
from contextlib import closing
from io import StringIO
from gym import utils
from gym.envs.toy_text import discrete
import numpy as np
import itertools
import random
from .config import multitaxifuel_rewards_w_idle

MAP = [
    "+---------+",
    "|X: |F: :X|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|X| :G|X: |",
    "+---------+",
]


class MultiTaxiFuelEnv(gym.Env):
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
    - 6: standby while engine is off
    - 7: turn engine on
    - 8: turn engine off
    - 9: standby engine on
    - 10: refuel fuel tank


    Rewards:
    There is a reward of -1 for each action and an additional reward of +20 for delivering the passenger.
    There is a reward of -10 for executing actions "pickup", "dropoff", and "refuel" illegally.

    Rendering:
    - blue: passenger
    - magenta: destination
    - yellow: empty taxi
    - green: full taxi
    - other letters (R, G, Y and B): locations for passengers and destinations

    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, num_taxis=2, num_passengers=1, max_fuel=8, map=MAP):
        self.desc = np.asarray(map, dtype='c')

        self.num_rows = num_rows = len(self.desc) - 2
        self.num_columns = num_columns = len(self.desc[0][1:-1:2])

        self.locs = []
        self.fuel_station1 = None
        self.fuel_station2 = None
        self.fuel_stations = []

        for i, row in enumerate(self.desc[1:-1]):
            for j, char in enumerate(row[1:-1:2]):
                loc = [i, j]
                if char == b'X':
                    self.locs.append(loc)
                elif char == b'F':
                    self.fuel_station1 = loc
                    self.fuel_stations.append(loc)
                elif char == b'G':
                    self.fuel_station2 = loc
                    self.fuel_stations.append(loc)

        self.coordinates = [[i, j] for i in range(num_rows) for j in range(num_columns)]

        self.max_fuel = max_fuel

        self.num_taxis = self.num_agents = num_taxis
        self.is_engine_on = list(np.ones(num_taxis).astype(bool))
        self.num_passengers = num_passengers

        self.num_actions = 11
        self.action_space = gym.spaces.MultiDiscrete([8 for _ in range(self.num_taxis)])
        self.lastaction = None

        self.seed()
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        taxis = random.sample(self.coordinates, self.num_taxis)
        fuels = [self.max_fuel for _ in range(self.num_taxis)]
        pass_start = [start for start in random.choices(self.locs, k=self.num_passengers)]
        destinations = [random.choice([x for x in self.locs if x != start]) for start in pass_start]
        pass_loc = [0 for _ in range(self.num_passengers)]
        self.state = [taxis, fuels, pass_start, destinations, pass_loc]
        self.lastaction = None
        return self.state

    def step(self, actions):
        max_row = self.num_rows - 1
        max_col = self.num_columns - 1
        rewards = []
        dones = []
        for taxi, action in enumerate(actions):
            taxis, fuels, pass_start, destinations, pass_loc = self.state
            taxi_loc = taxis[taxi]
            row, col = taxi_loc
            fuel = fuels[taxi]
            is_engine_on = self.is_engine_on[taxi]

            reward = multitaxifuel_rewards_w_idle['step']  # default reward when there is no pickup/dropoff
            done = False
            moved = False

            if not is_engine_on:
                if action == 6:  # standby while engine is off
                    reward = multitaxifuel_rewards_w_idle['standby_engine_off']
                elif action == 7:  # turn engine on
                    reward = multitaxifuel_rewards_w_idle['turn_engine_on']
                    self.is_engine_on[taxi] = True
            elif is_engine_on:
                # movement
                if action == 0:  # south
                    if row != max_row:
                        moved = True
                    row = min(row + 1, max_row)
                elif action == 1:  # north
                    if row != 0:
                        moved = True
                    row = max(row - 1, 0)
                if action == 2 and self.desc[1 + row, 2 * col + 2] == b":":  # east
                    if col != max_col:
                        moved = True
                    col = min(col + 1, max_col)
                elif action == 3 and self.desc[1 + row, 2 * col] == b":":  # west
                    if col != 0:
                        moved = True
                    col = max(col - 1, 0)

                # pickup/dropoff
                elif action == 4:  # pickup
                    successful_pickup = False
                    for i, loc in enumerate(pass_loc):
                        if loc == 0 and taxi_loc == pass_start[i] and taxi + 1 not in pass_loc:
                            pass_loc[i] = taxi + 1
                            successful_pickup = True
                            reward = multitaxifuel_rewards_w_idle['pickup']
                            break  # Picks up first passenger, modify this if capacity increases
                    if not successful_pickup:  # passenger not at location
                        reward = multitaxifuel_rewards_w_idle['bad_pickup']
                elif action == 5:  # dropoff
                    successful_dropoff = False
                    for i, loc in enumerate(pass_loc):  # at destination
                        if loc == taxi + 1 and taxi_loc == destinations[i]:
                            pass_loc[i] = -1
                            reward = multitaxifuel_rewards_w_idle['final_dropoff']
                            successful_dropoff = True
                        elif loc == taxi + 1:  # drops off passenger
                            pass_loc[i] = 0
                            pass_start[i] = taxi_loc
                            successful_dropoff = True
                            reward = multitaxifuel_rewards_w_idle['intermediate_dropoff']
                    if not successful_dropoff:  # not carrying a passenger
                        reward = multitaxifuel_rewards_w_idle['bad_dropoff']
                # Turning engine off
                elif action == 8:
                    reward = multitaxifuel_rewards_w_idle['turn_engine_off']
                    self.is_engine_on[taxi] = False
                elif action == 9: # standing by engine is on
                    reward = multitaxifuel_rewards_w_idle['standby_engine_on']
                # taxi refuel
                elif action == 10:
                    if taxi_loc in self.fuel_stations:
                        fuel = self.max_fuel
                    else:
                        reward = multitaxifuel_rewards_w_idle['bad_refuel']

            # fuel consumption
            if moved:
                if fuel == 0:
                    reward = multitaxifuel_rewards_w_idle['no_fuel']
                fuel = max(0, fuel - 1)
            if not moved and action < 4:
                reward = multitaxifuel_rewards_w_idle['hit_wall']

            taxis[taxi] = [row, col]
            fuels[taxi] = fuel

            # check for done: we are finished if all the passengers are at their destinations
            done = all(loc == -1 for loc in pass_loc)
            dones.append(done)

            rewards.append(reward)
            self.state = [taxis, fuels, pass_start, destinations, pass_loc]
            self.lastaction = actions

        return self.state, rewards, any(dones), {}

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
            if loc > 0:  # passenger in taxi
                taxi_row, taxi_col = taxis[loc - 1]
                out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                    out[1 + taxi_row][2 * taxi_col + 1], colors[loc - 1], highlight=True, bold=True)
                colored[loc - 1] = True
            elif loc == 0:  # passenger not in taxi
                pi, pj = pass_start[i]
                if out[1 + pi][2 * pj + 1] != 'X':
                    out[1 + pi][2 * pj + 1] = 'P'
                out[1 + pi][2 * pj + 1] = utils.colorize(out[1 + pi][2 * pj + 1], 'blue', bold=True)

        for i, taxi in enumerate(taxis):
            if not colored[i]:
                taxi_row, taxi_col = taxi
                out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                    ul(out[1 + taxi_row][2 * taxi_col + 1]), colors[i], highlight=True)

        for dest in destinations:
            di, dj = dest
            out[1 + di][2 * dj + 1] = utils.colorize(out[1 + di][2 * dj + 1], 'magenta')
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")

        if self.lastaction is not None:
            moves = ["South", "North", "East", "West", "Pickup", "Dropoff", "Refuel", "Nothing"]
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

    def get_num_jointactions(self):
        return (self.num_actions) ** (self.num_agents)
