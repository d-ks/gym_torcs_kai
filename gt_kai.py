# Gym-TORCS-Kai Environment for Reinforcement Learning in TORCS
# original author : Naoto Yoshida
# (https://github.com/ugo-nama-kun/gym_torcs)
# modified version author : Daiko Kishikawa
#
# This environment is under the modification. (2019.12)
#

import gym
from gym import spaces
from gym.utils import seeding

import numpy as np

import sys

sys.path.append("./gym_torcs_kai")

import snakeoil3_gym as snakeoil3

import os
import time


class TorcsKaiEnv(gym.Env):
    # Speed limit is applied after this step
    terminal_judge_start = 500

    # episode terminates if car is running slower than this limit
    termination_limit_progress = 5

    initial_reset = True

    def __init__(self, throttle=False, gear_change=False):
        self.throttle = throttle
        self.gear_change = gear_change

        self.initial_run = True

        self.obsdim = 2  # currently supports 2 (minimum) or 79 (maximum)
        self.maximum_distance = 1621.73  # Maximum distance of 1 episode
        self.default_speed = 200  # Target speed

        self.speedY = 0
        self.time = 0
        self.time_step = 0
        self.Yaclist = []

        self.reward_range = (-10, 10)

        self.testmode = False

        # History
        self.poshis = []
        self.anglehis = []
        self.sphis = []

        os.system("pkill torcs")
        time.sleep(0.5)

        if self.obsdim == 79:
            os.system("torcs &")
        else:
            os.system("torcs  -nofuel -nodamage -nolaptime &")
        time.sleep(0.5)
        os.system("sh ./gym_torcs_kai/autostart.sh")
        time.sleep(0.5)

        """
                # Modify here if you use multiple tracks in the environment
                self.client = snakeoil3.Client(p=3101, vision=False)  # Open new UDP in vtorcs
                self.client.MAX_STEPS = np.inf
                client = self.client
                client.get_servers_input()  # Get the initial input from torcs
                obs = client.S.d  # Get the current full-observation from torcs
        """

        if throttle is False:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

        if self.obsdim == 79:
            high = np.array([np.pi,  # angle
                             np.inf,  # curLapTime
                             np.inf,  # damage
                             np.inf,  # distFromStart
                             np.inf,  # distRaced

                             # focus (5 dim.)
                             200, 200, 200, 200, 200,

                             np.inf,  # fuel
                             6,  # gear
                             np.inf,  # lastLapTime

                             # opponents (36 dim.)
                             200, 200, 200, 200, 200, 200,
                             200, 200, 200, 200, 200, 200,
                             200, 200, 200, 200, 200, 200,
                             200, 200, 200, 200, 200, 200,
                             200, 200, 200, 200, 200, 200,
                             200, 200, 200, 200, 200, 200,

                             np.inf,  # racePos
                             np.inf,  # rpm
                             np.inf,  # speedX
                             np.inf,  # speedY
                             np.inf,  # speedZ

                             # track (19 dim.)
                             200, 200, 200, 200, 200,
                             200, 200, 200, 200, 200,
                             200, 200, 200, 200, 200,
                             200, 200, 200, 200,

                             np.inf,  # trackPos

                             # wheelSpinVel (4 dim.)
                             np.inf, np.inf, np.inf, np.inf,

                             np.inf,  # z
                             ])

            low = np.array([-np.pi,  # angle
                            0,  # curLapTime
                            0,  # damage
                            0,  # distFromStart
                            0,  # distRaced

                            # focus (5 dim.)
                            0, 0, 0, 0, 0,

                            0,  # fuel
                            -1,  # gear
                            0,  # lastLapTime

                            # opponents (36 dim.)
                            0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0,

                            1,  # racePos
                            0,  # rpm
                            -np.inf,  # speedX
                            -np.inf,  # speedY
                            -np.inf,  # speedZ

                            # track (19 dim.)
                            0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0,
                            0, 0, 0, 0,

                            -np.inf,  # trackPos

                            # wheelSpinVel (4 dim.)
                            0, 0, 0, 0,

                            -np.inf,  # z
                            ])

        elif self.obsdim == 2:
            high = np.array([np.pi,  # angle
                             np.inf])  # trackPos

            low = np.array([-np.pi,  # angle
                            -np.inf])  # trackPos

        elif self.obsdim == 31:

            high = np.array([np.pi,  # angle
                             6,  # gear
                             np.inf,  # rpm
                             np.inf,  # speedX
                             np.inf,  # speedY
                             np.inf,  # speedZ
                             # track (19 dim.)
                             200, 200, 200, 200, 200,
                             200, 200, 200, 200, 200,
                             200, 200, 200, 200, 200,
                             200, 200, 200, 200,
                             np.inf,  # trackPos
                             # wheelSpinVel (4 dim.)
                             np.inf, np.inf, np.inf, np.inf,
                             np.inf,  # z
                             ])

            low = np.array([-np.pi,  # angle
                            -1,  # gear
                            0,  # rpm
                            -np.inf,  # speedX
                            -np.inf,  # speedY
                            -np.inf,  # speedZ
                            # track (19 dim.)
                            0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0,
                            0, 0, 0, 0,
                            -np.inf,  # trackPos
                            # wheelSpinVel (4 dim.)
                            0, 0, 0, 0,
                            -np.inf,  # z
                            ])
        else:
            low = None
            high = None

        self.observation_space = spaces.Box(low=low, high=high)

    def testset(self, test):
        self.testmode = test

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        # convert thisAction to the actual torcs actionstr
        client = self.client

        this_action = self.agent_to_torcs(u)

        # Apply Action
        action_torcs = client.R.d

        # Steering
        action_torcs["steer"] = this_action["steer"]  # in [-1, 1]

        #  Simple Autnmatic Throttle Control by Snakeoil
        if self.throttle is False:
            target_speed = self.default_speed
            if client.S.d["speedX"] < target_speed - (client.R.d["steer"] * 50):
                if client.R.d["accel"] + 0.1 <= 1:
                    client.R.d["accel"] += 0.1
            else:
                if client.R.d["accel"] - 0.1 >= 0:
                    client.R.d["accel"] -= 0.1

            if client.S.d["speedX"] < 10:
                if (client.S.d["speedX"] + 0.1) != 0:
                    client.R.d["accel"] += 1 / (client.S.d["speedX"] + 0.1)

            # Traction Control System
            if (client.S.d["wheelSpinVel"][2] + client.S.d["wheelSpinVel"][3]) - (
                client.S.d["wheelSpinVel"][0] + client.S.d["wheelSpinVel"][1]
            ) > 5:
                action_torcs["accel"] -= 0.2
        else:
            action_torcs["accel"] = this_action["accel"]

        #  Automatic Gear Change by Snakeoil
        if self.gear_change is True:
            action_torcs["gear"] = this_action["gear"]
        else:
            #  Automatic Gear Change by Snakeoil is possible
            action_torcs["gear"] = 1

        # One-Step Dynamics Update #################################
        # Apply the Agent's action into torcs
        client.respond_to_server()
        # Get the response of TORCS
        client.get_servers_input()

        # Get the current full-observation from torcs
        obs = client.S.d

        # Make an obsevation from a raw observation vector from TORCS
        self.observation = self.make_observaton(obs)

        # Reward setting Here #######################################
        # direction-dependent positive reward
        progress = np.array(obs["speedX"]) * np.cos(obs["angle"])

        Yac = (obs["speedY"] - self.speedY) / (obs["curLapTime"] - self.time)
        self.speedY = obs["speedY"]
        self.time = obs["curLapTime"]
        self.Yaclist.append(Yac)

        self.poshis.append(obs["trackPos"])
        self.anglehis.append(obs["angle"])
        self.sphis.append(obs["speedX"])

        # reward is speed at basis
        eta_Yac = 1
        r_Yac = 1 / ((Yac / eta_Yac) ** 2 + 1)

        # reward for angle : 0 ~ 1
        eta_angle = 0.01
        r_angle = 1 / ((obs["angle"] / eta_angle) ** 2 + 1)

        # reward for trackPos : 0 ~ 1
        eta_pos = 0.01
        r_trackPos = 1 / ((obs["trackPos"] / eta_pos) ** 2 + 1)

        # reward for speedX : 0 ~ 1
        maxspeed = 100
        if obs["speedX"] >= 0:
            r_speed = min(obs["speedX"] / maxspeed, 1)
        else:
            r_speed = 0

        # reward function: -1 ~ 1
        reward = 0.2 * r_angle + 0.2 * r_trackPos + 0.3 * r_speed + 0.3 * r_Yac

        if np.abs(Yac) > 5:
            reward = -min(np.abs(Yac) / 250, 1)

        # Termination judgement #########################
        track = np.array(obs["track"])
        # Episode is terminated if the car is out of track
        if track.min() < 0:
            reward = -10
            client.R.d["meta"] = True

        # Episode terminates if the progress of agent is small
        if self.terminal_judge_start < self.time_step:
            if progress < self.termination_limit_progress:
                reward = -10
                client.R.d["meta"] = True

        # Episode is terminated if the agent runs backward
        if np.cos(obs["angle"]) < 0 or obs["distRaced"] < 0:
            reward = -10
            client.R.d["meta"] = True

        # Episode terminates when agent reached the maximum distance
        if obs["distRaced"] >= self.maximum_distance:
            reward = 10
            client.R.d["meta"] = True

        if client.R.d["meta"] is True:  # Send a reset signal
            # punish by sum from history of them
            poshis = np.array(self.poshis)
            anglehis = np.array(self.anglehis)
            sphis = np.array(self.sphis)
            Yachis = np.array(self.Yaclist)

            if self.testmode == False:
                print("---> raced: ", obs["distRaced"], " m <---")
                print("--- maxYac: ", np.max(Yachis), " km/h/s ---")
                print("--- minYac: ", np.min(Yachis), " km/h/s ---")
                if abs(np.max(Yachis)) >= abs(np.min(Yachis)):
                    absmaxYac = abs(np.max(Yachis))
                else:
                    absmaxYac = abs(np.min(Yachis))
                print("--- absmaxYac: ", absmaxYac, " km/h/s ---")
                print(
                    "--- meanYac: ",
                    np.mean(Yachis),
                    " km/h/s +- ",
                    np.std(Yachis),
                    "---",
                )
                print("--- medianYac: ", np.median(Yachis), " km/h/s ---")
                print(
                    "--- trackPos_mean: ",
                    np.mean(poshis),
                    " +- ",
                    np.std(poshis),
                    " ---",
                )
                print(
                    "--- angle_mean : ",
                    np.mean(anglehis),
                    " +- ",
                    np.std(anglehis),
                    " rad ---",
                )
                print(
                    "--- speedX_mean: ", np.mean(sphis), " +- ", np.std(sphis), " ---"
                )
            self.initial_run = False
            client.respond_to_server()

        self.time_step += 1

        return self.get_obs(), reward, client.R.d["meta"], {}

    def reset(self, relaunch=False):

        self.time_step = 0

        if self.initial_reset is not True:
            self.client.R.d["meta"] = True
            self.client.respond_to_server()

            ## TENTATIVE. Restarting TORCS every episode suffers the memory leak bug!
            if relaunch is True:
                self.reset_torcs()

        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=3101, vision=False)  # Open new UDP in vtorcs

        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        self.observation = self.make_observaton(obs)

        self.last_u = None

        self.speedY = obs["speedY"]
        self.time = obs["curLapTime"]

        self.Yaclist = []
        self.poshis = []
        self.anglehis = []
        self.sphis = []

        self.initial_reset = False
        return self.get_obs()

    def close(self):
        os.system("pkill torcs")

    def render(self, mode="human"):
        # TORCS has monitor of driving, so this method omitted.
        pass

    ####################################### making observation ############################################

    def get_obs(self):
        return self.observation

    def reset_torcs(self):
        os.system("pkill torcs")
        time.sleep(0.5)

        if self.obsdim == 79:
            os.system("torcs &")
        elif self.obsdim == 2:
            os.system("torcs -nofuel -nodamage -nolaptime &")
        else:
            os.system("torcs -nofuel -nodamage -nolaptime &")

        time.sleep(0.5)
        os.system("sh ./gym_torcs_kai/autostart.sh")
        time.sleep(0.5)

    def agent_to_torcs(self, u):
        torcs_action = {"steer": u[0]}

        if self.throttle is True:  # throttle action is enabled
            torcs_action.update({"accel": u[1]})

        if self.gear_change is True:  # gear change action is enabled
            torcs_action.update({"gear": u[2]})

        return torcs_action

    def make_observaton(self, raw_obs):
        if self.obsdim == 79:
            obs1 = np.array(
                [
                    raw_obs["angle"],
                    raw_obs["curLapTime"],
                    raw_obs["damage"],
                    raw_obs["distFromStart"],
                    raw_obs["distRaced"],
                ]
            )
            focus = raw_obs["focus"]
            obs2 = np.array([raw_obs["fuel"], raw_obs["gear"], raw_obs["lastLapTime"]])
            opponents = raw_obs["opponents"]
            obs3 = np.array(
                [
                    raw_obs["racePos"],
                    raw_obs["rpm"],
                    raw_obs["speedX"],
                    raw_obs["speedY"],
                    raw_obs["speedZ"],
                ]
            )
            track = raw_obs["track"]
            trackPos = np.array([raw_obs["trackPos"]])
            wheelSpinVel = raw_obs["wheelSpinVel"]
            z = np.array(raw_obs["z"])
            observ = np.hstack(
                [obs1, focus, obs2, opponents, obs3, track, trackPos, wheelSpinVel, z]
            )
            return observ

        elif self.obsdim == 2:
            return np.array([raw_obs["angle"], raw_obs["trackPos"]])

        elif self.obsdim == 31:

            obs1 = np.array(
                [
                    raw_obs["angle"],
                    raw_obs["gear"],
                    raw_obs["rpm"],
                    raw_obs["speedX"],
                    raw_obs["speedY"],
                    raw_obs["speedZ"],
                ]
            )

            trackPos = np.array([raw_obs["trackPos"]])
            z = np.array(raw_obs["z"])

            observ = np.hstack(
                [obs1, raw_obs["track"], trackPos, raw_obs["wheelSpinVel"], z]
            )

            return observ

        else:
            return None
