import gym
from gym import spaces
from gym.utils import seeding

import numpy as np

import sys
sys.path.append('./gym_torcs_kai')

import snakeoil3_gym as snakeoil3

import copy
import collections as col
import os
import time


class TorcsKaiEnv(gym.Env):
    terminal_judge_start = 500  # Speed limit is applied after this step
    termination_limit_progress = 5  # [km/h], episode terminates if car is running slower than this limit
    default_speed = 50

    initial_reset = True

    def __init__(self, vision=False, throttle=False, gear_change=False):
        self.vision = vision
        self.throttle = throttle
        self.gear_change = gear_change

        self.initial_run = True

        self.obsdim = 2 # currently supports 2 (minimum) or 79 (maximum)
        self.maximum_distance = 10000 # Maximum distance of 1 episode

        os.system('pkill torcs')
        time.sleep(0.5)
        if self.vision is True:
            os.system('torcs -nofuel -nodamage -nolaptime  -vision &')
        else:
            if self.obsdim == 79:
                os.system('torcs &')
            else:
                os.system('torcs  -nofuel -nodamage -nolaptime &')
        time.sleep(0.5)
        os.system('sh ./gym_torcs_kai/autostart.sh')
        time.sleep(0.5)

        """
                # Modify here if you use multiple tracks in the environment
                self.client = snakeoil3.Client(p=3101, vision=self.vision)  # Open new UDP in vtorcs
                self.client.MAX_STEPS = np.inf
                client = self.client
                client.get_servers_input()  # Get the initial input from torcs
                obs = client.S.d  # Get the current full-observation from torcs
        """

        if throttle is False:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

        if vision is False:

            if self.obsdim == 79:
                high = np.array([np.pi,  # angle
                                 np.inf, # curLapTime
                                 np.inf, # damage
                                 np.inf, # distFromStart
                                 np.inf, # distRaced

                                 # focus (5 dim.)
                                 200, 200, 200, 200, 200,

                                 np.inf, # fuel
                                 6,      # gear
                                 np.inf, # lastLapTime

                                 # opponents (36 dim.)
                                 200, 200, 200, 200, 200, 200,
                                 200, 200, 200, 200, 200, 200,
                                 200, 200, 200, 200, 200, 200,
                                 200, 200, 200, 200, 200, 200,
                                 200, 200, 200, 200, 200, 200,
                                 200, 200, 200, 200, 200, 200,

                                 np.inf, # racePos
                                 np.inf, # rpm
                                 np.inf, # speedX
                                 np.inf, # speedY
                                 np.inf, # speedZ

                                 # track (19 dim.)
                                 200, 200, 200, 200, 200,
                                 200, 200, 200, 200, 200,
                                 200, 200, 200, 200, 200,
                                 200, 200, 200, 200,

                                 np.inf, # trackPos

                                 # wheelSpinVel (4 dim.)
                                 np.inf, np.inf, np.inf, np.inf,

                                 np.inf, # z
                                 ])

                low = np.array([-np.pi,  # angle
                                  0,     # curLapTime
                                  0,     # damage
                                  0,     # distFromStart
                                  0,     # distRaced

                                  # focus (5 dim.)
                                  0, 0, 0, 0, 0,

                                  0,     # fuel
                                 -1,     # gear
                                  0,     # lastLapTime

                                  # opponents (36 dim.)
                                  0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0,

                                  1,     # racePos
                                  0,     # rpm
                                -np.inf, # speedX
                                -np.inf, # speedY
                                -np.inf, # speedZ

                                  # track (19 dim.)
                                  0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0,
                                  0, 0, 0, 0,

                                -np.inf, # trackPos

                                  # wheelSpinVel (4 dim.)
                                  0, 0, 0, 0,

                                -np.inf, # z
                                 ])

            elif self.obsdim == 2:
                high = np.array([np.pi,   # angle
                                 np.inf]) # trackPos

                low = np.array([-np.pi,   # angle
                                -np.inf]) # trackPos

            else:
                #original
                high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf])
                low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf])

            self.observation_space = spaces.Box(low=low, high=high)
        else:
            high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf, 255])
            low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf, 0])
            self.observation_space = spaces.Box(low=low, high=high)

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
        action_torcs['steer'] = this_action['steer']  # in [-1, 1]

        #  Simple Autnmatic Throttle Control by Snakeoil
        if self.throttle is False:
            target_speed = self.default_speed
            if client.S.d['speedX'] < target_speed - (client.R.d['steer']*50):
                client.R.d['accel'] += .01
            else:
                client.R.d['accel'] -= .01

            if client.R.d['accel'] > 0.2:
                client.R.d['accel'] = 0.2

            if client.S.d['speedX'] < 10:
                client.R.d['accel'] += 1/(client.S.d['speedX']+.1)

            # Traction Control System
            if ((client.S.d['wheelSpinVel'][2]+client.S.d['wheelSpinVel'][3]) -
               (client.S.d['wheelSpinVel'][0]+client.S.d['wheelSpinVel'][1]) > 5):
                action_torcs['accel'] -= .2
        else:
            action_torcs['accel'] = this_action['accel']

        #  Automatic Gear Change by Snakeoil
        if self.gear_change is True:
            action_torcs['gear'] = this_action['gear']
        else:
            #  Automatic Gear Change by Snakeoil is possible
            action_torcs['gear'] = 1

        # Save the privious full-obs from torcs for the reward calculation
        obs_pre = copy.deepcopy(client.S.d)

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
        progress = np.array(obs['speedX']) * np.cos(obs['angle'])

        # designed reward function by [Lau 16]
        # https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html
        reward = obs['speedX'] * np.cos(obs['angle']) \
                 - obs['speedX'] * np.sin(obs['angle']) \
                 - obs['speedX'] * np.abs(obs['trackPos'])

        # Termination judgement #########################
        track = np.array(obs['track'])
        if track.min() < 0:  # Episode is terminated if the car is out of track
            client.R.d['meta'] = True

        if self.terminal_judge_start < self.time_step: # Episode terminates if the progress of agent is small
            if progress < self.termination_limit_progress:
                client.R.d['meta'] = True

        if np.cos(obs['angle']) < 0: # Episode is terminated if the agent runs backward
            client.R.d['meta'] = True

        if obs['distRaced'] >= self.maximum_distance: # Episode terminates when agent reached the maximum distance
            client.R.d['meta'] = True

        if client.R.d['meta'] is True: # Send a reset signal
            print("--> raced: ", obs['distRaced']," m <--")
            self.initial_run = False
            client.respond_to_server()

        self.time_step += 1

        return self.get_obs(), reward, client.R.d['meta'], {}

    def reset(self, relaunch=False):

        self.time_step = 0

        if self.initial_reset is not True:
            self.client.R.d['meta'] = True
            self.client.respond_to_server()

            ## TENTATIVE. Restarting TORCS every episode suffers the memory leak bug!
            if relaunch is True:
                self.reset_torcs()

        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=3101, vision=self.vision)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        self.observation = self.make_observaton(obs)

        self.last_u = None

        self.initial_reset = False
        return self.get_obs()

    def close(self):
        os.system('pkill torcs')

    def render(self, mode='human'):
        #TORCS has monitor of driving, so this method omitted.
        pass

    ####################################### making observation ############################################


    def get_obs(self):
        return self.observation

    def reset_torcs(self):
        os.system('pkill torcs')
        time.sleep(0.5)
        if self.vision is False:
            if self.obsdim == 79:
                os.system('torcs &')
            elif self.obsdim == 2:
                os.system('torcs -nofuel -nodamage -nolaptime &')
            else:
                os.system('torcs -nofuel -nodamage -nolaptime &')
        else:
            os.system('torcs -nofuel -nodamage -nolaptime -vision &')
        time.sleep(0.5)
        os.system('sh ./gym_torcs_kai/autostart.sh')
        time.sleep(0.5)

    def agent_to_torcs(self, u):
        torcs_action = {'steer': u[0]}

        if self.throttle is True:  # throttle action is enabled
            torcs_action.update({'accel': u[1]})

        if self.gear_change is True: # gear change action is enabled
            torcs_action.update({'gear': u[2]})

        return torcs_action


    def obs_vision_to_image_rgb(self, obs_image_vec):
        image_vec =  obs_image_vec
        rgb = []
        temp = []
        # convert size 64x64x3 = 12288 to 64x64=4096 2-D list 
        # with rgb values grouped together.
        # Format similar to the observation in openai gym
        for i in range(0,12286,3):
            temp.append(image_vec[i])
            temp.append(image_vec[i+1])
            temp.append(image_vec[i+2])
            rgb.append(temp)
            temp = []
        return np.array(rgb, dtype=np.uint8)

    def make_observaton(self, raw_obs):
        if self.vision is False:
            if self.obsdim == 79:
                names = ['angle',
                         'curLapTime',
                         'damage',
                         'distFromStart', 'distRaced',
                         'focus',
                         'fuel',
                         'gear',
                         'lastLapTime',
                         'opponents',
                         'racePos',
                         'rpm',
                         'speedX', 'speedY', 'speedZ',
                         'track', 'trackPos',
                         'wheelSpinVel',
                         'z']
                Observation = col.namedtuple('Observaion', names)
                return Observation(angle=np.array(raw_obs['angle'], dtype=np.float32),
                                   curLapTime=np.array(raw_obs['curLapTime'], dtype=np.float32),
                                   damage=np.array(raw_obs['damage'], dtype=np.float32),
                                   distFromStart=np.array(raw_obs['distFromStart'], dtype=np.float32),
                                   distRaced=np.array(raw_obs['distRaced'], dtype=np.float32),
                                   focus=np.array(raw_obs['focus'], dtype=np.float32),
                                   fuel=np.array(raw_obs['fuel'], dtype=np.float32),
                                   gear=np.array(raw_obs['gear'], dtype=np.float32),
                                   lastLapTime=np.array(raw_obs['lastLapTime'], dtype=np.float32),
                                   opponents=np.array(raw_obs['opponents'], dtype=np.float32),
                                   racePos=np.array(raw_obs['racePos'], dtype=np.float32),
                                   rpm=np.array(raw_obs['rpm'], dtype=np.float32),
                                   speedX=np.array(raw_obs['speedX'], dtype=np.float32),
                                   speedY=np.array(raw_obs['speedY'], dtype=np.float32),
                                   speedZ=np.array(raw_obs['speedZ'], dtype=np.float32),
                                   track=np.array(raw_obs['track'], dtype=np.float32),
                                   trackPos=np.array(raw_obs['trackPos'], dtype=np.float32),
                                   wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32),
                                   z=np.array(raw_obs['z'], dtype=np.float32))


            elif self.obsdim == 2:
                return np.array([raw_obs['angle'], raw_obs['trackPos']])


            else:
                names = ['focus',
                         'speedX', 'speedY', 'speedZ',
                         'opponents',
                         'rpm',
                         'track',
                         'wheelSpinVel']
                Observation = col.namedtuple('Observaion', names)
                return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
                                   speedX=np.array(raw_obs['speedX'], dtype=np.float32)/self.default_speed,
                                   speedY=np.array(raw_obs['speedY'], dtype=np.float32)/self.default_speed,
                                   speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/self.default_speed,
                                   opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                                   rpm=np.array(raw_obs['rpm'], dtype=np.float32),
                                   track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                                   wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32))

        else:
            names = ['focus',
                     'speedX', 'speedY', 'speedZ',
                     'opponents',
                     'rpm',
                     'track',
                     'wheelSpinVel',
                     'img']
            Observation = col.namedtuple('Observaion', names)

            # Get RGB from observation
            image_rgb = self.obs_vision_to_image_rgb(raw_obs[names[8]])

            return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
                               speedX=np.array(raw_obs['speedX'], dtype=np.float32)/self.default_speed,
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32)/self.default_speed,
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/self.default_speed,
                               opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                               rpm=np.array(raw_obs['rpm'], dtype=np.float32),
                               track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                               wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32),
                               img=image_rgb)
