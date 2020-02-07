# Gym-TORCS-Kai

[Japanese](README_ja.md)

**Gym-TORCS-Kai** ("Kai" means "MOD(ification)" in Japanese) is a self-modified version of Gym-TORCS, the environment for reinforcement learning in The Open Racing Car Simulator (TORCS).
WIP.


## Install
Please read the descriptions in the [original repository](https://github.com/ugo-nama-kun/gym_torcs).

## Requirements
We are assuming you are using Ubuntu 16.04 LTS machine and installed
* Python 3
* xautomation (http://linux.die.net/man/7/xautomation)
* OpenAI-Gym (https://github.com/openai/gym)
* numpy
* vtorcs-RL-color (installation of vtorcs-RL-color is explained in vtorcs-RL-color directory)

## Usage
Put this **gym_torcs_kai** repository in place of RL code, and use the environment name
```
gym_torcs_kai:GymTorcsKai-v0
```

### New functions

- env.testset(test) : To / not to print the logging data.

```
env.testset(True): test mode. Env does not print the log.
env.testset(False) : train mode. Env prints the log. 
```

- env.set_params(throttle, gear, dim, max_dist, targ_speed): Set parameters for environment.
  - throttle : bool. to / not to use throttle control.
  - gear : bool. to / not to use gear change control.
  - dim : int. the size of dimension in observation. see the code for details.
  - max_dist : float. the maximum distance in one race.
  - targ_speed : float. the target speed to adjust.

## History
- 2019.11. : beta version **"v0"** released
- 2019.12. : fixed some bugs
- 2020.02. : fixed some bugs, add set_params

## Acknowledgement
- **gym_torcs** : developed by Naoto Yoshida.

