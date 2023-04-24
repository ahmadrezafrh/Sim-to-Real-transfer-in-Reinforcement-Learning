"""Implementation of the Hopper environment supporting
domain randomization optimization."""
import csv
import pdb
from copy import deepcopy

import numpy as np
import gym
from gym import utils
from .mujoco_env import MujocoEnv
from scipy.stats import truncnorm

class CustomHopper(MujocoEnv, utils.EzPickle):
    def __init__(self, domain=None, show=False):
        self.randomization = False
        self.show = show
        self.ep_count = 0
        self.n_distributions = 1
        MujocoEnv.__init__(self, 4)
        utils.EzPickle.__init__(self)
        
        self.original_masses = np.copy(self.sim.model.body_mass[1:])    # Default link masses
        if domain == 'source':  # Source environment has an imprecise torso mass (1kg shift)
            self.sim.model.body_mass[1] -= 1.0


    def set_distributions(self, distributions):
        self.randomization = True
        self.distributions = distributions
        self.n_distributions = len(distributions)
        

            
    def set_random_parameters(self):
        """Set random masses
        TODO
        """
        self.set_parameters(self.sample_parameters())

    def sample_parameters(self):
        """Sample masses according to a domain randomization distribution
        TODO
        """
        if self.n_distributions==1:
            task = np.random.uniform(self.distributions[0][0], self.distributions[0][1], size = 3)
        else :
            task = np.empty(self.n_distributions, dtype=np.float64)
            for i in range(self.n_distributions):
                sample = np.random.uniform(self.distributions[i][0], self.distributions[i][1])
                task[i] = sample
        return task

    def get_parameters(self):
        """Get value of mass for each link"""
        masses = np.array( self.sim.model.body_mass[1:] )
        return masses

    def set_parameters(self, task):
        """Set each hopper link's mass to a new value"""
        self.sim.model.body_mass[2:] = task
        if self.show:
            self.ep_count +=1
            print(f'\nepisode {self.ep_count} finished.')
            print('new dyynamics parameters:', self.get_parameters())

    def step(self, a):
        """Step the simulation to the next timestep

        Parameters
        ----------
        a : ndarray,
            action to be taken at the current timestep
        """
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        self.done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()
        

        if self.done & self.randomization:
            self.set_random_parameters()
    
        return ob, reward, self.done, {}

    def _get_obs(self):
        """Get current state"""
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
        ])

    def reset_model(self):
        """Reset the environment to a random initial state"""
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20



"""
    Registered environments
"""
gym.envs.register(
        id="CustomHopper-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
)

gym.envs.register(
        id="CustomHopper-source-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source"}
)

gym.envs.register(
        id="CustomHopper-target-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "target"}
)




# def randomize_mass(env, loweBound, upperBound):
#     new_env = env
#     # Define the distribution for randomizing the mass
#     distribution = [loweBound, upperBound]
#     randomized_masses = numpy.random.uniform(distribution[0], distribution[1], size= 3)
#     # Set the randomized masses
#     new_env.model.body_mass[2:] = randomized_masses
#     return new_env








