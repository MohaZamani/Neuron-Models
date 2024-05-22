from pymonntorch import Behavior
import numpy as np


class NoiseCurrent(Behavior):

    def initialize(self, ng):
        self.noise_mode = self.parameter('noise_mode')
        self.size = self.parameter('simulation_iter_no')

        self.current = self.__init_noise()

        ng.I = self.current[ng.network.iteration]

    def forward(self, ng):
        ng.I = ng.vector(mode=float(self.current[ng.network.iteration]))

    def __init_noise(self) -> np.array:
        if (self.noise_mode == 'gaussian'):
            gaussian_noise = np.random.normal(self.parameter(
                'mean'), self.parameter('std_deviation'), self.size)
            return gaussian_noise
        elif (self.noise_mode == 'uniform'):
            uniform_noise = np.random.uniform(self.parameter(
                'low', default=0), self.parameter('high', 2000), size=self.size)
            return uniform_noise
