from pymonntorch import Behavior
import numpy as np


class ConstantCurrent(Behavior):
    def initialize(self, ng):
        self.offset = self.parameter('value')
        ng.I = ng.vector(mode=self.parameter('value'))
        self.with_noise = self.parameter(
            'with_noise', default=False, required=False)
        self.noise_std = self.parameter('noise_std', default=5, required=False)
        self.noise_level = self.parameter(
            'noise_level', default=1, required=False)

    def forward(self, ng):
        if self.with_noise:
            # add a random noise in range (0, 20)
            ng.I.fill_(np.random.normal(0, self.noise_std) *
                       self.noise_level + self.parameter('value'))
        else:
            ng.I.fill_(self.parameter("value"))


class OneStepFunction(Behavior):
    def initialize(self, ng):
        self.value = self.parameter("value")
        self.t0 = self.parameter("t0")
        ng.I = ng.vector(mode=0)

    def forward(self, ng):
        if ng.network.iteration * ng.network.dt >= self.t0:
            ng.I = ng.vector(mode=self.value)


class StepFunction(Behavior):
    def initialize(self, ng):
        self.value = self.parameter("value")
        self.t0 = self.parameter("t0")
        self.pre_iter = 0
        ng.I = ng.vector(mode=0)

    def forward(self, ng):
        if (ng.network.iteration - self.pre_iter) * ng.network.dt >= self.t0:
            self.pre_iter = ng.network.iteration
            ng.I += ng.vector(mode=self.value) * ng.network.dt


class SinFunction(Behavior):
    def initialize(self, ng):
        self.resolution = self.parameter('simulation_iter_no')
        self.value = self.parameter('value')
        self.with_noise = self.parameter('with_noise', False)
        cycles = self.parameter('cycles')

        length = np.pi * 2 * cycles
        self.my_wave = self.value * np.sin(
            np.arange(0, length, length / self.resolution)) + self.value

        if (self.with_noise):
            self.my_wave += np.random.normal(5, 10, len(self.my_wave))

        ng.I = self.my_wave[ng.network.iteration]

    def forward(self, ng):
        ng.I = ng.vector(mode=float(self.my_wave[ng.network.iteration]))
