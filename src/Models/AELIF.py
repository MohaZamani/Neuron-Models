from pymonntorch import Behavior


from pymonntorch import Behavior
import math


class AELIF(Behavior):

    def initialize(self, ng):
        self.tau = self.parameter('tau')
        self.u_rest = self.parameter('u_rest')
        self.u_reset = self.parameter('u_reset')
        self.threshold = self.parameter('threshold')
        self.threshold_rh = self.parameter('threshold_rh')
        self.R = self.parameter('R')
        self.delta_t = self.parameter('delta_t')
        self.a_k = self.parameter('a_k')
        self.b_k = self.parameter('b_k')
        self.tau_k = self.parameter('tau_k')
        self.with_refactory = self.parameter('with_refactory', False)

        if self.with_refactory:
            self.t_ref = self.parameter('t_ref')

        ng.v = ng.vector(mode=self.parameter(
            'v_init_mode', default='normal(0.5, 0.05)'))

        ng.w = ng.vector(mode=self.parameter(
            'w_init_mode', default='normal(0.5, 0.05)'))
        ng.spike_counts = ng.vector(0)

        ng.spike = ng.v >= self.threshold
        ng.v[ng.spike] = self.u_reset
        ng.spike_counts += ng.spike

        ng.iter = ng.vector(0)
        ng.last_spike = ng.vector(-1)

    def forward(self, ng):
        ng.iter = ng.vector(ng.network.iteration)

        ng.w += ((self.a_k * (ng.v - self.u_rest) - ng.w +
                 self.b_k * self.tau_k * (ng.spike_counts)) / self.tau_k) * ng.network.dt

        if (self.with_refactory):
            if (ng.network.iteration * ng.network.dt < self.t_ref + ng.last_spike):
                ng.I -= ng.I

        ng.v += (((-1 * (ng.v - self.u_rest)) + (self.delta_t * math.exp((ng.v - self.threshold_rh) /
                 self.delta_t)) - 1 * (self.R * ng.w) + (self.R * ng.I))) / (self.tau) * ng.network.dt

        ng.spike = ng.v >= self.threshold
        ng.v[ng.spike] = self.u_reset
        ng.spike_counts += ng.spike
        ng.last_spike[ng.spike] = ng.network.iteration * ng.network.dt
