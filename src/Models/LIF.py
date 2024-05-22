from pymonntorch import Behavior


class LIF(Behavior):

    def initialize(self, ng):
        self.tau = self.parameter('tau')
        self.u_rest = self.parameter('u_rest')
        self.u_reset = self.parameter('u_reset')
        self.threshold = self.parameter('threshold')
        self.R = self.parameter('R')
        self.with_refactory = self.parameter('with_refactory', False)
        ng.spike_counts = ng.vector(0)

        if self.with_refactory:
            self.t_ref = self.parameter('t_ref')

        ng.v = ng.vector(mode=self.parameter(
            'v_init_mode', default='normal(0.5, 0.05)'))

        ng.spike = ng.v >= self.threshold
        ng.v[ng.spike] = self.u_reset
        ng.spike_counts += ng.spike

        ng.iter = ng.vector(0)

        ng.last_spike = ng.vector(-1)

    def forward(self, ng):
        ng.iter = ng.vector(ng.network.iteration)

        if (self.with_refactory):
            if (ng.network.iteration * ng.network.dt < self.t_ref + ng.last_spike):
                ng.I -= ng.I

        ng.v += (-1 * (ng.v - self.u_rest) + (self.R * ng.I)) / \
            self.tau * ng.network.dt

        ng.spike = ng.v >= self.threshold
        ng.last_spike[ng.spike] = ng.network.iteration * ng.network.dt
        ng.v[ng.spike] = self.u_reset
        ng.spike_counts += ng.spike
