from pymonntorch import Behavior


class TimeResolution(Behavior):
    def initialize(self, ng):
        ng.dt = self.parameter('dt', 0.01)
