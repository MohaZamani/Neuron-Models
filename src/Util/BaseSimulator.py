from pymonntorch import Network, NeuronGroup, Recorder, EventRecorder
import torch
from matplotlib import pyplot as plt
from Util.TimeResolution import TimeResolution


class BaseSimulator():
    def __init__(self, dt: float) -> None:
        self.net = Network(behavior={1: TimeResolution(dt=dt)},
                           dtype=torch.float64, tag='NET')

    def buildNeuronGroup(self, current, neuron_model) -> None:
        self.ng = NeuronGroup(
            size=1,
            net=self.net,
            behavior={
                2: current,
                5: neuron_model,
                8: Recorder(variables=["v", "I"], tag="ng_rec, ng_recorder"),
                9: Recorder(variables=['iter', 'spike_counts'], tag='iter_rec, spike_n_rec'),
                10: EventRecorder("spike", tag="ng_evrec"),
            },
            tag="NG",
        )

    def exec(self, iterations: int) -> None:
        self.net.initialize()
        self.net.simulate_iterations(iterations)

    def plot_voltage_current(self):
        # plot 1:
        plt.figure(figsize=(20, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.net["v", 0])
        plt.xlabel('iteration')
        plt.ylabel('Voltage')

        # plot 2:
        plt.subplot(1, 2, 2)
        plt.plot(self.net["I", 0])
        plt.xlabel('iteration')
        plt.ylabel('input current')

        plt.show()

    def plot_just_voltage(self):
        plt.figure(figsize=(4.5, 2.2))
        plt.plot(self.net["v", 0])
        plt.xlabel('iteration')
        plt.ylabel('Voltage')
        plt.show()

    def plot_spike_times(self):
        plt.figure(figsize=(4.5, 3.5))
        plt.scatter(self.net["spike", 0], self.net["spike", 0])
        plt.title('spike times')
        plt.show()
