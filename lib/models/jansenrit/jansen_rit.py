import numpy as np
import matplotlib.pyplot as plt

"""
Jansen and Rit Model
"""

class JansenRit(object):

    def __init__(self, C = 0):
        # Model Parameters (Jansen and Rit, 1995)
        self.A = 3.25  # [mV]
        self.B = 22.0  # [mV]
        self.a = 100.0  # [s^-1]
        self.b = 50.0  # [s^-1]
        self.v0 = 6.0  # [mV]
        self.e0 = 2.5  # [s^-1]
        self.r = 0.56  # [mV^-1]

        # Time
        self.dt = 0.001

        # Set layer connectivity
        self._C = C
        self.C1 = self._C
        self.C2 = 0.8 * self._C
        self.C3 = 0.25 * self._C
        self.C4 = 0.25 * self._C

        # Initialize
        self.y = np.zeros(6)
        self.activity = None

    def simulate(self, t, n_save = 2, t0 = 1):

        t = t + t0
        dt_save = self.dt * n_save
        n_sim_steps = int(t / self.dt + 1)
        n_save_steps = int(t / dt_save + 1)

        self.activity = np.empty(n_save_steps)
        self.input = np.random.uniform(120, 320, n_sim_steps)

        for i in range(1, n_sim_steps):
            self._step(self.input[i])
            if not (i % n_save):
                i_save = int(i / n_save)
                self.activity[i_save] = self.y[1] - self.y[2]

        crop = int(t0 / self.dt)
        self.activity = self.activity[crop:]


    def _sigm(self, v):
        return 2.0 * self.e0 / (1.0 + np.exp(self.r * (self.v0 - v)))

    def _step(self, p):
        self.y[0] += self.dt * self.y[3]
        self.y[3] += self.dt * (self.A * self.a * self._sigm(self.y[1] - self.y[2]) -
                                2.0 * self.a * self.y[3] -
                                self.a * self.a * self.y[0])
        self.y[1] += self.dt * self.y[4]
        self.y[4] += self.dt * (self.A * self.a * (p + self.C2 * self._sigm(self.C1 * self.y[0])) -
                                2.0 * self.a * self.y[4] -
                                self.a * self.a * self.y[1])
        self.y[2] += self.dt * self.y[5]
        self.y[5] += self.dt * (self.B * self.b * self.C4 * self._sigm(self.C3 * self.y[0]) -
                               2.0 * self.b * self.y[5] - self.b * self.b * self.y[2])

    def _reset(self):
        self.y = np.zeros(6)
        self.activity = None

    @property
    def C(self):
        return self._C

    @C.setter
    def C(self, C):
        self._reset()
        self._C = C
        self.C1 = self._C
        self.C2 = 0.8 * self._C
        self.C3 = 0.25 * self._C
        self.C4 = 0.25 * self._C

C_array = [68, 128, 135, 270, 675, 1350]
jrit = JansenRit()

fig, ax = plt.subplots(6)

for ix in range(6):
    jrit.C = C_array[ix]
    jrit.simulate(10)
    ax[ix].plot(jrit.activity[:1000], lw=2.0, color=[0.0,0.0,0.0])
plt.savefig('figure_12.png', dpi=300)