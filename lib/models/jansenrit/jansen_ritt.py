import numpy as np
import matplotlib.pyplot as plt

def jansenrit(C):
    dt=0.001
    A = 3.25 #mV
    B = 22.0 #mV
    a = 100.0 #s-1
    b = 50.0 #s-1
    v0 = 6.0 #mV
    e0 = 2.5 #s-1
    r = 0.56 #mV-1
    C1 = C
    C2 = 0.8 * C
    C3 = 0.25 * C
    C4 = 0.25 * C

    T = int(10/dt)

    y, y0, y1, y2, y3, y4, y5 = np.zeros(T), np.zeros(T), \
                             np.zeros(T), np.zeros(T), \
                             np.zeros(T), np.zeros(T), \
                             np.zeros(T)

    p = np.random.uniform(120, 320, T)

    def Sigm(v):
        return 2.0 * e0 / (1.0 + np.exp(r * (v0 - v)))

    for t in range(1, T):
        y0[t] = y0[t-1] + dt * y3[t-1]
        y3[t] = y3[t-1] + dt * (A * a * Sigm(y1[t-1] - y2[t-1]) - 2.0 * a * y3[t-1] - a * a * y0[t-1])
        y1[t] = y1[t-1] + dt * y4[t-1]
        y4[t] = y4[t-1] + dt * (A * a * (p[t-1] + C2 * Sigm(C1 * y0[t-1])) - 2.0 * a * y4[t-1] - a * a * y1[t-1])
        y2[t] = y2[t-1] + dt * y5[t-1]
        y5[t] = y5[t-1] + dt * (B * b * C4 * Sigm(C3 * y0[t-1]) - 2.0 * b * y5[t-1] - b * b * y2[t-1])
        y[t] = y1[t-1] - y2[t-1]
    return y[::2]

t0 = 200
t1 = t0 + int(1/0.001)

plt.subplot(611)
yy = jansenrit(68)
plt.plot(yy[t0:t1])
plt.subplot(612)
yy = jansenrit(128)
plt.plot(yy[t0:t1])
plt.subplot(613)
yy = jansenrit(135)
plt.plot(yy[t0:t1])
plt.subplot(614)
yy = jansenrit(270)
plt.plot(yy[t0:t1])
plt.subplot(615)
yy = jansenrit(675)
plt.plot(yy[t0:t1])
plt.subplot(616)
yy = jansenrit(1350)
plt.plot(yy[t0:t1])
plt.show()