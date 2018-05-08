#!/usr/bin/env python
import numpy as np
import sympy as sym
from scipy.optimize import fsolve, minimize
from copy import copy

class Dmf():
    def __init__(self, I_ext=0.0):
        ## Excitatory gating variables

        a_E = 310. # [nC^-1]
        b_E = 125. # [Hz]
        d_E = 0.16 # [s]
        tau_E = 0.1 # tau_NMDA; [s]
        self._tau_E = tau_E
        W_E = 1.0 # excitatory external input weight

        ## Inhibitory gating variables
        a_I = 615. # [nC^-1]
        b_I = 177. # [Hz]
        d_I = 0.087 # [s]
        tau_I = .01 # tau_GABA; [s]
        self._tau_I = tau_I
        W_I = 0.7 # inhibitory external input weight

        ## Other variables from text ##
        gamma = 0.641
        self._gamma = gamma
        I_0 = 0.382 # [nA]; overall effective external input

        I0_E = I_0 * W_E
        I0_I = I_0 * W_I
        self._I0_E = I0_E
        self._I0_I = I0_I

        # FIC curve parameters
        self.I_E_ss = 0.3773805650
        self.I_I_ss = 0.2528951325
        self.S_I_ss = 0.0392184486
        self.S_E_ss = 0.1647572075

        self._set_phi(a_E, a_I, b_E, b_I, d_E, d_I)

        # Equations
        S_E, S_I, S_I_ss, JN_rec, JN_EI = sym.symbols('S_E S_I S_I_ss JN_rec JN_EI')

        I_E = self._I0_E + JN_rec * S_E - ((-1. / S_I_ss) * (self.I_E_ss - self._I0_E - JN_rec * self.S_E_ss - I_ext)) * S_I
        I_I = self._I0_I + JN_EI * S_E - S_I
        r_E = (a_E * I_E - b_E) / (1. - sym.exp(-d_E * (a_E * I_E - b_E)))
        r_I = (a_I * I_I - b_I) / (1. - sym.exp(-d_I * (a_I * I_I - b_I)))

        dS_E = -S_E / tau_E + gamma * (1. - S_E) * r_E
        dS_I = -S_I / tau_I + r_I

        # Jacobian of the system
        self.jacobian = sym.zeros(2, 2)
        self.jacobian[0, 0] = sym.diff(dS_E, S_E)
        self.jacobian[1, 0] = sym.diff(dS_E, S_I)
        self.jacobian[0, 1] = sym.diff(dS_I, S_E)
        self.jacobian[1, 1] = sym.diff(dS_I, S_I)

        T = self.jacobian[0, 0] + self.jacobian[1, 1]
        D = self.jacobian[0, 0] * self.jacobian[1, 1] - self.jacobian[0, 1] * self.jacobian[1, 0]

        self.eval_1 = 0.5 * T + sym.sqrt((T ** 2) / 4. - D)
        self.eval_2 = 0.5 * T - sym.sqrt((T ** 2) / 4. - D)

        import pdb; pdb.set_trace()

    def phase_diagram_compact(self, j_min = 0.001, j_max=5.0, n=100):
        jnei_vect = np.linspace(j_min, j_max, n)
        cp = 0.1*np.ones(len(jnei_vect)+1) # critical points
        hp, hb, lb = np.zeros(len(jnei_vect)), np.zeros(len(jnei_vect)), np.zeros(len(jnei_vect)) # hopf, high-, low-branch
        #critical_points = np.zeros(3)
        is_h, is_p1, is_p2, is_p3, is_j1, is_j2 = True, True, True, True, True, True

        for ji, je in enumerate(jnei_vect):
            self.JN_EI = np.copy(je)

            # Steady state inhibitory current and synaptic gating variable
            I_I_ss, infodict, ier, mesg = fsolve(self._inh_curr_fixed_pts, x0=self.I_I_ss, full_output=True)
            self.I_I_ss = np.copy(I_I_ss)[0]  # update stored steady state value
            r_I_ss = self.phi_I(self.I_I_ss)  # compute new steady state rate
            self.S_I_ss = np.copy(r_I_ss) * self._tau_I  # update stored val.

            # Substitute parameters to eigenvalues
            S_E, S_I, S_I_ss, JN_rec, JN_EI = sym.symbols('S_E S_I S_I_ss JN_rec JN_EI')
            self.eval1 = self.eval_1.subs([(S_E, self.S_E_ss), (S_I, self.S_I_ss), (S_I_ss, self.S_I_ss), (JN_EI, self.JN_EI)])
            self.eval2 = self.eval_2.subs([(S_E, self.S_E_ss), (S_I, self.S_I_ss), (S_I_ss, self.S_I_ss), (JN_EI, self.JN_EI)])

            # Define Critical Point for stability
            unstable = np.abs(fsolve(self._isstable, cp[ji]))[0]
            if self._jacobian(unstable).real < -0.1:
                unstable = cp[ji]
                while self._jacobian(unstable).real < -0.01:
                    unstable += 0.0001
            else:
                while self._jacobian(unstable).real >= 0:
                    unstable -= 0.0001

            cp[ji + 1] = unstable
            #print self._jacobian(unstable).real

            # Define Hopf bifurcation
            if np.abs(self._jacobian(unstable).imag) > 0:
                if is_h:
                    print 'Hopf point'
                    critical_points = np.array([ji, je, unstable])
                    is_h = False
                hp[ji] = minimize(self._iscomplex, 0.5*(hp[ji-1]+unstable), bounds=((0.001, unstable),)).x
            else:
                hp[ji] = unstable

            print [je, unstable, hp[ji]]

        phase_portrait = np.vstack((jnei_vect, cp[1:], hp))
        return phase_portrait, critical_points


    def phase_diagram(self, j_max=5.0, n=100):
        jnei_vect = np.linspace(0.001, j_max, n)
        cp = 0.1*np.ones(len(jnei_vect)+1) # critical points
        hp, hb, lb = np.zeros(len(jnei_vect)), np.zeros(len(jnei_vect)), np.zeros(len(jnei_vect)) # hopf, high-, low-branch
        critical_points = np.zeros((6, 3))
        is_h, is_p1, is_p2, is_p3, is_j1, is_j2 = True, True, True, True, True, True

        for ji, je in enumerate(jnei_vect):
            self.JN_EI = np.copy(je)

            # Steady state inhibitory current and synaptic gating variable
            I_I_ss, infodict, ier, mesg = fsolve(self._inh_curr_fixed_pts, x0=self.I_I_ss, full_output=True)
            self.I_I_ss = np.copy(I_I_ss)[0]  # update stored steady state value
            r_I_ss = self.phi_I(self.I_I_ss)  # compute new steady state rate
            self.S_I_ss = np.copy(r_I_ss) * self._tau_I  # update stored val.

            # Substitute parameters to eigenvalues
            S_E, S_I, S_I_ss, JN_rec, JN_EI = sym.symbols('S_E S_I S_I_ss JN_rec JN_EI')
            self.eval1 = self.eval_1.subs([(S_E, self.S_E_ss), (S_I, self.S_I_ss), (S_I_ss, self.S_I_ss), (JN_EI, self.JN_EI)])
            self.eval2 = self.eval_2.subs([(S_E, self.S_E_ss), (S_I, self.S_I_ss), (S_I_ss, self.S_I_ss), (JN_EI, self.JN_EI)])

            # Define Critical Point for stability
            unstable = np.abs(fsolve(self._isstable, cp[ji]))[0]
            while self._jacobian(unstable).real >= 0:
                unstable -= 0.0001
            cp[ji + 1] = unstable


            import pdb; pdb.set_trace()
            # Define Hopf bifurcation
            if np.abs(self._jacobian(unstable).imag) > 0:
                if is_h:
                    print 'Hopf point'
                    critical_points[0, :] = np.array([ji, je, unstable])
                    is_h = False
                hp[ji] = minimize(self._iscomplex, 0.5*(hp[ji-1]+unstable), bounds=((0.001, unstable),)).x
            else:
                hp[ji] = unstable


            # Define Lower Branch
            if is_j2:
                ep_l, sign_l = self._find_ep(unstable, 0.05)
                if is_j1:
                    if sign_l < 0.0:
                        critical_points[1, :] = np.array([ji, je, ep_l])
                        is_j1 = False
                        print 'Low branch found'
                else:
                    if sign_l > 0.0:
                        critical_points[2, :] = np.array([ji, je, ep_l])
                        is_j2 = False
                        print 'Low branch unstable'
                lb[ji] = ep_l
            else:
                lb[ji] = unstable

            # Define Upper Branch
            ep_h, sign_h = self._find_ep(unstable, 0.9, tol=0.01)
            if is_p1:
                if ep_h == unstable:
                    critical_points[3, :] = np.array([ji, je, ep_h])
                    print 'Upper branch merged'
                    is_p1 = False
            else:
                if is_p2:
                    if sign_h < 0.0:
                        critical_points[4, :] = np.array([ji, je, ep_h])
                        print 'High branch unstable'
                        is_p2 = False
                else:
                    if is_p3:
                        if sign_h > 0.0:
                            critical_points[5, :] = np.array([ji, je, ep_h])
                            print 'High branch stable'
                            is_p3 = False
            hb[ji] = ep_h

            print [je, unstable, hb[ji], lb[ji]]

        phase_portrait = np.vstack((jnei_vect, cp[1:], hp, hb, lb))
        return phase_portrait, critical_points


    def bifurcation(self, J_N_EI, JN_rec_vect):
        self.JN_EI = J_N_EI

        I_I_ss, infodict, ier, mesg = fsolve(self._inh_curr_fixed_pts, x0=self.I_I_ss, full_output=True)
        self.I_I_ss = np.copy(I_I_ss)[0]  # update stored steady state value
        r_I_ss = self.phi_I(self.I_I_ss)  # compute new steady state rate
        self.S_I_ss = np.copy(r_I_ss) * self._tau_I  # update stored val.

        S_E, S_I, S_I_ss, JN_rec, JN_EI = sym.symbols('S_E S_I S_I_ss JN_rec JN_EI')
        self.eval1 = self.eval_1.subs([(S_E, self.S_E_ss), (S_I, self.S_I_ss), (S_I_ss, self.S_I_ss), (JN_EI, self.JN_EI)])
        self.eval2 = self.eval_2.subs([(S_E, self.S_E_ss), (S_I, self.S_I_ss), (S_I_ss, self.S_I_ss), (JN_EI, self.JN_EI)])

        SE_ss = np.zeros((2,len(JN_rec_vect)))
        SI_ss = np.zeros((2,len(JN_rec_vect)))
        SE_unstable = self.S_E_ss * np.ones((2, len(JN_rec_vect)))
        is_cp = True
        is_complex = True
        cp = 0.0
        h = 0.0
        for ji, jn in enumerate(JN_rec_vect):
            self.JN_rec = copy(jn)
            self.J_I = (-1. / self.S_I_ss) * (self.I_E_ss - self._I0_E - self.JN_rec * self.S_E_ss)
            jac = self._jacobian(self.JN_rec)
            if is_cp:
                if jac.real >= 0:
                    cp = copy(self.JN_rec)
                    is_cp = False
            if is_complex:
                if jac.imag > 0:
                    h = copy(self.JN_rec)
                    is_complex = False

            SE_ss[0,ji], SI_ss[0,ji] = self._steady_state(0.9, self.S_I_ss)
            SE_ss[1,ji], SI_ss[1, ji] = self._steady_state(0.001, self.S_I_ss)

            if is_cp:
                if np.abs(SE_ss[0,ji] - self.S_E_ss) > 0.001:
                    stable = True
                    unstable_point = copy(self.S_E_ss)
                    up_unstable = copy(self.S_E_ss)
                    while stable:
                        s_dummy, dummy = self._steady_state(unstable_point, self.S_I_ss)
                        if np.abs(s_dummy - self.S_E_ss) > 0.001:
                            up_unstable = unstable_point
                            stable = False
                        else:
                            unstable_point += 0.001

                    SE_unstable[0, ji] = up_unstable

                if np.abs(SE_ss[1,ji] - self.S_E_ss) > 0.001:
                    stable = True
                    unstable_point = copy(self.S_E_ss)
                    down_unstable = copy(self.S_E_ss)
                    while stable:
                        s_dummy, dummy = self._steady_state(unstable_point, self.S_I_ss)
                        if np.abs(s_dummy - self.S_E_ss) > 0.001:
                            down_unstable = unstable_point
                            stable = False
                        else:
                            unstable_point -= 0.001

                        if unstable_point < 0:
                            stable = False
                            down_unstable = 0.0

                    SE_unstable[1, ji] = down_unstable


        return SE_ss, SI_ss, cp, h, SE_unstable


    def dynamics(self, JN_EI_vect, JN_rec_vect):
        S_EI_ratio, r_EI_ratio, jac_ts, jac_f0 = np.zeros((len(JN_EI_vect), len(JN_rec_vect))), np.zeros((len(JN_EI_vect), len(JN_rec_vect))), \
                                      np.zeros((len(JN_EI_vect), len(JN_rec_vect))), np.zeros((len(JN_EI_vect), len(JN_rec_vect)))
        for ji1, je in enumerate(JN_EI_vect):
            self.JN_EI = je
            I_I_ss, infodict, ier, mesg = fsolve(self._inh_curr_fixed_pts, x0=self.I_I_ss, full_output=True)
            self.I_I_ss = np.copy(I_I_ss)[0]  # update stored steady state value
            r_I_ss = self.phi_I(self.I_I_ss)  # compute new steady state rate
            self.S_I_ss = np.copy(r_I_ss) * self._tau_I  # update stored val.

            S_E, S_I, S_I_ss, JN_rec, JN_EI = sym.symbols('S_E S_I S_I_ss JN_rec JN_EI')
            self.eval1 = self.eval_1.subs([(S_E, self.S_E_ss), (S_I, self.S_I_ss), (S_I_ss, self.S_I_ss), (JN_EI, self.JN_EI)])
            self.eval2 = self.eval_2.subs([(S_E, self.S_E_ss), (S_I, self.S_I_ss), (S_I_ss, self.S_I_ss), (JN_EI, self.JN_EI)])

            for ji, jn in enumerate(JN_rec_vect):
                self.JN_rec = copy(jn)
                self.J_I = (-1. / self.S_I_ss) * (self.I_E_ss - self._I0_E - self.JN_rec * self.S_E_ss)
                S_EI_ratio[ji1, ji] = self.S_E_ss / self.S_I_ss
                r_EI_ratio[ji1, ji] = self.phi_E(self.I_E_ss) / r_I_ss
                jac_f0[ji1, ji] = self._jacobian(self.JN_rec).imag / (2. * np.pi)
                jac_ts[ji1, ji] = -1./self._jacobian(self.JN_rec).real
            print ji1

        return S_EI_ratio, r_EI_ratio, jac_f0, jac_ts



    def nullclines(self, J_N_EI, JN_recc):
        self.JN_EI = J_N_EI

        I_I_ss, infodict, ier, mesg = fsolve(self._inh_curr_fixed_pts, x0=self.I_I_ss, full_output=True)
        self.I_I_ss = np.copy(I_I_ss)[0]  # update stored steady state value
        r_I_ss = self.phi_I(self.I_I_ss)  # compute new steady state rate
        self.S_I_ss = np.copy(r_I_ss) * self._tau_I  # update stored val.

        S_E, S_I, S_I_ss, JN_rec, JN_EI = sym.symbols('S_E S_I S_I_ss JN_rec JN_EI')
        self.eval1 = self.eval_1.subs([(S_E, self.S_E_ss), (S_I, self.S_I_ss), (S_I_ss, self.S_I_ss), (JN_EI, self.JN_EI)])
        self.eval2 = self.eval_2.subs([(S_E, self.S_E_ss), (S_I, self.S_I_ss), (S_I_ss, self.S_I_ss), (JN_EI, self.JN_EI)])

        JN_rec = sym.symbols('JN_rec')
        print complex(self.eval1.subs([(JN_rec, JN_recc)]))

        self.JN_rec = copy(JN_recc)
        self.J_I = (-1. / self.S_I_ss) * (self.I_E_ss - self._I0_E - self.JN_rec * self.S_E_ss)


        #plt.plot(see[1:]-self.S_E_ss); plt.show()
        #import pdb; pdb.set_trace()
        print self.steady_state(0.001, 0.001)[0]
        see1, sii1 = self.integrate(self.S_E_ss+0.01, self.S_I_ss, 10000)
        plt.plot(see1)
        plt.show()
        import pdb; pdb.set_trace()
        see2, sii2 = self.integrate(0.05, self.S_I_ss, 10000)
        import scipy
        #xv, yv = scipy.meshgrid(np.linspace(self.S_E_ss-0.001,self.S_E_ss+0.001,50), np.linspace(self.S_I_ss - 0.1, self.S_I_ss + 0.1,25))
        #xv1, yv1 = scipy.meshgrid(np.linspace(0.0,1.0, 10), np.linspace(0.0,1.0, 10))
        #see2, sii2 = self._func2(xv1, yv1)
        plt.plot(see1, sii1, 'k')
        plt.plot(see2, sii2, 'c')
        #plt.show()
        #import pdb; pdb.set_trace()
        xv, yv = scipy.meshgrid(np.linspace(0.0, 1.0, 200), np.linspace(0.0, 1.0, 200))
        #plt.quiver(xv1, yv1, see2, sii2, alpha=0.5, linewidth=0.5)
        #plt.show()
        see, sii = self._func2(xv, yv)
        plt.contour(xv, yv, see, levels=[0], colors='r')
        #null_se = cs.collections[0].get_paths()[0].vertices
        plt.contour(xv, yv, sii, levels=[0], colors='g')
        #null_si = cs.collections[0].get_paths()[0].vertices
        plt.show()
        import pdb; pdb.set_trace()


    def integrate(self, J_N_EI, JN_rec, t, S_E=None, S_I=None):
        self.JN_EI = J_N_EI

        I_I_ss, infodict, ier, mesg = fsolve(self._inh_curr_fixed_pts, x0=self.I_I_ss, full_output=True)
        self.I_I_ss = np.copy(I_I_ss)[0]  # update stored steady state value
        r_I_ss = self.phi_I(self.I_I_ss)  # compute new steady state rate
        self.S_I_ss = np.copy(r_I_ss) * self._tau_I  # update stored val.

        self.JN_rec = copy(JN_rec)
        self.J_I = (-1. / self.S_I_ss) * (self.I_E_ss - self._I0_E - self.JN_rec * self.S_E_ss)

        S_E = self.S_E_ss if S_E is None else S_E
        S_I = self.S_I_ss if S_I is None else S_I

        dt = 1e-4
        SE = np.zeros(t)
        SI = np.zeros(t)
        for ii in range(t):
            IE = self._I0_E + self.JN_rec * S_E - self.J_I * S_I
            II = self._I0_I + self.JN_EI * S_E - S_I

            r_E = self.phi_E(IE)
            r_I = self.phi_I(II)

            dSE = -(S_E / self._tau_E) + (self._gamma * r_E) * (1. - S_E)
            dSI = -(S_I / self._tau_I) + r_I

            S_E += dt * dSE
            S_I += dt * dSI

            S_E = np.clip(S_E, 0., 1.)
            S_I = np.clip(S_I, 0., 1.)

            SE[ii] = S_E
            SI[ii] = S_I

        return SE, SI


    def _set_phi(self, a_E, a_I, b_E, b_I, d_E, d_I):
        ## Helper function for phi
        IE = sym.symbols('IE')
        II = sym.symbols('II')
        phi_E = (a_E * IE - b_E) / (1. - sym.exp(-d_E * (a_E * IE - b_E)))
        phi_I = ((a_I * II - b_I) / (1. - sym.exp(-d_I * (a_I * II - b_I))))
        self.phi_E = sym.lambdify(IE, phi_E)
        self.phi_I = sym.lambdify(II, phi_I)
        self.phi_E2 = sym.lambdify(IE, phi_E, "numpy")
        self.phi_I2 = sym.lambdify(II, phi_I, "numpy")


    def _inh_curr_fixed_pts(self, I):
        return self._I0_I + self.JN_EI * self.S_E_ss - self._tau_I * self.phi_I(I) - I


    def _find_ep(self, jn_rec, ic, tol=0.001):
        isstable = True
        # error = self._get_ep(jn_rec, ic)
        sign_ep = 0.0
        delta = 0.01
        while isstable:
            error = self._get_ep(jn_rec, ic)
            if np.abs(error) < tol:
                isstable = False
            else:
                jn_rec -= delta
                if error > 0:
                    sign_ep = 1.0
                else:
                    sign_ep = -1.0
        return jn_rec, sign_ep


    def _get_ep(self, jn_rec, ic):
        self.JN_rec = copy(jn_rec)
        self.J_I = (-1. / self.S_I_ss) * (self.I_E_ss - self._I0_E - jn_rec * self.S_E_ss)
        ep = self._steady_state(ic, self.S_I_ss)[0]
        return ep - self.S_E_ss


    def _jacobian(self, jn):
        JN_rec = sym.symbols('JN_rec')
        return complex(self.eval1.subs([(JN_rec, np.abs(jn))]))


    def _isstable(self, jn):
        return self._jacobian(jn).real


    def _iscomplex(self, jn):
        jac = self._jacobian(jn)
        if np.abs(np.angle(jac) - np.pi) < 0.0000001:
            return np.abs(jac.real)
        else:
            return 1.0 + np.cos(np.angle(jac))


    def _func2(self, x, y):
        # y = Chebfun.identity()
        # -(x / self._tau_E) + (self._gamma * self.phi_E2(self._I0_E + self.JN_rec * x - self.J_I * y)) * (1. - x)
        see = -(x / self._tau_E) + (self._gamma * self.phi_E2(self._I0_E + self.JN_rec * x - self.J_I * y)) * (1. - x)
        sii = -(y / self._tau_I) + self.phi_I2(self._I0_I + self.JN_EI * x - y)
        return see, sii


    def _steady_state(self, S_E, S_I):
        dt = 1e-4
        dSE, dSI = 1., 1.
        S_E = np.copy(S_E)
        S_I = np.copy(S_I)
        while (np.abs(dSE) > 0.001):
            S_E, S_I, dSE = self._step(dt, S_E, S_I)

        for t in range(200):
            S_E, S_I, dSE = self._step(dt, S_E, S_I)

        dSE, dSI = 1., 1.
        while (np.abs(dSE) > 0.00001):
            S_E, S_I, dSE = self._step(dt, S_E, S_I)

        for t in range(200):
            S_E, S_I, dSE = self._step(dt, S_E, S_I)

        dSE, dSI = 1., 1.
        while (np.abs(dSE) > 0.00001):
            S_E, S_I, dSE = self._step(dt, S_E, S_I)

        return S_E, S_I


    def _step(self, dt, S_E, S_I):
        IE = self._I0_E + self.JN_rec * S_E - self.J_I * S_I
        II = self._I0_I + self.JN_EI * S_E - S_I

        r_E = self.phi_E(IE)
        r_I = self.phi_I(II)

        dSE = -(S_E / self._tau_E) + (self._gamma * r_E) * (1. - S_E)
        dSI = -(S_I / self._tau_I) + r_I

        # print np.abs(dSE)# + np.abs(dSI)
        S_E += dt * dSE
        S_I += dt * dSI

        S_E = np.clip(S_E, 0., 1.)
        S_I = np.clip(S_I, 0., 1.)

        return S_E, S_I, dSE