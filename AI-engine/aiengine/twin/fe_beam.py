"""1D Euler-Bernoulli FE rotor, lateral plane, steady-state unbalance response.

Hermite cubic beam elements (2 DOF per node: w, theta). Consistent mass. Lumped disk
translational mass. Bearings = linear springs + dashpots at bearing nodes. Unbalance force
m*e*w^2*exp(j*phi) at the disk node. Response at speed w from (K - w^2 M + j w C) q = F.

Convention: x(t) = Re(X exp(j w t)); amplitude = |X| (peak, converted to um),
phase = arg(X) in rad. The Keyphasor/probe sign convention must be fixed by ME/COE at install.
"""
import numpy as np
from .params import RotorParams, DEFAULT


class FeBeam:
    def __init__(self, p: RotorParams = DEFAULT):
        self.p = p
        n = p.n_elem + 1
        self.ndof = 2 * n
        le = p.length / p.n_elem
        A = np.pi * p.shaft_d ** 2 / 4
        I = np.pi * p.shaft_d ** 4 / 64
        EI = p.E * I
        ke = EI / le ** 3 * np.array([
            [12, 6 * le, -12, 6 * le],
            [6 * le, 4 * le ** 2, -6 * le, 2 * le ** 2],
            [-12, -6 * le, 12, -6 * le],
            [6 * le, 2 * le ** 2, -6 * le, 4 * le ** 2]])
        me = p.rho * A * le / 420 * np.array([
            [156, 22 * le, 54, -13 * le],
            [22 * le, 4 * le ** 2, 13 * le, -3 * le ** 2],
            [54, 13 * le, 156, -22 * le],
            [-13 * le, -3 * le ** 2, -22 * le, 4 * le ** 2]])
        K = np.zeros((self.ndof, self.ndof))
        M = np.zeros((self.ndof, self.ndof))
        for e in range(p.n_elem):
            s = slice(2 * e, 2 * e + 4)
            K[s, s] += ke
            M[s, s] += me
        M[2 * p.disk_node, 2 * p.disk_node] += p.disk_mass
        self.K, self.M = K, M
        self.bdof = [2 * i for i in p.bearing_nodes]
        self.pdof = [2 * i for i in p.probe_nodes]
        self.ddof = 2 * p.disk_node

    def response(self, k1, k2, rpm, unb=None, c=None):
        """Complex probe displacements [m] (2,) for unbalance `unb` (complex, kg m)."""
        p = self.p
        if unb is None:
            unb = p.unbalance_nominal * np.exp(1j * p.unbalance_phase_nominal)
        cc = p.c_bearing if c is None else c
        w = 2 * np.pi * rpm / 60.0
        D = (self.K - w ** 2 * self.M).astype(complex)
        for dof, kk, ci in zip(self.bdof, (k1, k2), cc):
            D[dof, dof] += kk + 1j * w * ci
        f = np.zeros(self.ndof, complex)
        f[self.ddof] = unb * w ** 2
        q = np.linalg.solve(D, f)
        return q[self.pdof]

    def unit_response(self, k1, k2, rpm):
        """Probe response per unit unbalance (1 kg m, phase 0)."""
        return self.response(k1, k2, rpm, unb=1.0 + 0j)

    def probes(self, k1, k2, rpm, unb=None):
        """-> (p1_amp_um, p1_phase_rad, p2_amp_um, p2_phase_rad)."""
        x = self.response(k1, k2, rpm, unb) * 1e6
        a = np.abs(x)
        ph = np.angle(x)
        return float(a[0]), float(ph[0]), float(a[1]), float(ph[1])
