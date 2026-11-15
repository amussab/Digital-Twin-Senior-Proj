"""rig_anchor: [SIMULATION; endpoints resampled from rig windows, 1 bearing/fault, curve shape assumed].

Healthy -> failure trajectories built in the REAL 32-feature node-window space from resampled real rig windows
(healthy H-01 and one faulted bearing per fault class). No real run-to-failure data exists for 4 of 5 fault classes;
the only real failure is one ball-bearing trace (n=1) used to size the late-stage amplification.
Test units are held-out SIMULATED TRAJECTORIES, never "held-out bearings".
"""
from .constants import PROVENANCE  # noqa: F401
