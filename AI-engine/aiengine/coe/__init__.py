"""COE edge-processing pipeline (Components 1-9) in Python, hardware-ready.

Source of truth: Project architecture/COE/COE_Processing_Pipeline_Component_Details.md.
A real AD7606/STM32 node replaces `sources.SimulatedRigSource`; see README.md in this folder.
"""
from .config import CoeConfig
from .sources import (SampleBlock, SampleSource, SimulatedRigSource, FileReplaySource,
                      HardwareSource, DropInjector, record_blocks)
from .pipeline import CoePipeline, WindowResult

__all__ = ["CoeConfig", "SampleBlock", "SampleSource", "SimulatedRigSource", "FileReplaySource",
           "HardwareSource", "DropInjector", "record_blocks", "CoePipeline", "WindowResult"]
