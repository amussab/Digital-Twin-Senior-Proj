"""Build shaft_sensors.c + shaft_demo.c with GCC (strict, sanitizers) and run the checks.

Run in WSL: python3 run_shaft_checks.py
"""
from pathlib import Path
import subprocess

here = Path(__file__).resolve().parent
cm7 = here.parent.parent
build = here / 'build'
build.mkdir(exist_ok=True)
binary = build / 'test_shaft'
sources = [here / 'test_shaft.c', cm7 / 'App/dsp/shaft/shaft_sensors.c', cm7 / 'TestData/demos/shaft_demo.c']
command = ['gcc', '-std=c11', '-O1', '-Wall', '-Wextra', '-Werror', '-pedantic', 
           '-g', '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
           '-I', str(cm7 / 'App/dsp/shaft'), '-I', str(cm7 / 'TestData/demos')]
command += [str(s) for s in sources] + ['-o', str(binary), '-lm']
subprocess.run(command, check=True)
subprocess.run([str(binary)], check=True)
