"""Run real communication C sources with a simulated HAL using Linux GCC.

No SDK or tool installation. This verifies software logic, not electrical timing.
"""
from pathlib import Path
import argparse
import json
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--firmware', required=True)
args = parser.parse_args()
firmware = Path(args.firmware).resolve()
esp = firmware.parent / 'esp32'  # team repo layout (COE/firmware)
if not esp.exists():
    esp = firmware.parent / 'coe_esp32'  # original local folders
here = Path(__file__).resolve().parent
host = here / 'host'
build = host / 'build'
build.mkdir(exist_ok=True)
binary = build / 'test_communication'
sources = [host / 'test_communication.c', firmware / 'CM7/App/communication/feature_uart.c',
           firmware / 'CM7/TestData/demos/communication_demo.c',
           esp / 'main/feature_packet.c', esp / 'main/transport/latency_packet.c',
           firmware / 'CM7/App/communication/delivery_timing.c']
includes = [host, firmware / 'CM7/App/communication', firmware / 'CM7/TestData/demos',
            firmware / 'CM7/TestData/fixtures', esp / 'main', esp / 'main/transport']
command = ['gcc', '-std=c11', '-Wall', '-Wextra', '-Werror', '-pedantic', '-g',
           '-fsanitize=address,undefined', '-fno-omit-frame-pointer']
for include in includes:
    command += ['-I', str(include)]
command += [str(source) for source in sources] + ['-o', str(binary)]
subprocess.run(command, check=True)
results = {}
for case in ('decoder', 'queue', 'timeout', 'start_failure', 'uart_error', 'init_failure', 'resynchronization', 'delivery', 'uart_timing'):
    result = subprocess.run([str(binary), case], text=True, capture_output=True)
    print(result.stdout, end='')
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f'Failed {case}: exit {result.returncode}')
    results[case] = 'passed'
report = {'compiler': subprocess.check_output(['gcc', '--version'], text=True).splitlines()[0],
          'actual_sources_compiled': [str(p) for p in sources[1:]],
          'sanitizers': ['address', 'undefined_behavior'], 'strict_compile': 'passed',
          'scenarios': results, 'decoder_cases': 22, 'additional_stream_cases': 480,
          'scope': 'C software logic with mocked HAL; no ESP-IDF build or physical UART proof'}
(here / 'host_test_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
