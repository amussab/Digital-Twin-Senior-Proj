"""Back-compat wrapper: node_rpm.py is now node_config.py (RPM, BEARING, CONFIG?).

    python node_rpm.py 3600        set 3600 RPM
    python node_rpm.py --query     print the node's @CONFIG line
The functions find_port, send, is_ok, pick_reply, build_command are re-exported for the recorder.
"""
import importlib.util
import pathlib
import sys

_spec = importlib.util.spec_from_file_location('node_config', pathlib.Path(__file__).resolve().parent / 'node_config.py')
_nc = importlib.util.module_from_spec(_spec)
sys.modules.setdefault('node_config', _nc)
_spec.loader.exec_module(_nc)

find_port, send, is_ok, pick_reply = _nc.find_port, _nc.send, _nc.is_ok, _nc.pick_reply
build_command, Port, parse_config = _nc.build_command, _nc.Port, _nc.parse_config
BAUD, WAIT_S, TRIES = _nc.BAUD, _nc.WAIT_S, _nc.TRIES

if __name__ == '__main__':
    sys.exit(_nc.main(legacy_positional=True))
