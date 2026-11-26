"""Rig-day run-sheet driver for the training recorder (stdlib only, Python 3.8+).

The recorder (training_recorder.py) must be running. For every step of the plan this script:
  1. tells you what to mount and which VFD speed to set, and waits for Enter,
  2. sets the node's RPM through the recorder (/api/rpm, ST-LINK USB) and checks the packets report it,
  3. records one labelled run for the planned minutes, watching that data keeps arriving,
  4. stops the run, checks the window count, backs up training_data and shows coverage,
  5. saves progress, so a restart continues where it stopped.

Each mounted bearing is recorded at every planned speed before the next swap.

  python rig_session.py --make-plan            # write rig_plan.csv (edit bearing IDs if you like)
  python rig_session.py                        # run the plan, step by step
  python rig_session.py --status               # progress + coverage, no recording
Options: --minutes N (override run length), --no-rpm (do not send RPM; only check packets),
         --yes (no prompts, rehearsal), --url http://127.0.0.1:8000
"""
import argparse
import csv
import json
import shutil
import sys
import time
import urllib.request
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLAN = HERE / 'rig_plan.csv'
PROGRESS = HERE / 'rig_plan_progress.json'
DATA = HERE / 'training_data'
BACKUP = HERE.parents[2] / 'rig_data' / f'{date.today().isoformat()}_training_data'
FIELDS = ['step', 'mount', 'rpm', 'label', 'severity', 'bearing_id', 'position', 'is_baseline', 'minutes', 'note']
NAMES = {'healthy': 'Healthy', 'outer_race': 'Outer race', 'inner_race': 'Inner race', 'ball': 'Ball', 'cage': 'Cage'}
SPEEDS = (1800, 3600)
MINUTES = 4.0   # the first 120 s of every run are dropped as warm-up by training


def make_plan():
    """Priority order: baseline + healthy, then every fault at level 2, then levels 1 and 3."""
    mounts = [('healthy', 0, 'H-01', True, 'commissioning baseline')]
    mounts += [('healthy', 0, f'H-0{i}', False, 'healthy remount') for i in (1, 2, 3)]
    for level in (2, 1, 3):
        for label, tag in (('outer_race', 'OR'), ('inner_race', 'IR'), ('ball', 'BA'), ('cage', 'CA')):
            mounts += [(label, level, f'{tag}-L{level}-{k}', False, '') for k in ('a', 'b')]
    rows, step = [], 0
    for m, (label, sev, bid, base, note) in enumerate(mounts, 1):
        for rpm in SPEEDS:
            step += 1
            rows.append(dict(step=step, mount=m, rpm=rpm, label=label, severity=sev, bearing_id=bid,
                             position='b1', is_baseline=int(base), minutes=MINUTES, note=note))
    with PLAN.open('w', newline='') as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f'Wrote {PLAN} ({len(rows)} runs, {len(mounts)} mounts, ~{len(rows) * MINUTES / 60:.1f} h of recording).')
    print('Edit bearing_id/position to match your labelled bearings; delete rows you cannot do.')


def api(url, path, body=None, timeout=20):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url + path, data=data, method='POST' if body is not None else 'GET',
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b'{}')
    except urllib.error.HTTPError as e:
        return {**json.loads(e.read() or b'{}'), 'ok': False, 'http': e.code}


def load_progress():
    try:
        return json.loads(PROGRESS.read_text())
    except (OSError, ValueError):
        return {'done': {}}


def save_progress(p):
    PROGRESS.write_text(json.dumps(p, indent=1))


def coverage(url):
    s = api(url, '/api/status')
    print('  coverage (condition, level, rpm: runs / minutes):')
    for c in s.get('coverage', []):
        print(f"    {c['condition']:<11} L{c['severity']}  {c['rpm']:>4}: {c['runs']} / {c['minutes']:.1f} min")


def wait_for_data(url, rpm, seconds=20):
    """True when packets are fresh, all 4 channels valid and reporting the target rpm."""
    end = time.time() + seconds
    last = None
    while time.time() < end:
        last = (api(url, '/api/status') or {}).get('last_packet')
        if last and last['age_s'] < 3 and last['rpm'] == rpm:
            if last.get('valid_channels') != 15:
                print(f"  WARN: valid_channels={last.get('valid_channels')} (15 = all four)")
            return True
        time.sleep(1)
    print(f'  no fresh packets at {rpm} rpm (last: {last})')
    return False


def ask(prompt, yes):
    if yes:
        print(prompt + ' [auto]')
        return ''
    return input(prompt).strip()


def run(args):
    if not PLAN.exists():
        make_plan()
    plan = list(csv.DictReader(PLAN.open()))
    prog = load_progress()
    url = args.url.rstrip('/')
    s = api(url, '/api/status')
    if s.get('recording'):
        sys.exit('A run is already recording in the recorder page; stop it first.')
    current_rpm, current_mount = None, None
    todo = [r for r in plan if r['step'] not in prog['done']]
    print(f'{len(plan) - len(todo)} of {len(plan)} runs done; {len(todo)} to go.')
    for row in todo:
        rpm, minutes = int(row['rpm']), float(args.minutes or row['minutes'])
        bid = row['bearing_id']
        if row['mount'] != current_mount:
            print(f"\n=== MOUNT {row['mount']}: {NAMES[row['label']]} level {row['severity']}, bearing {bid}, "
                  f"position {row['position']}{' (BASELINE)' if row['is_baseline'] == '1' else ''} {row['note']}")
            a = ask('  Swap the bearing now. Enter = done, a new ID = done with that bearing, s = skip mount, q = quit: ',
                    args.yes)
            if a.lower() == 'q':
                break
            if a.lower() == 's':
                for r in plan:
                    if r['mount'] == row['mount']:
                        prog['done'][r['step']] = {'skipped': True}
                save_progress(prog)
                current_mount = row['mount']
                continue
            if a:
                bid = a
                for r in plan:
                    if r['mount'] == row['mount']:
                        r['bearing_id'] = a
            current_mount, current_rpm = row['mount'], None
        if row['step'] in prog['done']:
            continue
        print(f"\n--- step {row['step']}: {rpm} rpm, {minutes:g} min")
        if rpm != current_rpm:
            ask(f'  Set the VFD to {rpm} rpm, wait until it is steady, then press Enter: ', args.yes)
            if not args.no_rpm:
                r = api(url, '/api/rpm', {'rpm': rpm})
                print(f"  node: {r.get('reply') or r.get('error')}")
                if not r.get('ok'):
                    print('  RPM was not confirmed by the node. Fix it (see RIG_DAY.md troubleshooting) and rerun; '
                          'progress is saved.')
                    break
            if not wait_for_data(url, rpm):
                print('  Stopping here; rerun when data arrives at the right rpm.')
                break
            current_rpm = rpm
        r = api(url, '/api/start', {'label': row['label'], 'severity': int(row['severity']), 'bearing_id': bid,
                                    'position': row['position'], 'is_baseline': row['is_baseline'] == '1',
                                    'note': f"plan step {row['step']} {row['note']}".strip()})
        if not r.get('ok', True) or 'run_id' not in r:
            print(f'  could not start: {r}')
            break
        run_id, t0, gaps = r['run_id'], time.time(), 0
        print(f'  recording {run_id} ...')
        try:
            while time.time() - t0 < minutes * 60:
                time.sleep(5)
                last = (api(url, '/api/status') or {}).get('last_packet') or {}
                if last.get('age_s', 99) > 5:
                    gaps += 1
                    print(f"  WARN: no data for {last.get('age_s')} s")
                left = minutes * 60 - (time.time() - t0)
                print(f'  {max(0, left):5.0f} s left', end='\r')
        except KeyboardInterrupt:
            print('\n  interrupted: stopping the run (it is kept; delete it in the page if unwanted)')
            api(url, '/api/stop', {})
            break
        api(url, '/api/stop', {})
        meta = next((x for x in api(url, '/api/status').get('runs', []) if x['run_id'] == run_id), {})
        expected = minutes * 60 * rpm / 1200
        windows = meta.get('windows', 0)
        flag = 'OK' if windows >= 0.9 * expected else 'SHORT'
        print(f'\n  done: {windows} windows (expected ~{expected:.0f}) {flag}; gaps {gaps}')
        prog['done'][row['step']] = {'run_id': run_id, 'windows': windows, 'flag': flag, 'bearing_id': bid}
        save_progress(prog)
        if Path(args.data_dir).exists():
            shutil.copytree(args.data_dir, BACKUP, dirs_exist_ok=True)
            print(f'  backed up to {BACKUP}')
        coverage(url)
    print('\nSession paused/finished. Rerun the same command to continue.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--make-plan', action='store_true')
    p.add_argument('--status', action='store_true')
    p.add_argument('--minutes', type=float)
    p.add_argument('--no-rpm', action='store_true')
    p.add_argument('--yes', action='store_true')
    p.add_argument('--url', default='http://127.0.0.1:8000')
    p.add_argument('--data-dir', default=str(DATA), help='same --data-dir as the recorder (backup source)')
    a = p.parse_args()
    if a.make_plan:
        make_plan()
    elif a.status:
        pr = load_progress()
        print(f"{len(pr['done'])} steps done")
        coverage(a.url.rstrip('/'))
    else:
        run(a)
