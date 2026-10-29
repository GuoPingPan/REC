"""Run as the same unprivileged user as the official simulator manager."""
import argparse
import json
from pathlib import Path
import os
import shutil
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--work', type=Path, required=True)
p.add_argument('--scene', type=int, required=True)
p.add_argument('native_args', nargs=argparse.REMAINDER)
a = p.parse_args()
assert os.geteuid() != 0
W = a.work.resolve()
assert json.loads((W / 'owner.json').read_text())['kind'] == 'official_aeroact_policy_evaluation'
args = a.native_args[1:] if a.native_args[:1] == ['--'] else a.native_args
settings = Path(args[args.index('--settings') + 1]).resolve()
assert settings.is_relative_to(W)
home = W / 'runtime-home'
shutil.copy2(settings, home / 'Documents/AirSim/settings.json')
directory = W / 'ENVs' / f'env_{a.scene}' / 'LinuxNoEditor'
binary = directory / 'AirVLN/Binaries/Linux/AirVLN-Linux-Shipping'
cmd = [str(binary), 'AirVLN', *args, f'-UserDir={home}', '-unattended', '-NoSplash']
logs = W / 'native-logs'
logs.mkdir(exist_ok=True)
tag = f'scene-{a.scene}-{time.time_ns()}'
with (logs / (tag + '.log')).open('w') as log:
    proc = subprocess.Popen(cmd, cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors='replace', bufsize=1)
    (logs / (tag + '.process.json')).write_text(json.dumps({'pid': proc.pid, 'uid': os.geteuid(), 'scene': a.scene, 'command': cmd}, indent=2))
    forwarded = False
    for line in proc.stdout:
        log.write(line)
        log.flush()
        if not forwarded and 'Drone_' in line:
            print(line, end='', flush=True)
            forwarded = True
    result = proc.wait()
(logs / (tag + '.exit.json')).write_text(json.dumps({'exit_code': result, 'time': time.time()}))
