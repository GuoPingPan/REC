"""Private deployment: readable shared PAK cache and unprivileged official server."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pwd
import shutil

COMMON = Path('/mnt/data/users/kindel/uav/experiments/aeroact/20261018-012530-nvila8b-aerialvln-s-official')
CACHE = Path('/dev/shm') / ('aeroact_policy_paks_' + hashlib.sha256(str(COMMON).encode()).hexdigest()[:12])


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(16 * 1024**2), b''):
            h.update(data)
    return h.hexdigest()


def atomic(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)


def stage_assets(work, scenes):
    assert os.geteuid() == 0
    owner = json.loads((work / 'owner.json').read_text())
    assert owner['kind'] == 'official_aeroact_policy_evaluation'
    user = pwd.getpwnam('uavrun')
    CACHE.mkdir(exist_ok=True)
    assert not CACHE.is_symlink()
    CACHE.chmod(0o755)
    with (CACHE / 'prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker = CACHE / 'owner.json'
        expected = {'kind': 'aeroact_readonly_native_pak_cache', 'experiment': str(COMMON)}
        if marker.exists():
            assert json.loads(marker.read_text()) == expected
        else:
            atomic(marker, expected)
        for scene in scenes:
            src = Path('/mnt/data/users/kindel/uav/envs/aerialvln') / f'env_{scene}'
            receipt_path = src / 'download_receipt.json'
            receipt = json.loads(receipt_path.read_text())
            pack_dir = CACHE / f'env_{scene}'
            pack_dir.mkdir(exist_ok=True)
            proof_path = pack_dir / 'cache_manifest.json'
            proof = json.loads(proof_path.read_text()) if proof_path.exists() else None
            receipt_sha = digest(receipt_path)
            if proof:
                assert proof['source_receipt_sha256'] == receipt_sha
            hashes = {}
            for item in receipt['source']['files']:
                rel = Path(*Path(item['name']).parts[2:])
                assert not rel.is_absolute() and '..' not in rel.parts
                target = work / 'ENVs' / f'env_{scene}' / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                source = src / rel
                assert source.stat().st_size == item['bytes']
                if source.suffix != '.pak':
                    assert not target.exists()
                    shutil.copy2(source, target)
                    continue
                cached = pack_dir / rel
                cached.parent.mkdir(parents=True, exist_ok=True)
                if proof:
                    record = proof['files'][str(rel)]
                    stat = cached.stat()
                    assert not cached.is_symlink() and stat.st_size == item['bytes']
                    assert stat.st_mtime_ns == record['mtime_ns']
                    hashes[str(rel)] = record
                else:
                    assert not cached.exists(), 'Inspect incomplete owned cache before retrying'
                    assert shutil.disk_usage(CACHE).free > item['bytes'] + 2_000_000_000
                    available = next(int(l.split()[1]) * 1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
                    assert available > item['bytes'] + 32_000_000_000, 'Insufficient available RAM for private cache'
                    partial = cached.with_suffix('.pak.partial')
                    assert not partial.exists()
                    h = hashlib.sha256()
                    with source.open('rb') as incoming, partial.open('xb') as outgoing:
                        for block in iter(lambda: incoming.read(16 * 1024**2), b''):
                            h.update(block)
                            outgoing.write(block)
                    assert partial.stat().st_size == item['bytes']
                    assert digest(partial) == h.hexdigest(), 'Private PAK byte equality check failed'
                    partial.chmod(0o444)
                    partial.replace(cached)
                    hashes[str(rel)] = {'size': item['bytes'], 'sha256': h.hexdigest(), 'mtime_ns': cached.stat().st_mtime_ns}
                target.symlink_to(cached)
            if proof is None:
                atomic(proof_path, {'scene': scene, 'source_receipt_sha256': receipt_sha, 'files': hashes,
                                    'all_cached_pak_bytes_equal_source_verified': True})
    home = work / 'runtime-home'
    (home / 'Documents/AirSim').mkdir(parents=True)
    (home / 'xdg').mkdir()
    atomic(home / 'nvidia_icd.json', {'file_format_version': '1.0.0', 'ICD': {'library_path': 'libGLX_nvidia.so.0', 'api_version': '1.3.242'}})
    atomic(home / 'nvidia_egl.json', {'file_format_version': '1.0.0', 'ICD': {'library_path': 'libEGL_nvidia.so.0'}})
    for scene in scenes:
        directory = work / 'ENVs' / f'env_{scene}' / 'LinuxNoEditor'
        binary = directory / 'AirVLN/Binaries/Linux/AirVLN-Linux-Shipping'
        binary.chmod(0o755)
        (directory / 'AirVLN.sh').write_text('#!/bin/bash\nexec /usr/bin/python3 ' + str(work / 'policy_eval_unprivileged_launcher_v1.py') +
                                           ' --work ' + str(work) + ' --scene ' + str(scene) + ' -- "$@"\n')
        (directory / 'AirVLN.sh').chmod(0o755)
    tools = work / 'native-tools'
    tools.mkdir()
    shutil.copy2(COMMON / 'runtime/native_eval_tools-v1/bin/netstat', tools / 'netstat')
    (tools / 'netstat').chmod(0o755)
    (work / 'code/AirVLN/airsim_plugin/settings').mkdir(parents=True, exist_ok=True)
    # Follow no symlinks: neither shared PAKs nor checkpoint/data sources change ownership.
    for root, dirs, files in os.walk(work, followlinks=False):
        os.chown(root, user.pw_uid, user.pw_gid)
        os.chmod(root, 0o755)
        for name in files:
            path = Path(root) / name
            if not path.is_symlink():
                os.chown(path, user.pw_uid, user.pw_gid)
    (home / 'xdg').chmod(0o700)
    return home, tools
