#!/usr/bin/python3
"""Reject silent performance fallback on this four-V100 deployment."""
import argparse
import json
import pathlib
import re
import sys
import time
import urllib.request

EXPECTED_EXPERTS = 24576
EXPECTED_HUGETLB_KIB = 23983 * 2048

def configured_context(cfg):
    args = cfg['args']
    if args.count('--max-context') != 1:
        raise RuntimeError('exactly one configured context budget is required')
    value = int(args[args.index('--max-context') + 1])
    if value not in (131072, 262144):
        raise RuntimeError('unsupported configured context budget')
    return value

def validate_loaded_engine(status, segment, context, length_policy=False, mmq_direct=False, expected_context=131072):
    match = re.search(r'^HugetlbPages:\s+(\d+)', status, re.M)
    if not match or int(match[1]) < EXPECTED_HUGETLB_KIB:
        raise RuntimeError('expert arena is not backed by the required explicit huge pages')
    if f'token graph hit path: {EXPECTED_EXPERTS} resident experts' not in segment:
        raise RuntimeError('not all 24576 experts are resident in VRAM')
    # pw2 owns two verifiers on each of the four stages; require all eight.
    if segment.count('100% VRAM resident: zero-doorbell graph') < 8:
        raise RuntimeError('four full-resident stages (eight verifier graphs) are not ready')
    if '--pipeline-windows 2: two verifiers per stage, 4 stages overlap' not in segment:
        raise RuntimeError('single-stream window pipeline is not enabled')
    if expected_context not in (131072, 262144):
        raise RuntimeError('unsupported configured context budget')
    if context != expected_context:
        raise RuntimeError(f'unexpected context budget: expected {expected_context}')
    if length_policy and 'measured length chunks on:' not in segment:
        raise RuntimeError('configured length chunk policy is not active in this engine')
    if mmq_direct and '48 of 48 layers' not in segment:
        raise RuntimeError('configured direct MMQ does not cover the expected 48 layers')

def check(config, log=None, url=None):
    cfg = json.loads(pathlib.Path(config).read_text())
    log = pathlib.Path(log or cfg['log'])
    pid = None
    for entry in pathlib.Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / 'cmdline').read_bytes().split(b'\0')
            if args[0].decode() == cfg['exe'] and b'--serve' in args:
                if pid is not None:
                    raise RuntimeError('multiple production engines')
                pid = int(entry.name)
        except (OSError, IndexError, UnicodeError):
            pass
    if pid is None:
        raise RuntimeError('engine process has not started')
    status = pathlib.Path(f'/proc/{pid}/status').read_text()
    segment = log.read_text(errors='replace').rsplit(' engine started:', 1)[-1]
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request((url or f"http://127.0.0.1:{cfg.get('port', 8080)}").rstrip('/') + '/metrics', headers={
        'Authorization': 'Bearer ' + cfg['api_key']})
    with opener.open(request, timeout=5) as response:
        metrics = json.load(response)
    env = cfg.get('env', {})
    validate_loaded_engine(status, segment, metrics['engine']['max_context'],
                           env.get('STRATA_PREFILL_LENGTH_CHUNKS') == '1',
                           env.get('STRATA_MMQ_DIRECT') == '1', expected_context=configured_context(cfg))
    return pid

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=pathlib.Path)
    parser.add_argument('--log', type=pathlib.Path)
    parser.add_argument('--url')
    parser.add_argument('--wait', type=float, default=0, metavar='SECONDS')
    options = parser.parse_args()
    deadline = time.monotonic() + max(options.wait, 0)
    while True:
        try:
            pid = check(options.config, options.log, options.url)
            print(f'Strata performance startup check passed: PID {pid}, context matches explicit config, 24576 experts, four resident stages, hugetlb, pw2', flush=True)
            break
        except Exception as error:
            if time.monotonic() >= deadline:
                print('Strata performance startup check FAILED: ' + str(error), file=sys.stderr, flush=True)
                sys.exit(1)
            time.sleep(2)
