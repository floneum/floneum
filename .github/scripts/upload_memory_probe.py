"""Small synthetic GGUF upload experiment on an isolated macOS CI runner."""
import json
from pathlib import Path
import re
import struct
import subprocess
import time

def memory():
    raw = subprocess.check_output(['vm_stat'], text=True)
    page = int(re.search(r'page size of (\d+) bytes', raw)[1])
    pages = {key.strip().strip('"'): int(value) for key, value in
             re.findall(r'^([^:\n]+):\s+(\d+)\.', raw, re.M)}
    return {'available': page * sum(pages[key] for key in
            ['Pages free', 'Pages inactive', 'Pages speculative']),
            'compressed': page * pages['Pages occupied by compressor'],
            'swapouts': page * pages['Swapouts']}

def fixture(path):
    count, tensor_bytes = 16, 9 * 1024**2
    header = bytearray(struct.pack('<4sIQQ', b'GGUF', 3, count, 0))
    for index in range(count):
        name = f'weight.{index:02}'.encode()
        header += struct.pack('<Q', len(name)) + name
        header += struct.pack('<IQQIQ', 2, 16384, 1024, 2, index*tensor_bytes)
    header += bytes((-len(header)) % 32)
    block = bytes(range(256)) * 4096
    with path.open('wb') as stream:
        stream.write(header)
        for _ in range(count * 9):
            stream.write(block)

fixture_path = Path('artifacts/upload-fixture.gguf')
fixture(fixture_path)
results = []
for mode in ['retain', 'reclaim']:
    phases = []
    with Path(f'artifacts/{mode}.stderr.txt').open('w') as error:
        child = subprocess.Popen(['artifacts/upload-memory-lab', str(fixture_path), mode, '128'],
                                 stdout=subprocess.PIPE, stderr=error, text=True)
        try:
            # The workflow timeout is the outer bound. These stage snapshots
            # compare only this bounded upload's phases, never model throughput.
            for line in child.stdout:
                item = json.loads(line)
                item['memory'] = memory()
                phases.append(item)
                print(json.dumps({'mode': mode, **item}), flush=True)
            assert child.wait(timeout=30) == 0
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=10)
    assert phases[-1]['phase'] == 'host_bytes_verified'
    results.append({'mode': mode, 'phases': phases})
    time.sleep(2)
fixture_path.unlink()
Path('artifacts/memory-results.json').write_text(json.dumps(results, indent=2)+'\n')
