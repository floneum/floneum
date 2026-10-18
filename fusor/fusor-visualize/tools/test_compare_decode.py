import json
from pathlib import Path
import tempfile
import unittest
from compare_decode import compare, compare_bracketed


class DecodeComparison(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name) / 'base.log'
        self.candidate = Path(self.temp.name) / 'candidate.log'

    def write(self, path, gaps, output='same text', rounds=2, warmup_bytes=2):
        lines = ['llama_forward_config warmup=1 measured=3 seed=0 repeats=2 chat=false prompt="p"']
        for i in range(rounds):
            values = [900, 900, 900] if i == 0 else gaps
            lines += [f'llama_forward_profile chunks=3 requested=3 warmup_bytes={warmup_bytes}',
                      'llama_forward_gaps_ms ' + json.dumps(values),
                      'llama_forward_output ' + json.dumps(output)]
        path.write_text('\n'.join(lines))

    def test_reports_slow_tail_without_discarding_it(self):
        self.write(self.base, [20, 20, 100])
        self.write(self.candidate, [10, 10, 80])
        result = compare(self.base, self.candidate)
        self.assertEqual(result['elapsed_speedup'], 1.4)
        self.assertEqual(result['p95_speedup'], 1.25)
        self.assertEqual(result['candidate']['max_ms'], 80)
        self.assertEqual(result['candidate']['chunks'], 3)

    def test_refuses_different_outputs_and_boundaries(self):
        self.write(self.base, [20, 20, 100])
        for output, boundary in [('different text', 2), ('same text', 3)]:
            self.write(self.candidate, [10, 10, 50], output=output, warmup_bytes=boundary)
            with self.assertRaisesRegex(ValueError, 'differs'):
                compare(self.base, self.candidate)

    def test_refuses_partial_or_invalid_samples(self):
        self.write(self.base, [20, 20, 100])
        self.write(self.candidate, [10, 10, 50], rounds=1)
        with self.assertRaisesRegex(ValueError, 'rounds'):
            compare(self.base, self.candidate)
        for gaps in [[10, 10], [10, 0, 10], [10, float('nan'), 10]]:
            self.write(self.candidate, gaps)
            with self.assertRaisesRegex(ValueError, 'latency'):
                compare(self.base, self.candidate)

    def test_bracketed_run_uses_the_faster_baseline(self):
        after = Path(self.temp.name) / 'after.log'
        self.write(self.base, [40, 40, 40])
        self.write(self.candidate, [10, 10, 10])
        self.write(after, [20, 20, 20])
        result = compare_bracketed(self.base, self.candidate, after)
        self.assertEqual(result['elapsed_speedup_min'], 2)
        self.assertEqual(result['p95_speedup_min'], 2)
        self.assertEqual(result['baseline_elapsed_ratio'], 2)

    def test_bracketed_run_checks_the_second_baseline_output(self):
        after = Path(self.temp.name) / 'after.log'
        self.write(self.base, [40, 40, 40])
        self.write(self.candidate, [10, 10, 10])
        self.write(after, [20, 20, 20], output='different text')
        with self.assertRaisesRegex(ValueError, 'differs'):
            compare_bracketed(self.base, self.candidate, after)
