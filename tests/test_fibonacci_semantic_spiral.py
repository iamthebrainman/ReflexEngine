import unittest
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ReflexEngine_SelfPublished_v1 import MemoryCrystal, fibonacci_semantic_spiral, embed, now_utc

class TestFibonacciSemanticSpiral(unittest.TestCase):
    def setUp(self):
        self.mem = MemoryCrystal()
        ts_base = 1000.0
        for i in range(10):
            self.mem.add(f"node {i}", "role", "tid", ts=ts_base + i)

    def test_includes_zero_index(self):
        # Build expected result based on fibonacci starting with 0
        v = embed("query")
        ts = now_utc()
        with self.mem.lock:
            scored = sorted(
                self.mem.nodes.values(),
                key=lambda n: n.spiral_score(v, ts, self.mem.nodes),
                reverse=True,
            )
        fib = [0, 1]
        while len(fib) < 5:
            fib.append(fib[-1] + fib[-2])
        indices = sorted(set(fib[:5]))
        expected = [scored[i] for i in indices if i < len(scored)]

        result = self.mem.fibonacci_semantic_spiral("query", max_k=5)
        self.assertEqual(result, expected)
        if scored:
            self.assertIn(scored[0], result)

if __name__ == "__main__":
    unittest.main()
