import json
import pathlib
import re
import unittest
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import check_startup as G

STATUS = "HugetlbPages: 49117184 kB\n"
LOG = ("token graph hit path: 24576 resident experts\n"
       + "100% VRAM resident: zero-doorbell graph\n" * 8
       + "--pipeline-windows 2: two verifiers per stage, 4 stages overlap\n")

class GuardTest(unittest.TestCase):
    def test_healthy_128(self): G.validate_loaded_engine(STATUS, LOG, 131072)
    def test_healthy_256(self): G.validate_loaded_engine(STATUS, LOG, 262144, expected_context=262144)
    def test_configuration_is_explicit(self): self.assertEqual(G.configured_context({'args':['--max-context','262144']}),262144)
    def test_unsupported_configuration(self):
        for args in ([], ['--max-context','65536'], ['--max-context','131072','--max-context','262144']):
            with self.assertRaises(RuntimeError): G.configured_context({'args':args})
    def test_runtime_context_mismatch(self):
        with self.assertRaisesRegex(RuntimeError,'context budget'): G.validate_loaded_engine(STATUS,LOG,131072,expected_context=262144)
    def test_no_actual_hugepage_allocation(self):
        with self.assertRaisesRegex(RuntimeError,'huge pages'): G.validate_loaded_engine(re.sub(r'HugetlbPages:\s+\d+','HugetlbPages: 0',STATUS),LOG,131072)
    def test_missing_experts(self):
        with self.assertRaisesRegex(RuntimeError,'24576'): G.validate_loaded_engine(STATUS,LOG.replace('24576 resident experts','24269 resident experts'),131072)
    def test_missing_graph(self):
        with self.assertRaisesRegex(RuntimeError,'four full-resident'): G.validate_loaded_engine(STATUS,LOG.replace('100% VRAM resident: zero-doorbell graph','',1),131072)
    def test_missing_pipeline(self):
        with self.assertRaisesRegex(RuntimeError,'pipeline'): G.validate_loaded_engine(STATUS,LOG.replace('--pipeline-windows 2: two verifiers per stage, 4 stages overlap','off'),131072)
    def test_ignored_length_policy(self):
        with self.assertRaisesRegex(RuntimeError,'length chunk policy'): G.validate_loaded_engine(STATUS,LOG,131072,length_policy=True)
    def test_ignored_direct_mmq(self):
        with self.assertRaisesRegex(RuntimeError,'direct MMQ'): G.validate_loaded_engine(STATUS,LOG,131072,mmq_direct=True)

if __name__ == '__main__': unittest.main()
