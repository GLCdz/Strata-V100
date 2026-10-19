import copy
import importlib.util
import json
import pathlib
import stat
import sys
import tempfile
import unittest
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import make_config
import bench

class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.base={'exe':'/opt/example/strata','args':['--native','/models/example.gguf','--batch','4','--batch-groups','4','--max-context','131072','--kv-resident','32768','--layer-split','auto'],
                   'tokenizer':'/models/tokenizer','vision':{'exe':'/opt/example/vision'},'api_key':'example-test-key',
                   'env':{'STRATA_PIPED_CONCURRENT':'1'},'parallel':4}

    def test_keep_model_paths_sampling_and_input_unchanged(self):
        previous=copy.deepcopy(self.base)
        self.base['presence_penalty']=1.5
        c=make_config.configure(self.base,['0','1','2','3'])
        self.assertIn('/models/example.gguf',c['args'])
        self.assertEqual(c['tokenizer'],self.base['tokenizer'])
        self.assertEqual(c['vision'],self.base['vision'])
        self.assertEqual(c['presence_penalty'],1.5)
        self.assertEqual(c['api_key'],'example-test-key')
        self.assertEqual({k:v for k,v in self.base.items() if k!='presence_penalty'},previous)

    def test_no_duplicate_context_or_batch_flags(self):
        c=make_config.configure(self.base,['0','1','2','3'])
        self.assertEqual(c['args'].count('--max-context'),1)
        self.assertEqual(c['args'][c['args'].index('--kv-resident')+1],'262144')
        self.assertNotIn('--batch',c['args']);self.assertNotIn('--batch-groups',c['args'])
        self.assertNotIn('parallel',c);self.assertNotIn('STRATA_PIPED_CONCURRENT',c['env'])
        self.assertNotIn('STRATA_AUX_CPUS',c['env'])

    def test_uuid_order_and_explicit_aux_cpu_choice(self):
        c=make_config.configure(self.base,['GPU-a','GPU-b','GPU-c','GPU-d'],'25-47')
        self.assertEqual(c['env']['CUDA_VISIBLE_DEVICES'],'GPU-a,GPU-b,GPU-c,GPU-d')
        self.assertEqual(c['env']['STRATA_AUX_CPUS'],'25-47')

    def test_invalid_or_duplicate_gpu_selection_rejected(self):
        for ids in (['0','1'],['0','1','2','2'],['0','1','2','bad']):
            with self.assertRaises(ValueError):make_config.configure(self.base,ids)

    def test_generated_key_and_loopback_binding(self):
        self.base['api_key']='';self.base['host']='0.0.0.0'
        c=make_config.configure(self.base,['0','1','2','3'])
        self.assertGreaterEqual(len(c['api_key']),32)
        self.assertEqual(c['host'],'127.0.0.1')

    def test_private_file_and_never_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=pathlib.Path(d)/'config.json'
            make_config.write_new(p,self.base)
            self.assertEqual(stat.S_IMODE(p.stat().st_mode),0o600)
            with self.assertRaises(FileExistsError):make_config.write_new(p,{})
            self.assertEqual(json.loads(p.read_text()),self.base)

class AssociationTest(unittest.TestCase):
    def setUp(self):
        self.usage={'prompt_tokens':12,'completion_tokens':4}
        self.row={'prompt_tokens':12,'output_tokens':4,'time':100}

    def test_new_row_must_match_this_request(self):
        row=bench.associate({'requests':[{'time':1}]},{'requests':[{'time':1},self.row]},self.usage,99)
        self.assertEqual(row,self.row)

    def test_external_request_cannot_be_silently_used(self):
        with self.assertRaises(RuntimeError):
            bench.associate({'requests':[]},{'requests':[self.row,{'time':101}]},self.usage,99)

    def test_stale_and_wrong_token_records_rejected(self):
        for usage,start in ((self.usage,101),({'prompt_tokens':13,'completion_tokens':4},99)):
            with self.assertRaises(RuntimeError):bench.associate({'requests':[]},{'requests':[self.row]},usage,start)

if __name__=='__main__':unittest.main()
