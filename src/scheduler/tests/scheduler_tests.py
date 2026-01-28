import os
import subprocess
import unittest

class TestScheduler(unittest.TestCase):

    def test_simple(self):
        trace_file = '../traces/simple.trace'
        reference_output = '../traces/simple.output'

        # Skip if required files don't exist
        if not os.path.exists(os.path.join(os.path.dirname(__file__), trace_file)):
            self.skipTest(f"Required file {trace_file} does not exist")
        if not os.path.exists(os.path.join(os.path.dirname(__file__), reference_output)):
            self.skipTest(f"Required file {reference_output} does not exist")

        FNULL = open(os.devnull, 'w')
        subprocess.check_call('python3 run_scheduler_with_trace.py '
                               '-t traces/simple.trace '
                               '--simulate '
                               '--worker_types v100 '
                               '-l /tmp/simple.output',
                               cwd='..', stdout=FNULL,
                               stderr=FNULL, shell=True)
        ref = open('../traces/simple.output', 'r')
        res = open('/tmp/simple.output', 'r')
        reflines = ref.readlines()
        reslines = res.readlines()
        self.assertEqual(len(reflines), len(reslines))
        for i in range(len(reflines)):
            self.assertEqual(reflines[i], reslines[i])

if __name__=='__main__':
    unittest.main()
