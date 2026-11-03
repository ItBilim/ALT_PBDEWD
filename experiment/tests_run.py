import io,unittest
from .common import *
def main():
 suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_*.py')
 output=io.StringIO();result=unittest.TextTestRunner(stream=output,verbosity=2).run(suite)
 (ROOT/'results').mkdir(exist_ok=True);(ROOT/'results/test_receipt.txt').write_text(output.getvalue(),encoding='utf-8')
 write_json(ROOT/'results/test_summary.json',dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped)))
 print(output.getvalue(),flush=True)
 if not result.wasSuccessful():raise SystemExit(1)
if __name__=='__main__':main()
