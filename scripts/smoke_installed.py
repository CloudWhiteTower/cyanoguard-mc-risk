"""Exercise installed console entry points from an unrelated directory."""
from pathlib import Path
import shutil
import subprocess
import tempfile

from check_example import check

root = Path(__file__).resolve().parents[1]
command = shutil.which('mc-risk-predict')
assert command, 'Installed console command is missing'
for entry in ['mc-risk-build', 'mc-risk-prepare-stonelab', 'mc-risk-reproduce']:
    subprocess.run([shutil.which(entry) or entry, '--help'], check=True, capture_output=True)
with tempfile.TemporaryDirectory(prefix='mc-risk-smoke-') as temp:
    output = Path(temp) / 'predictions.csv'
    subprocess.run([command, '--input', str(root / 'examples/synthetic_features.csv'),
                    '--output', str(output)], cwd=temp, check=True)
    check(output, root / 'examples/expected_predictions.csv')
print('Installed commands and frozen package resources passed.')
