import subprocess
import sys

process = subprocess.Popen(["sbatch", "/nfs/guille/eecs_research/soundbendor/junkinso/river-level-forecasting/grid_search.sh", sys.argv[1]])

process.wait()
