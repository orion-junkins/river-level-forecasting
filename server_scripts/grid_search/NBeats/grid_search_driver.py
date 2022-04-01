import subprocess
import sys
outfile = sys.argv[1]
process = subprocess.Popen(["sbatch", "/nfs/guille/eecs_research/soundbendor/junkinso/river-level-forecasting/grid_search.sh", outfile])

process.wait()
