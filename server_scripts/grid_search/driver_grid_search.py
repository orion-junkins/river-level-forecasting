import subprocess

process = subprocess.Popen(["sbatch", "/nfs/guille/eecs_research/soundbendor/junkinso/river-level-forecasting/grid_search.sh"])

process.wait()
