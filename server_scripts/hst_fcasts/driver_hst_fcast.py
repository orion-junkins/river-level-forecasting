import subprocess

processes = []

horizons = [12, 24, 48, 72, 96]
for horizon in horizons:
    process = subprocess.Popen(["sbatch", "/nfs/guille/eecs_research/soundbendor/junkinso/river-level-forecasting/generate_historical.sh", str(horizon), "1"])
    processes.append(process)

for process in processes:
    process.wait()