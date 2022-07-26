import sys
import subprocess
import time
import datetime
from datetime import date, timedelta
import numpy as np


def main(jobId):
    process = subprocess.Popen(
        [
            'bash', '-c',
            "sacct --format=CPUTime,MaxRSS,State,JobName --job {} -P | grep batch | grep COMPLETED".format(
                jobId)
        ],
        stdout=subprocess.PIPE
    )
    output, error = process.communicate()
    output = output.decode().split('\n')[:-1]
    output = [o.split('|') for o in output]

    def parse_time(o):
        try:
            return (
                datetime.datetime.strptime(o[0], '%H:%M:%S')
                - datetime.datetime(1900, 1, 1)).total_seconds()
        except:
            return (
                datetime.datetime.strptime(o[0], '%d-%H:%M:%S')
                - datetime.datetime(1900, 1, 1) + timedelta(days=1)
            ).total_seconds()
    times = list(map(parse_time, output))
    times = np.array(times)
    time_mean = datetime.timedelta(seconds=np.mean(times))
    time_max = datetime.timedelta(seconds=np.max(times))
    time_median = datetime.timedelta(seconds=np.median(times))
    print("Time: max:{}, mean:{}, median:{}".format(
        time_max, time_mean, time_median))

    mems = [o[1] for o in output]
    mems = np.array([float(m[:-1]) for m in mems if len(m) > 1])
    mem_max = np.max(mems)
    mem_mean = np.mean(mems)
    mem_median = np.median(mems)
    print("Memory: max:{}, mean:{}, median:{}".format(
        mem_max, mem_mean, mem_median))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 get_slurm_stats.py jobId")
    else:
        main(sys.argv[1])
