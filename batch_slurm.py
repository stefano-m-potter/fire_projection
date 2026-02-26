import subprocess
import os

# Define the years for LOYO (Leave One Year Out)
years = range(2001, 2023)  # 2001 to 2022

years = [2011, 2009, 2008, 2007]
for year in years:   
    # Set the job name to the year for easy tracking in squeue
    j_name = "-J {}".format(year)
    
    # Pass the year as the first argument ($1) to the shell script
    subprocess.call(["sbatch", j_name, "single_slurm.sh", str(year)])