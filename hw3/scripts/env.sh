module purge
module load cmake
module load cuda

username=$(whoami)
if [ "$username" = "u9658571" ]; then
    conda_init
    conda activate hw3
fi
