username=$(whoami)
if [ "$username" = "u9658571" ]; then
    module purge
    module load cmake
    module load cuda

    conda_init
fi

conda activate hw4
