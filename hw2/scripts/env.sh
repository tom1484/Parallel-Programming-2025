user=$(whoami)
if [ "$user" == "u9658571" ]; then
    module purge
    module load miniconda3
    module load gcc/13
    module load openmpi
fi

eval "$(conda shell.bash hook)"
conda activate hw2
export UCX_NET_DEVICES=mlx5_0:1
