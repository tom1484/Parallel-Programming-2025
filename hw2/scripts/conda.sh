user=$(whoami)
if [ "$user" == "u9658571" ]; then
    module load miniconda3
fi

conda create -n hw2 python=3.12 -y
conda activate hw2

pip install --upgrade pip
pip install scikit-image
pip install opencv-python

