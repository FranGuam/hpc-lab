source /home/spack/spack/share/spack/setup-env.sh
spack load ucx

srun -n 8 ./main $1
