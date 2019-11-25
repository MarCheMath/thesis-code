echo "python main_DCGAN.py --dataset $2 --input_height=108 --crop --gan_type $1 --train" > tmp.sh
qsub -cwd -N "$1" -j y -l h_rt=100000 ./tmp.sh

