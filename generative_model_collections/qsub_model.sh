echo "python main_DCGAN.py --dataset $2 --input_height=108 --crop --gan_type $1 --train --grid $3 --epoch $4 --data-dir $5" > tmp.sh
qsub -cwd -N "$1" -j y -l h_rt=500000 ./tmp.sh
