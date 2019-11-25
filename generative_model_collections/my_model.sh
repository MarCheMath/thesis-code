echo "python ./main.py --gan_type $1 --grid $2 --dataset $3 --epoch $4" > tmp.sh
qsub -cwd -N "$1" -j y -l h_rt=500000 ./tmp.sh

