
for (( i=0; i<2; i=i+1 )); do
  command python3 examples/generate_counter_examples.py --randomseed $i
  command python3 train_scripts/train_gym.py --experiment_name $i --minWith target --tMax 0. --tMin -1.1 --num_src_samples 6000 --pretrain --pretrain_iters 150 --num_epochs 601 --counter_end 1
done
