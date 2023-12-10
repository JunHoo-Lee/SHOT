#!/bin/bash

# train base maml
#python main.py --datasource SineLine --ml-algorithm MAML \
#      --first-order --network-architecture FcNet --no-batchnorm \
#      --num-ways 1 --k-shot 5 --inner-lr 0.001 --meta-lr 0.001 \
#      --num-epochs 100 --resume-epoch 0 \
#      --train --logdir outputs/maml

# resume train shot maml
#python main.py --datasource SineLine --ml-algorithm MAML \
#      --first-order --network-architecture FcNet --no-batchnorm \
#      --num-ways 1 --k-shot 5 --inner-lr 0.001 --meta-lr 0.001 \
#      --num-epochs 100 --resume-epoch 0 \
#      --train --logdir outputs/shot_0.1_innernum2 --SHOT

# test base maml
python main.py --datasource SineLine --ml-algorithm MAML \
      --first-order --network-architecture FcNet --no-batchnorm \
      --num-ways 1 --k-shot 5 --inner-lr 0.001 --meta-lr 0.001 \
      --num-epochs 100 --resume-epoch 100 --num-episodes 10000  \
      --test --logdir outputs/maml

# test shot maml
python main.py --datasource SineLine --ml-algorithm MAML \
      --first-order --network-architecture FcNet --no-batchnorm \
      --num-ways 1 --k-shot 5 --inner-lr 0.001 --meta-lr 0.001 \
      --num-epochs 100 --resume-epoch 100 --num-episodes 10000 \
      --test --logdir outputs/shot_0.1_innernum2 --SHOT