#!/bin/bash
    python -m train --gym-id humanoid-walk \
                    --total-timesteps 300000000 \
                    --learning-rate 1e-4 \
                    --num-envs 4 \
                    --num-minibatches 128 \
                    --num-steps 2048 \
                    --update-epochs 5 \
                    --max-grad-norm 0.5 \
                    --clip-coef 0.2 \
                    --vf-coef 0.2 \
                    --ent-coef 0.0003 \
                    --gamma 0.995 \
                    --gae-lambda 0.95 \
                    --capture-video False \
                    --from-pretrained False \

#目前表現最好是
    # python -m train --gym-id humanoid-walk \
    #                 --total-timesteps 100000000 \
    #                 --learning-rate 1e-4 \
    #                 --num-envs 4 \
    #                 --num-minibatches 64 \
    #                 --num-steps 4096 \
    #                 --update-epochs 10 \
    #                 --max-grad-norm 0.5 \
    #                 --clip-coef 0.2 \
    #                 --vf-coef 0.2 \
    #                 --ent-coef 0.0003 \
    #                 --gamma 0.99 \
    #                 --gae-lambda 0.95 \
    #                 --capture-video False \
    #                 --from-pretrained True \