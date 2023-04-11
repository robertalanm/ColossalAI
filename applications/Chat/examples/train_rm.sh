python train_reward_model.py --pretrain '/home/ubuntu/hf_models/7b' \
                             --model 'llama' \
                             --strategy naive \
                             --loss_fn 'log_exp'\
                             --save_path 'bpt-rm-7b.pt' \
                             --test True
