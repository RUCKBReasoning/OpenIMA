data_device=0
drop_rate=0.5
cd ../
mkdir log/
cd log/
mkdir ogbn-arxiv/
cd ../

for dataset in ogbn-arxiv
do
    for seed in 2406525885 1660347731 3164031153 1454191016 1583215992 765984986 258270452 3808600642 292690791 2492579272
    do
        for alpha in 25
        do
            for learning_rate in 0.01
            do
                for scale in 1
                do
                    for tau in 0.7
                    do
                        python train_ours_large.py --tau $tau --dataset $dataset --pretrain_epochs 20 --nodes_per_class 500 --hidden_dim 128 --batch_size 4096 --filter 1 --alpha $alpha --scale $scale --learning_rate $learning_rate --seed $seed --feat_drop_rate $drop_rate --attn_drop_rate $drop_rate --device $data_device
                    done
                done
            done
        done
    done
done

cd run/
python print_result.py ogbn-arxiv 0.5 10