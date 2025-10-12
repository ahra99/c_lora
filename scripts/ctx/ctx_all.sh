 
modelwrapper=c_lora 
model=meta-llama/Llama-2-7b-hf
eps=0.05
beta=0.2 
gamma=8
momentum=0
bklr=1
subset_size=0.8
wd=0
 
for dataset in winogrande_s obqa boolq #ARC-Challenge ARC-Easy winogrande_m winogrande_s obqa boolq 
do
for seed in 9 
do
case $dataset in 
ARC-Challenge|ARC-Easy|obqa) kllr=0.01; kllrstd=0.01; max_tr_stp=1500;;
winogrande_s|boolq) kllr=0.015; kllrstd=0.01; max_tr_stp=1500;;
winogrande_m) kllr=0.02; kllrstd=0.01; max_tr_stp=2000;;
esac
    name=$modelwrapper-$dataset-seed$seed-kllr$kllr-kllrstd$kllrstd-trnum$max_tr_stp
    CUDA_VISIBLE_DEVICES=0 python run/main.py --dataset-type mcdataset --dataset $dataset \
    --model-type causallm --model $model --modelwrapper $modelwrapper \
    --lr 1e-4 --batch-size 4 \
    --opt adamw --warmup-ratio 0.06 \
    --max-seq-len 300 \
    --seed $seed \
    --evaluate \
    --wandb-name $name  --wandb-project 'project-name' --wandb-entity='wandb-entity' \
    --apply-classhead-lora --lora-r 8 --lora-alpha 16 --lora-dropout 0 \
    --log-path $name \
    --max-train-steps $max_tr_stp \
    --eval-per-steps 100 \
    --bayes-kl-reweighting $bklr\
    --subset-size $subset_size\
    --bayes-eps $eps --bayes-beta $beta --bayes-gamma $gamma --bayes-kllr $kllr --bayes-kllr-std $kllrstd --bayes-momentum $momentum --bayes-opt2-wd $wd \
    --bayes-train-n-samples 1 --bayes-eval-n-samples 1 
done
done
