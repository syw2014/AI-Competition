#!/bin/bash
cd  `dirname $0`

echo `pwd`

DATA_DIR = "/data/research/data/short_video/icme2019/dataset/"

training_path="$DATA_DIR/final_track2_train.txt.80"     # 交互文件,final_track[1-2]_train.txt
validation_path="$DATA_DIR/final_track2_valid.txt.20"   # 需要自行划分验证集

echo "training path: " $training_path
echo "validation path: " $validation_path


save_model_dir="./output"     # output model directory
echo "save model on: " $save_model_dir

# parameters for model
# baseline for track2 like task: embedding_size=40, optimizer=adam, lr=0.0005, auc=0.865
# baseline for finish task: embedding_size=40, optimizer=adam, lr=0.0001, auc=0.698

batch_size=32
embedding_size=40
echo "batch size: " $batch_size
echo "embedding size: " $embedding_size

optimizer="adam"
lr=0.0005

task="finish" # finish or like
track=2     # track1 or track2
echo "task: " $task
echo "track: " $track

mkdir ${save_model_dir};

python train.py \
  --training_path $training_path \
  --validation_path $validation_path \
  --save_model_dir $save_model_dir \
  --batch_size $batch_size \
  --embedding_size $embedding_size \
  --lr $lr \
  --task $task \
  --track $track \
  --optimizer $optimizer
