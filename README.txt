Experiment pipeline
===================

1. Compute datasets (data > diagram > persistence images), result: data/dataset_{name}.pt
   $ nohup python -u setup.py > dataset_preparation_{name}.log &

2. Run experiment, result: images_{name}.log
   $ nohup python -u image_experiments.py > results_images_{name}.log &
   
   W2          - 
   WB          - 
   PIE         - 
   Time        - 
   LR          - 
   RF          - 
   Acc, PDrea  - ?
   Acc, PDpre  - ?
   
   1) End of file, LR: 0.927 RF: 0.949 pointnet_topn_Orbit5k
   2) Full model runtime pointnet_topn_Orbit5k 0.0005548889636993408
   3) `bottleneck full model test` > W2, Wb
   4) pretrained_models/model_classifier{real|pred}_{acc} -> ???, ???
       