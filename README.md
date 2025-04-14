# Scene Complexity Aware Network for Weakly-Supervised Video Moment Retrieval, ICCV'2023


## Compute Scene Complexity
```
cd scene_complexity_estimation/charades_sta
bash run.sh
```

## Environment
python 3.7.6
CUDA 11.5 - 12.4
```
pip install -r requirements.txt
```

## training
```
python train.py
```

## inference

```
python train.py --eval --resume ./checkpoints/charades/model-best.pt
```


