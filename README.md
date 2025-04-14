# Scene Complexity Aware Network for Weakly-Supervised Video Moment Retrieval, ICCV'2023

[![arXiv](https://img.shields.io/badge/arXiv-SCANet-b31b1b.svg)](https://arxiv.org/abs/2310.05241)


[//]: # (### Abstract)
Video moment retrieval aims to localize moments in video corresponding to a given language query. To avoid the expensive cost of annotating the temporal moments, weakly-supervised VMR (wsVMR) systems have been studied. For such systems, generating a number of proposals as moment candidates and then selecting the most appropriate proposal has been a popular approach. These proposals are assumed to contain many distinguishable scenes in a video as candidates. However, existing proposals of wsVMR systems do not respect the varying numbers of scenes in each video, where the proposals are heuristically determined irrespective of the video. We argue that the retrieval system should be able to counter the complexities caused by varying numbers of scenes in each video. To this end, we present a novel concept of a retrieval system referred to as Scene Complexity Aware Network (SCANet), which measures the `scene complexity' of multiple scenes in each video and generates adaptive proposals responding to variable complexities of scenes in each video. Experimental results on three retrieval benchmarks (i.e., Charades-STA, ActivityNet, TVR) achieve state-of-the-art performances and demonstrate the effectiveness of incorporating the scene complexity.

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

## Acknowledgement

This code is implemented on top of following contributions: [CPL](https://github.com/minghangz/cpl), [CNM](https://github.com/minghangz/cnm)

We thank the authors for open-sourcing these great projects and papers!

This work was supported by Institute for Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No. 2021 0-01381, Development of Causal AI through Video Understanding and Reinforcement Learning, and Its Applications to Real Environments) and partly supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government(MSIT) (No. 2022R1A2C2012706).

## Citation
Please kindly cite our paper if you use our code, data, models or results:

```bibtex
@inproceedings{yoon2023scanet,
  title={Scanet: Scene complexity aware network for weakly-supervised video moment retrieval},
  author={Yoon, Sunjae and Koo, Gwanhyeong and Kim, Dahyun and Yoo, Chang D},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={13576--13586},
  year={2023}
}
```
