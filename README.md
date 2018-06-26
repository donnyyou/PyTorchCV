### Source code for Deep Learning Based CV Problems(Pytorch)
```
@misc{CV2018,
  author =       {Donny You (youansheng@gmail.com)},
  howpublished = {\url{https://github.com/CVBox/PytorchCV}},
  year =         {2018}
}
```

This repository provides source code for some deep learning based cv problems. We'll do our best to keep this repository up to date.  If you do find a problem about this repository, please raise it as an issue. We will fix it immediately.

#### Examples (trained with PyTorchCV)

<div align="center">

<img src="val/examples/pose/coco/000000319721_vis.png" width="500px"/>

<p> Example output of <b>VGG19-OpenPose</b></p>

<img src="val/examples/pose/coco/000000475191_vis.png" width="500px"/>

<p> Example output of <b>VGG19-OpenPose</b></p>

</div>


#### Details:

- [Pose Estimation](https://github.com/CVBox/PytorchCV/tree/master/methods/pose)
    - Convolutional Pose Machines
    - Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields
    - Associative Embedding: End-to-End Learning for Joint Detection and Grouping
    
- [Object Detection](https://github.com/CVBox/PytorchCV/tree/master/methods/det)
    - SSD: Single Shot MultiBox Detector
    
- [Semantic Segmentation](https://github.com/CVBox/PytorchCV/tree/master/methods/seg)
    - Efficient ConvNet for Real-time Semantic Segmentation

- [Image Classification](https://github.com/CVBox/PytorchCV/tree/master/methods/cls)
    - Densely Connected Convolutional Networks

#### Examples
- Train the openpose model
```bash
python main.py  --hypes hypes/pose/coco/op_coco_pose.json \
                --base_lr 0.001 \
                --phase train
```

- Finetune the openpose model
```bash
python main.py  --hypes hypes/pose/coco/op_coco_pose.json \
                --base_lr 0.001 \
                --phase train \
                --resume checkpoints/pose/coco/coco_open_pose_65000.pth
```

- Test the openpose model(test_img):
```bash
python main.py  --hypes hypes/pose/coco/op_coco_pose.json \
                --phase test \
                --resume checkpoints/pose/coco/coco_open_pose_65000.pth \
                --test_img val/samples/ski.jpg
```

- Test the openpose model(test_dir):
```bash
python main.py  --hypes hypes/pose/coco/op_coco_pose.json \
                --phase test \
                --resume checkpoints/pose/coco/coco_open_pose_65000.pth \
                --test_dir val/samples \
                --gpu 0
```

- Attention: Other command line parameters are showed in main file. You can refer & use them.
