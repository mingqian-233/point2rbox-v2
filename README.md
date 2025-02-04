## Introduction
We develop **Point2RBox-v2** (Rethinking Point-supervised Oriented Object Detection with Spatial Layout Among Instances). In this paper, we rethink this challenging task setting with the layout among instances. At the core are three principles: 1) Gaussian overlap loss. It learns an upper bound for each instance by treating objects as 2D Gaussian distributions and minimizing their overlap. 2) Voronoi watershed loss. It learns a lower bound for each instance through watershed on Voronoi tessellation. 3) Consistency loss. It learns the size/rotation variation between two output sets with respect to an input image and its augmented view. Supplemented by a few devised techniques, e.g. edge loss and copy-paste, the detector is further enhanced.

This project is the implementation of Point2RBov-v2. The code works with **PyTorch 1.13+** and it is forked from [MMRotate dev-1.x](https://github.com/open-mmlab/mmrotate/tree/dev-1.x). MMRotate is an open-source toolbox for rotated object detection based on PyTorch. It is a part of the [OpenMMLab project](https://github.com/open-mmlab).

## Installation
Please refer to [Installation](https://mmrotate.readthedocs.io/en/1.x/get_started.html) for more detailed instructions.

## Getting Started
Please see [Overview](https://mmrotate.readthedocs.io/en/1.x/overview.html) for the general introduction of MMRotate. 

For detailed user guides and advanced guides, please refer to MMRotate's [documentation](https://mmrotate.readthedocs.io/en/1.x/).

The examples of training and testing Point2RBox-v2 can be found [here](configs/point2rbox_v2/README.md).

## Model Zoo
This repository contains our series of work on weakly-supervised OOD.

<details open>
<summary><b>Supported algorithms:</b></summary>

- [x] [Wholly-WOOD](configs/whollywood/README.md)
- [x] [H2RBox](configs/h2rbox/README.md)
- [x] [H2RBox-v2](configs/h2rbox_v2/README.md)
- [x] [Point2RBox](configs/point2rbox/README.md)
- [x] [Point2RBox-v2](configs/point2rbox_v2/README.md)

</details>

## Data Preparation
Please refer to [data_preparation.md](tools/data/README.md) to prepare the data.

## FAQ
Please refer to [FAQ](docs/en/notes/faq.md) for frequently asked questions.

## Acknowledgement
This project is based on MMRotate, an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks. We appreciate the [Student Innovation Center of SJTU](https://www.si.sjtu.edu.cn/) for providing rich computing resources at the beginning of the project. We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new methods.

## Citation
```
@inproceedings{yang2023h2rbox,
  title={H2RBox: Horizontal Box Annotation is All You Need for Oriented Object Detection},
  author={Yang, Xue and Zhang, Gefan and Li, Wentong and Wang, Xuehui and Zhou, Yue and Yan, Junchi},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
@inproceedings{yu2023h2rboxv2,
  author={Yi Yu and Xue Yang and Qingyun Li and Yue Zhou and Feipeng Da and Junchi Yan},
  title={H2RBox-v2: Incorporating Symmetry for Boosting Horizontal Box Supervised Oriented Object Detection}, 
  booktitle={Advances in Neural Information Processing Systems},
  year={2023},
}
@inproceedings{yu2024point2rbox,
  title={Point2RBox: Combine Knowledge from Synthetic Visual Patterns for End-to-end Oriented Object Detection with Single Point Supervision},
  author={Yu, Yi and Yang, Xue and Li, Qingyun and Da, Feipeng and Dai, Jifeng and Qiao, Yu and Yan, Junchi},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16783--16793},
  year={2024}
}
```


