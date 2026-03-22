# Text2VDM: Text to Vector Displacement Maps for Expressive and Interactive 3D Sculpting

<h4 align="center">

[Hengyu Meng](https://hengyumeng.github.io/), [Duotun Wang](https://www.duotun-wang.co.uk/), [Zhijing Shao](https://initialneil.github.io/), [Ligang Liu](http://staff.ustc.edu.cn/~lgliu/), [Zeyu Wang<sup>†</sup>](https://cislab.hkust-gz.edu.cn/members/zeyu-wang/)

[![arXiv](https://img.shields.io/badge/arXiv-2506.05573-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2502.20045)

<p>
    <img width="90%" alt="pipeline", src="./assets/text2vdm.jpg">
</p>
</h4>

##  Abstract

Professional 3D asset creation often requires diverse sculpting brushes to add surface details and geometric structures. Despite recent progress in 3D generation, producing reusable sculpting brushes compatible with artists' workflows remains an open and challenging problem. These sculpting brushes are typically represented as vector displacement maps (VDMs), which existing models cannot easily generate compared to natural images. This paper presents Text2VDM, a novel framework for text-to-VDM brush generation through the deformation of a dense planar mesh guided by score distillation sampling (SDS). The original SDS loss is designed for generating full objects and struggles with generating desirable sub-object structures from scratch in brush generation. We refer to this issue as semantic coupling, which we address by introducing weighted blending of prompt tokens to SDS, resulting in a more accurate target distribution and semantic guidance. Experiments demonstrate that Text2VDM can generate diverse, high-quality VDM brushes for sculpting surface details and geometric structures. Our generated brushes can be seamlessly integrated into mainstream modeling software, enabling various applications such as mesh stylization and real-time interactive modeling.


## News
- **2025-06-26**: Text2VDM is accepted by ICCV 2025! 
- **Code released!**

## Getting started

This code was developed on Ubuntu 22.04 with Python 3.9, CUDA 11.8 and PyTorch 2.1.0, using NVIDIA RTX 4090 (24GB) GPU. Later versions should work, but have not been tested.

### Environment setup

```python
conda create -n text2vdm python=3.9
conda activate text2vdm

# install required packages
pip install -r requirements.txt

# install PyTorch3D: 
pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'
```

### Generate VDM brushes from text

```python
# Generate VDM brush from text
sh run.sh
```

### Control the generation

```python
# Control with initial shape
	use_shape_map = True # set this flag with True
    mask_strength = 1
    shape_strength = 0.8 # the strength of displacement of vertices for shape map
	path = 'horn'  # change the path in gen_inverse.py
    base_mask = cv2.imread('./shape_map/'+ path + '_mask.png')
    shape_map = cv2.imread('./shape_map/'+ path + '.png')

# Control only with mask
	use_shape_map = False # set this flag with False
    mask_strength = 0.5 # you can change the rate form 0 to 1
    base_mask = cv2.imread('./masks/'+ 'your_mask.png')
```



## Acknowledgement

We would like to thank the authors of [Large Steps in Inverse Rendering of Geometry](https://github.com/rgl-epfl/large-steps-pytorch), and [Paint-it](https://github.com/kaist-ami/Paint-it) for their great work and generously providing source codes, which inspired our work and helped us a lot in the implementation. 


## Citation
If you find our work helpful, please consider citing:
```bibtex
@inproceedings{meng2025text2vdmtextvectordisplacement,
  title={Text2VDM: Text to Vector Displacement Maps for Expressive and Interactive 3D Sculpting}, 
  author={Meng, Hengyu and Wang, Duotun and Shao, Zhijing and Liu, Ligang and Wang, Zeyu},
  booktitle = {IEEE Conference on International Conference on Computer Vision (ICCV)},
  publisher={IEEE},
  html={https://arxiv.org/abs/2502.20045}, 
  year={2025}
}
```
