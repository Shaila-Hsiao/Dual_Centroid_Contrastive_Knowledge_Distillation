## Dynamic Cluster-Based Contrastive Learning with Knowledge Distillation
<img src="./img/DCBCL_framework.png" width="600">

### Requirements:
* Tiny-ImageNet dataset
* Python ≥ 3.10
* PyTorch ≥ 2.5
* <a href="https://anaconda.org/conda-forge/faiss-gpu">faiss-gpu</a>: conda install conda-forge::faiss-gpu
* pip install tqdm

### Pre-Train: 
<pre>python main_dcbcl_resnet.py \ 
  -a resnet50 \ 
  --lr 0.05 \
  --batch-size 256 \
  --temperature 0.05 \
  --mlp --aug-plus --cos (only activated for PCL v2)\
  --proportion 0.2 \
  --alpha 0.2 \
  --dataset CIFAR10 \
  --exp-dir []\
  --pretrained [path to pretrained teacher checkpoint]\
  --student-ratio 20% \
  --use-kd \
  --use-centroid \
  --use-masking \
  [Tiny-Imagenet dataset folder]
</pre>

### Download Pre-trained Teacher Models
<a href="https://drive.google.com/file/d/1JZ5YX6AUukPm8hB2RWMCgW0MUABG6650/view?usp=drive_link">MoCo</a>| 


### Linear Evaluation 
<pre>python eval_cls_imagenet_ratio.py --pretrained [your pretrained model] \
  -a resnet50 \
  --lr 0.01 \
  --batch-size 256 \
  --epochs 200 \
  --student-ratio 20% \
  --dataset TinyImageNet \
  --exp-dir [your evaluation output directory] \
  [Tiny-ImageNet dataset folder]
</pre>

