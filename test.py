#python main_pcl_mask.py -a resnet50 --epoch 200 --aug-plus --cos "C:\Users\user\.cache\kagglehub\datasets\akash2sharma\tiny-imagenet\versions\1\tiny-imagenet-200\tiny-imagenet-200" 

import subprocess

# # 第一個指令 (PCL 訓練)
# cmd1 = [
#     "python", "main_pcl.py", 
#     "-a", "resnet50",
#     "--lr", "0.03",
#     "--batch-size", "256",
#     "--temperature", "0.2",
#     "--aug-plus", "--cos",  # PCL v2 激活參數
#      r"C:\Users\k3866\Documents\Datasets\tiny_imagenet\tiny-imagenet-200",
#     "--gpu", "0", 
#     "--workers", "12", 
#     "--epochs", "200"
# ]
# 第一個指令 (PCL 訓練)
cmd1 = [
    "python", "main_pcl.py", 
    "-a", "resnet50",
    "--lr", "0.03",
    "--batch-size", "256",
    "--temperature", "0.2",
    "--aug-plus", "--cos","--mlp",  # PCL v2 激活參數
    # r"C:\Users\k3866\Documents\Datasets\tiny_imagenet\tiny-imagenet-200" ,
    "--gpu", "0", 
    "--workers", "16", 
    "--epochs", "200",
    "--exp-dir",r"pth_pcl\cifar100",
    "--id", "PCL",
    "--dataset","CIFAR100",
    r"C:\Users\k3866\Documents\Datasets",
]

# 第二個指令 (線性分類器評估)
cmd2 = [
    "python", "eval_cls_imagenet.py", 
    "--pretrained", r"pth_pcl\cifar100\checkpoint_0199.pth.tar",  # 更新模型檔路徑
    # "--pretrained", r"C:\Users\k3866\Documents\Projects\Dual_Centroid_Contrastive_Knowledge_Distillation\checkpoint\pcl\train\checkpoint_0399.pth.tar",  # 更新模型檔路徑
    "-a", "resnet50", 
    "--lr", "5",
    "--batch-size", "256",
    "--epochs","200",
     "--workers", "16",
    "--id", "PCL",
    "--dataset","CIFAR100",
    r"C:\Users\k3866\Documents\Datasets"
    # r"C:\Users\k3866\Documents\Datasets\tiny_imagenet\tiny-imagenet-200" ,
]

# # 第二個指令 (線性分類器評估)
# cmd3 = [
#     "python", "eval_cls_imagenet.py", 
#     "--pretrained", r"pth_pcl\checkpoint_0199.pth.tar",  # 更新模型檔路徑
#     # "--pretrained", r"C:\Users\k3866\Documents\Projects\Dual_Centroid_Contrastive_Knowledge_Distillation\checkpoint\pcl\train\checkpoint_0399.pth.tar",  # 更新模型檔路徑
#     "-a", "resnet50", 
#     "--lr", "0.1",
#     "--batch-size", "256",
#     "--epochs","200",
#      "--workers", "12",
#     "--id", "Tiny_ImageNet_linear_PCL",
#     "--dataset","TinyImageNet",
#     r"C:\Users\user\.cache\kagglehub\datasets\akash2sharma\tiny-imagenet\versions\1\tiny-imagenet-200\tiny-imagenet-200" 
# ]

try:
    print("執行第一個指令 (PCL 訓練)...")
    subprocess.run(cmd1, check=True)  # 執行第一個指令
    print("第一個指令執行完成，執行第二個指令 (線性分類器評估)...")
    subprocess.run(cmd2, check=True)  # 執行第二個指令
    print("第二個指令執行完成")
    # print("第二個指令執行完成，執行第三個指令 (PCL 線性分類器評估)...")
    # subprocess.run(cmd3, check=True)  # 執行第二個指令
    # print("第三個指令執行完成")
except subprocess.CalledProcessError as e:
    print(f"執行失敗: {e}")
