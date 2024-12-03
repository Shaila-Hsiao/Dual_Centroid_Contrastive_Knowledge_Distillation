import subprocess

# 第一個指令 (PCL 訓練)
cmd1 = [
    "python", "main_pcl.py", 
    "-a", "resnet50",
    "--lr", "0.03",
    "--batch-size", "256",
    "--temperature", "0.2",
    "--mlp", "--aug-plus", "--cos",  # PCL v2 激活參數
     r"C:\Users\k3866\Documents\Datasets\tiny_imagenet\tiny-imagenet-200",
    "--gpu", "0", 
    "--workers", "12", 
    "--epochs", "200"
]

# 第二個指令 (線性分類器評估)
cmd2 = [
    "python", "eval_cls_imagenet.py", 
    "--pretrained", r"C:\Users\k3866\Documents\PretrianedModel\PCL\PCL_v2_epoch200.pth.tar",  # 更新模型檔路徑
    # "--pretrained", r"C:\Users\k3866\Documents\Projects\Dual_Centroid_Contrastive_Knowledge_Distillation\checkpoint\pcl\train\checkpoint_0399.pth.tar",  # 更新模型檔路徑
    "-a", "resnet50", 
    "--lr", "0.1",
    "--batch-size", "256",
     "--workers", "12",
    "--id", "ImageNet_linear",
    r"C:\Users\k3866\Documents\Datasets\tiny_imagenet\tiny-imagenet-200"
]

try:
    # print("執行第一個指令 (PCL 訓練)...")
    # subprocess.run(cmd1, check=True)  # 執行第一個指令
    print("第一個指令執行完成，執行第二個指令 (線性分類器評估)...")
    subprocess.run(cmd2, check=True)  # 執行第二個指令
    print("第二個指令執行完成")
except subprocess.CalledProcessError as e:
    print(f"執行失敗: {e}")
