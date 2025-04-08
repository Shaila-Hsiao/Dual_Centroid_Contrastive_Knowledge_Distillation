#python main_pcl_mask.py -a resnet50 --epoch 200 --aug-plus --cos "C:\Users\user\.cache\kagglehub\datasets\akash2sharma\tiny-imagenet\versions\1\tiny-imagenet-200\tiny-imagenet-200" 

import subprocess
# tiny Imagenet
# # 第一個指令 (PCL 訓練)
# mask_proportion  
# mask_threshold 1~6
cmd1 = [
    "python", "main_pcl_csim_mask_kd.py", 
    "-a", "resnet50",
    "--lr", "0.03",
    "--batch-size", "256",
    "--temperature", "0.03",
    "--aug-plus", "--cos","--mlp",  # PCL v2 激活參數
    "--gpu", "0", 
    "--workers", "12",
    "--warmup-epoch","20",
    "--epochs", "200",
    "--mask_mode","mask_proportion",
    # "--mask_mode","mask_threshold",
    # "--dist_threshold","0.1",
    "--proportion","0.1",
    "--alpha","0.2",
    "--id", "PCMK",
    "--dataset","TinyImageNet",
    "--exp-dir",r"save\pcl_csim_mask_kd\tinyimagenet\train",
    "--pretrained",r"D:\Document\Project\PretrainedModel\MoCo\save\tinyimagenet\checkpoint_last.pth.tar",
    r"C:\Users\user\.cache\kagglehub\datasets\akash2sharma\tiny-imagenet\versions\1\tiny-imagenet-200\tiny-imagenet-200" ,
    # r"D:\Document\Project\Dataset",
    '--use-kd',
    '--use-centroid',
    '--use-masking'
]
# 第二個指令 (線性分類器評估)
cmd2 = [
    "python", "eval_cls_imagenet.py", 
    "--pretrained", r"save\pcl_csim_mask_kd\tinyimagenet\train\20250407\model_best.pth.tar",  # 更新模型檔路徑
    # "--pretrained", r"C:\Users\k3866\Documents\Projects\Dual_Centroid_Contrastive_Knowledge_Distillation\checkpoint\pcl\train\checkpoint_0399.pth.tar",  # 更新模型檔路徑
    "-a", "resnet50", 
    "--lr", "0.01",
    "--batch-size", "256",
    "--epochs","200",
    "--workers", "12",
    "--id", "PCMK",
    "--dataset","TinyImageNet",
    "--exp-dir",r"save\pcl_csim_mask_kd\tinyimagenet\val",
    r"C:\Users\user\.cache\kagglehub\datasets\akash2sharma\tiny-imagenet\versions\1\tiny-imagenet-200\tiny-imagenet-200" 
    # r"D:\Document\Project\Dataset",
]
# CIFAR100
cmd3 = [
    "python", "main_pcl_csim_mask_kd.py", 
    "-a", "resnet50",
    "--lr", "0.01",
    "--batch-size", "128",
    "--temperature", "0.03",
    "--aug-plus", "--cos","--mlp",  # PCL v2 激活參數
    "--gpu", "0", 
    "--workers", "12", 
    "--warmup-epoch","10",
    "--epochs", "200",
    "--mask_mode","mask_proportion",
    # "--mask_mode","mask_threshold",
    # "--dist_threshold","1",
    "--proportion","0.2",
    "--alpha","0.2",
    "--id", "PCMK",
    "--dataset","CIFAR100",
    "--exp-dir",r"save\dcbcl\cifar100\bs128\train",
    "--pretrained",r"D:\Document\Project\PretrainedModel\MoCo\save\cifar100\checkpoint_last.pth.tar",
    # r"C:\Users\user\.cache\kagglehub\datasets\akash2sharma\tiny-imagenet\versions\1\tiny-imagenet-200\tiny-imagenet-200" ,
    r"D:\Document\Project\Dataset",
    '--use-kd',
    '--use-centroid',
    '--use-masking'
]
cmd4 = [
    "python", "eval_cls_imagenet.py", 
    "--pretrained", r"save\dcbcl\cifar100\bs128\train\20250408\model_best.pth.tar",  # 更新模型檔路徑
    # "--pretrained", r"C:\Users\k3866\Documents\Projects\Dual_Centroid_Contrastive_Knowledge_Distillation\checkpoint\pcl\train\checkpoint_0399.pth.tar",  # 更新模型檔路徑
    "-a", "resnet50", 
    "--lr", "0.01",
    "--batch-size", "256",
    "--epochs","200",
     "--workers", "12",
    "--id", "PCMK",
    "--dataset","CIFAR100",
    "--exp-dir",r"save\dcbcl\cifar100\val",
    # r"C:\Users\user\.cache\kagglehub\datasets\akash2sharma\tiny-imagenet\versions\1\tiny-imagenet-200\tiny-imagenet-200" 
    r"D:\Document\Project\Dataset",
]

# mask_threshold 0.5
# CIFAR10
cmd5 = [
    "python", "main_pcl_csim_mask_kd.py", 
    "-a", "resnet50",
    "--lr", "0.03",
    "--batch-size", "256",
    "--temperature", "0.03",
    "--aug-plus", "--cos","--mlp",  # PCL v2 激活參數
    "--gpu", "0", 
    "--workers", "12", 
    "--warmup-epoch","10",
    "--epochs", "200",
    "--mask_mode","mask_proportion",
    "--proportion","0.2",
    #  "--mask_mode","mask_threshold",
    # "--dist_threshold","1",
    "--alpha","0.8",
    "--id", "PCMK",
    "--dataset","CIFAR10",
    "--exp-dir",r"save\pcl_csim_mask_kd\cifar10\train",
    "--pretrained",r"D:\Document\Project\PretrainedModel\MoCo\save\cifar10\checkpoint_last.pth.tar",
    # r"C:\Users\user\.cache\kagglehub\datasets\akash2sharma\tiny-imagenet\versions\1\tiny-imagenet-200\tiny-imagenet-200" ,
    r"D:\Document\Project\Dataset",
]
cmd6 = [
    "python", "eval_cls_imagenet.py", 
    "--pretrained", r"save\pcl_csim_mask_kd\cifar10\train\checkpoint_0199.pth.tar",  # 更新模型檔路徑
    # "--pretrained", r"C:\Users\k3866\Documents\Projects\Dual_Centroid_Contrastive_Knowledge_Distillation\checkpoint\pcl\train\checkpoint_0399.pth.tar",  # 更新模型檔路徑
    "-a", "resnet50", 
    "--lr", "0.1",
    "--batch-size", "256",
    "--epochs","200",
     "--workers", "12",
    "--id", "PCMK",
    "--dataset","CIFAR10",
    "--exp-dir",r"save\pcl_csim_mask_kd\cifar10\val",
    # r"C:\Users\user\.cache\kagglehub\datasets\akash2sharma\tiny-imagenet\versions\1\tiny-imagenet-200\tiny-imagenet-200" 
    r"D:\Document\Project\Dataset",
]
# threshold 0.5
# TinyImageNet
cmd7 = [
    "python", "main_pcl_csim_mask_kd.py", 
    "-a", "resnet50",
    "--lr", "0.03",
    "--batch-size", "256",
    "--temperature", "0.03",
    "--aug-plus", "--cos","--mlp",  # PCL v2 激活參數
    "--gpu", "0", 
    "--workers", "12",
    "--warmup-epoch","10", 
    "--epochs", "200",
    # "--mask_mode","mask_proportion",
    "--mask_mode","mask_threshold",
    "--dist_threshold","1",
    # "--proportion","0.05",
    "--alpha","0.2",
    "--id", "PCMK",
    "--dataset","TinyImageNet",
    "--exp-dir",r"save\pcl_csim_mask_kd\tinyimagenet\train",
    "--pretrained",r"D:\Document\Project\PretrainedModel\MoCo\save\tinyimagenet\checkpoint_last.pth.tar",
    r"C:\Users\user\.cache\kagglehub\datasets\akash2sharma\tiny-imagenet\versions\1\tiny-imagenet-200\tiny-imagenet-200" ,
    # r"D:\Document\Project\Dataset",
]
# 第八個指令 (線性分類器評估)
cmd8 = [
    "python", "eval_cls_imagenet.py", 
    "--pretrained", r"save\pcl_csim_mask_kd\tinyimagenet\train\checkpoint_0199.pth.tar",  # 更新模型檔路徑
    # "--pretrained", r"C:\Users\k3866\Documents\Projects\Dual_Centroid_Contrastive_Knowledge_Distillation\checkpoint\pcl\train\checkpoint_0399.pth.tar",  # 更新模型檔路徑
    "-a", "resnet50", 
    "--lr", "0.1",
    "--batch-size", "256",
    "--epochs","200",
    "--workers", "12",
    
    "--id", "PCMK",
    "--dataset","TinyImageNet",
    "--exp-dir",r"save\pcl_csim_mask_kd\tinyimagenet\val",
    r"C:\Users\user\.cache\kagglehub\datasets\akash2sharma\tiny-imagenet\versions\1\tiny-imagenet-200\tiny-imagenet-200" 
    # r"D:\Document\Project\Dataset",
]
try:
    # print("執行第一個指令 (PCMK 訓練)...")
    # subprocess.run(cmd1, check=True)  # 執行第一個指令
    # print("第一個指令執行完成，執行第二個指令 (線性分類器評估)...")
    # subprocess.run(cmd2, check=True)  # 執行第二個指令
    # print("第二個指令執行完成")
    print("執行第三個指令 (PCMK 訓練)...")
    subprocess.run(cmd3, check=True)  # 執行第一個指令
    print("第三個指令執行完成，執行第四個指令 (線性分類器評估)...")
    subprocess.run(cmd4, check=True)  # 執行第二個指令
    # print("第四個指令執行完成")
    # print("執行第五個指令 (PCMK 訓練)...")
    # subprocess.run(cmd5, check=True)  # 執行第一個指令
    # print("第五個指令執行完成，執行第六個指令 (線性分類器評估)...")
    # subprocess.run(cmd6, check=True)  # 執行第二個指令
    # # print("執行第七個指令 (PCMK 訓練)...")
    # subprocess.run(cmd7, check=True)  # 執行第一個指令
    # print("第七個指令執行完成，執行第八個指令 (線性分類器評估)...")
    # subprocess.run(cmd8, check=True)  # 執行第二個指令
    # print("第八個指令執行完成")
except subprocess.CalledProcessError as e:
    print(f"執行失敗: {e}")
