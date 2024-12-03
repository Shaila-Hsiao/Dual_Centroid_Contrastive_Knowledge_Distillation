import os
import shutil

def organize_tinyimagenet_val(data_dir):
    """
    Reorganizes the 'val' folder of Tiny ImageNet dataset.

    Args:
        data_dir (str): Path to the Tiny ImageNet dataset.
    """
    val_dir = os.path.join(data_dir, "val")
    images_dir = os.path.join(val_dir, "images")
    annotations_file = os.path.join(val_dir, "val_annotations.txt")

    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not os.path.exists(annotations_file):
        raise FileNotFoundError(f"Annotations file not found: {annotations_file}")

    # Step 1: Create directories for each class
    with open(annotations_file, "r") as f:
        annotations = f.readlines()

    for line in annotations:
        parts = line.strip().split("\t")
        image_name, class_label = parts[0], parts[1]
        class_dir = os.path.join(val_dir, class_label)
        os.makedirs(class_dir, exist_ok=True)

        # Step 2: Move images into corresponding class directories
        src_path = os.path.join(images_dir, image_name)
        dst_path = os.path.join(class_dir, image_name)

        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
        else:
            print(f"Warning: {src_path} does not exist!")

    # Step 3: Remove the original 'images' directory
    if os.path.exists(images_dir) and not os.listdir(images_dir):
        os.rmdir(images_dir)
        print(f"Removed empty directory: {images_dir}")
    else:
        print(f"Could not remove {images_dir}. It may contain residual files.")

    print("Reorganization complete!")


if __name__ == "__main__":
    # Modify the path below to the root directory of Tiny ImageNet dataset
    #tiny_imagenet_root = r"C:\Users\user\.cache\kagglehub\datasets\akash2sharma\tiny-imagenet\versions\1\tiny-imagenet-200\tiny-imagenet-200"
    tiny_imagenet_root = r"C:\Users\k3866\Documents\Datasets\tiny_imagenet\tiny-imagenet-200"
    organize_tinyimagenet_val(tiny_imagenet_root)
