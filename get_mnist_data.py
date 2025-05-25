import os
from torchvision.datasets import MNIST
from PIL import Image

# 1. 加載 MNIST 數據集 (這次不立即轉換為 Tensor 或進行 transform)
print("下載 MNIST 數據集 (如果需要)...")
mnist_download = MNIST(root='./data', train=True, download=True)

# 2. 創建保存圖像的文件夾
output_folder = './data/mnist_image_folder' # 你可以自己命名
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"已創建文件夾: {output_folder}")
else:
    print(f"文件夾已存在: {output_folder}")

# 3. 遍歷 MNIST 數據集並保存為圖像文件 (例如 PNG)
print(f"開始將 MNIST 圖像保存到 {output_folder}...")
num_images_saved = 0
for i, (img_pil, label) in enumerate(mnist_download):
    # img_pil 是一個 PIL Image 對象
    try:
        img_pil.save(os.path.join(output_folder, f'mnist_train_{i:05d}.png'))
        num_images_saved += 1
    except Exception as e:
        print(f"保存圖像 {i} 時出錯: {e}")
    if (i+1) % 5000 == 0:
        print(f"已處理 {i+1}/{len(mnist_download)} 圖像...")

print(f"MNIST 圖像保存完成。總共保存了 {num_images_saved} 張圖像到 {output_folder}")