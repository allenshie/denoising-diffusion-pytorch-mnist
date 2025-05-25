import torch
# from torchvision import transforms # Trainer 內部會處理 transform
# from torchvision.datasets import MNIST # 不需要直接加載 MNIST 了
# from torch.utils.data import DataLoader # Trainer 內部會創建 DataLoader
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer # Trainer 來自庫

# 0. 一些設定
image_size = 32       # U-Net 和 Diffusion 期望的圖像大小
batch_size = 128
num_workers = 4       # 給 Trainer 內部 DataLoader 使用
train_timesteps = 1000
train_lr = 1e-5
train_num_steps = 100000 # 總訓練步數
gradient_accumulate_every = 2
ema_decay = 0.995
amp = True
results_folder = './results_mnist_ddpm'
# model_save_path 不需要單獨定義，Trainer 會處理

mnist_image_folder_path = './data/mnist_image_folder' # <--- 指向你保存圖像的文件夾

# 1. 準備數據集 (這一步現在由 Trainer 內部完成，我們只需要提供文件夾路徑)
# --- 以下部分不再需要 ---
# transform = transforms.Compose([
#     transforms.Resize(image_size),
#     transforms.ToTensor(),
#     transforms.Lambda(lambda t: (t * 2) - 1)
# ])
# dataset = MNIST(root='./data', train=True, download=True, transform=transform)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
# --- 以上部分不再需要 ---

# 2. 定義 U-Net 模型
model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels = 1,         # 輸入圖像通道數 (MNIST 是 1)
    flash_attn = False    # 如果你的 PyTorch 版本支持 Flash Attention 且硬體允許，可以設為 True 加速
).cuda()

# 3. 定義高斯擴散過程 (移除 loss_type，因為 denoising-diffusion-pytorch==2.1.1 不支持)
diffusion = GaussianDiffusion(
    model,
    image_size = image_size,    # Trainer 會從 diffusion_model 獲取 image_size
    timesteps = train_timesteps,
    beta_schedule = 'cosine' # 2.1.1 版本應該支持 'cosine'，如果報錯可嘗試 'linear'
).cuda()

# 4. 設置訓練器 Trainer (注意參數名和順序，對應 v2.1.1)
# Trainer 的 __init__ 參數順序是 (self, diffusion_model, folder, *, ...)
# 參考 v2.1.1 的 Trainer 參數:
# ema_decay = 0.995, train_batch_size = 32, train_lr = 1e-4, train_num_steps = 100000,
# gradient_accumulate_every = 2, amp = False, step_start_ema = 2000, update_ema_every = 10,
# save_and_sample_every = 1000, results_folder = './results', num_samples = 25, max_grad_norm = None,
# num_workers = None, persistent_workers = False

trainer = Trainer(
    diffusion_model = diffusion,        # 第一個位置參數
    folder = mnist_image_folder_path,   # 第二個位置參數：圖像文件夾路徑
    # --- 以下為關鍵字參數 (keyword arguments) ---
    train_batch_size = batch_size,
    train_lr = train_lr,
    train_num_steps = train_num_steps,
    gradient_accumulate_every = gradient_accumulate_every,
    ema_decay = ema_decay,
    amp = amp,
    results_folder = results_folder,
    save_and_sample_every = 1000,
    calculate_fid = False
    # num_workers = num_workers # v2.1.1 Trainer 支持 num_workers
    # calculate_fid 在 v2.1.1 的 Trainer 中不存在
    # step_start_ema, update_ema_every, num_samples, max_grad_norm 等參數可以按需設置或使用默認值
)

# (可選) 如果有已保存的模型，v2.1.1 的 Trainer load 方法可能不同，通常是 trainer.load(milestone_number)
# trainer.load(10) # 加載第 10 次保存的模型 (例如 10000 步)
milestone_to_load = 10
model_path_to_load = trainer.results_folder / f'model-{milestone_to_load}.pt'

if model_path_to_load.exists():
    print(f"從 milestone {milestone_to_load} 加載預訓練權重...")
    trainer.load(milestone_to_load) # Trainer.load 會處理模型、EMA、優化器狀態的加載
    # Trainer.load 會更新 self.step，所以訓練會從加載的步數繼續
    # 你可能需要手動調整 pbar 的 initial 值，或者確保 trainer.step 被正確設置
    print(f"權重加載完畢，當前 step: {trainer.step}")
else:
    print(f"未找到預訓練權重: {model_path_to_load}")

print("開始新的訓練 (或繼續訓練)...")




# 5. 開始訓練
print("開始訓練 DDPM on MNIST...")
trainer.train()

print(f"訓練完成！結果保存在 {trainer.results_folder}")
# v2.1.1 Trainer 會將 EMA 模型保存在 results_folder 下，文件名通常是 model-{step}.pt
# 最新的 EMA 模型通常不會直接叫 model.pt，而是按步數命名

# 6. (可選) 訓練完成後進行採樣
# 需要加載對應的 EMA 模型權重
# 例如，如果訓練了 10000 步，保存間隔是 1000，最後一個模型可能是 model-10.pt (代表 10*1000 步)
# trained_ema_model_path = f'{results_folder}/model-10.pt' # 假設訓練完成後的模型
# if os.path.exists(trained_ema_model_path):
#     print(f"加載訓練好的 EMA 模型從: {trained_ema_model_path}")
#     # 在 v2.1.1 中，Trainer 保存的是整個 diffusion_model 的 state_dict
#     # 但加載時通常是加載到 diffusion_model.ema_model (即 EMA 版本的 Unet)
#     # 或者直接用 Trainer 的 load 方法，它會處理 EMA
#     # 更簡單的方式是，如果 trainer 實例還在，可以直接用 trainer.ema_model 採樣
#     # diffusion.ema_model.load_state_dict(torch.load(trained_ema_model_path)['ema_model']) # 類似這樣，具體看保存的字典結構
#
#     # 或者更直接地，如果 Trainer.train() 執行完畢，diffusion.ema_model 就是最新的EMA模型
#     # sampled_images = diffusion.sample(batch_size = 16) # 使用 EMA 模型進行採樣 (GaussianDiffusion 內部會判斷是否用 EMA)
#
#     # 使用 Trainer 內部管理的 EMA 模型進行採樣
#     # (需要確保 diffusion 實例是 Trainer 使用的那個，並且 EMA 已經更新)
#     all_images_list = list(map(lambda n: trainer.ema_model.sample(batch_size=n), [16])) # trainer.ema_model 就是EMA UNET
#     sampled_images = all_images_list[0]
#
#     print(f"採樣圖片的形狀: {sampled_images.shape}")
#
#     from torchvision.utils import save_image
#     save_image(sampled_images, f'{results_folder}/final_sampled_images.png', nrow=4, normalize=True, value_range=(-1,1))
#     print(f"採樣圖片已保存到 {results_folder}/final_sampled_images.png")
# else:
# print(f"未找到訓練好的模型於: {trained_ema_model_path}")