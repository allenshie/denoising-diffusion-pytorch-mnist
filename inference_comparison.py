import torch
from torchvision.utils import save_image
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from pathlib import Path
import time
import argparse # 引入 argparse

# --- 預設設定 (大部分會從 args 讀取或基於 args 計算) ---
# 模型和數據相關
IMAGE_SIZE_DEFAULT = 32
CHANNELS_DEFAULT = 1
RESULTS_FOLDER_DEFAULT = './results_mnist_ddpm'
UNET_DIM_DEFAULT = 64
UNET_DIM_MULTS_DEFAULT = (1, 2, 4, 8) # 注意 argparse 如何處理 tuple

# Diffusion 參數 (與訓練時的 GaussianDiffusion 設置大部分一致)
ORIGINAL_TRAIN_TIMESTEPS_DEFAULT = 1000
BETA_SCHEDULE_TRAIN_DEFAULT = 'cosine'
OBJECTIVE_TRAIN_DEFAULT = 'pred_noise' # 非常重要，要與訓練時一致

# 設備
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Compare DDPM and DDIM sampling strategies.")
    parser.add_argument("--strategy", type=str, required=True, choices=['ddpm', 'ddim'],
                        help="Sampling strategy to use: 'ddpm' or 'ddim'.")
    parser.add_argument("--milestone", type=int, required=True,
                        help="Milestone number of the trained model checkpoint to load.")
    parser.add_argument("--sample_steps", type=int, default=None,
                        help="Number of steps for sampling. Default: 1000 for DDPM, 50 for DDIM.")
    parser.add_argument("--ddim_eta", type=float, default=0.0,
                        help="Eta parameter for DDIM sampling (0.0 for deterministic). Only used if strategy is 'ddim'.")
    parser.add_argument("--num_samples", type=int, default=25,
                        help="Number of images to generate.")
    parser.add_argument("--batch_size", type=int, default=25,
                        help="Batch size for inference.")
    parser.add_argument("--results_folder", type=str, default=RESULTS_FOLDER_DEFAULT,
                        help="Path to the folder where trained models and results are stored.")
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE_DEFAULT, help="Size of the input image.")
    parser.add_argument("--channels", type=int, default=CHANNELS_DEFAULT, help="Number of image channels.")
    parser.add_argument("--unet_dim", type=int, default=UNET_DIM_DEFAULT, help="Base dimension for U-Net.")
    # argparse 不直接支持 tuple，通常用 nargs='+' 和 type=int，然後在代碼中轉換
    # 或者用字符串然後分割，這裡為了簡單，保持為固定值或讓用戶在腳本內修改 UNET_DIM_MULTS_DEFAULT
    parser.add_argument("--train_timesteps", type=int, default=ORIGINAL_TRAIN_TIMESTEPS_DEFAULT,
                        help="Total timesteps used during the original training of the U-Net model.")
    parser.add_argument("--beta_schedule", type=str, default=BETA_SCHEDULE_TRAIN_DEFAULT,
                        help="Beta schedule used during original training (e.g., 'cosine', 'linear').")
    parser.add_argument("--objective", type=str, default=OBJECTIVE_TRAIN_DEFAULT,
                        help="Objective used during original training (e.g., 'pred_noise', 'pred_x0', 'pred_v').")


    args = parser.parse_args()

    # 根據策略設置 sample_steps 的默認值
    if args.sample_steps is None:
        if args.strategy == 'ddpm':
            args.sample_steps = args.train_timesteps # DDPM 通常用訓練時的總步數
        elif args.strategy == 'ddim':
            args.sample_steps = 50 # DDIM 的一個常用默認快速採樣步數
    
    # 確保 DDPM 的 sample_steps 等於 train_timesteps (如果使用 DDPM 策略)
    # 或者說，DDPM 的 sampling_timesteps 參數在 GaussianDiffusion 中應等於 timesteps
    if args.strategy == 'ddpm' and args.sample_steps != args.train_timesteps:
        print(f"警告: DDPM 策略通常使用與訓練時相同的步數 ({args.train_timesteps})。")
        print(f"當前 sample_steps 設置為 {args.sample_steps}。")
        # 在 GaussianDiffusion 中，如果 sampling_timesteps < timesteps，它會自動切換到 DDIM。
        # 所以對於純 DDPM，要確保 GaussianDiffusion 的 sampling_timesteps 參數等於其 timesteps 參數。

    return args

def main():
    args = parse_args()

    print(f"使用設備: {DEVICE}")
    print(f"當前配置: {args}")

    # 輸出文件夾
    output_strategy_folder = Path(args.results_folder) / f"inference_milestone_{args.milestone}" / args.strategy
    output_strategy_folder.mkdir(parents=True, exist_ok=True)

    # --- 1. 創建 U-Net 模型結構 ---
    print("創建 U-Net 模型...")
    model = Unet(
        dim=args.unet_dim,
        dim_mults=UNET_DIM_MULTS_DEFAULT, # 從全局默認獲取，或更複雜的 argparse 處理
        channels=args.channels,
        flash_attn=False
    ).to(DEVICE)

    # --- 2. 加載訓練好的權重 ---
    model_path = Path(args.results_folder) / f'model-{args.milestone}.pt'
    if not model_path.exists():
        print(f"錯誤：未找到模型權重文件 {model_path}")
        return

    print(f"從 {model_path} 加載權重...")
    # 注意：如果你的 PyTorch 版本較新且 .pt 文件來自舊版本，可能需要 weights_only=True
    # 但對於這種自己保存和加載的情況，通常不需要。
    # 如果遇到 UnpicklingError 或類似問題，可以嘗試 map_location=DEVICE 或 torch.load(..., weights_only=True)
    # 但首先確保加載邏輯正確。
    saved_checkpoint = torch.load(str(model_path), map_location=DEVICE) # <--- 修改變量名以示區分

    # 關鍵：從保存的 checkpoint 中提取 U-Net (EMA 版本) 的 state_dict
    if 'ema' in saved_checkpoint and hasattr(saved_checkpoint['ema'], 'state_dict'):
        # 如果保存的是整個 EMA 對象的 state_dict (如新版 EMA)
        ema_state_dict = saved_checkpoint['ema']
        if 'ema_model' in ema_state_dict: # 假設 ema_model 是 Unet 的 state_dict
            unet_weights_to_load = ema_state_dict['ema_model']
        elif 'model' in ema_state_dict and isinstance(ema_state_dict['model'], dict): # 另一種可能的保存結構
            unet_weights_to_load = ema_state_dict['model']
        else: # 嘗試直接加載 ema_state_dict 是否就是 unet 的 state_dict (不太可能但兼容一下)
            # 檢查鍵是否匹配 Unet
            if all(k.startswith("ema_model.") for k in ema_state_dict.keys()): # 舊版 ema_pytorch 保存方式
                unet_weights_to_load = {k.replace("ema_model.", ""): v for k,v in ema_state_dict.items() if k.startswith("ema_model.")}
            else: # 如果 ema_state_dict 的鍵直接匹配 Unet
                is_unet_dict = True
                temp_unet = Unet(dim=args.unet_dim, dim_mults=UNET_DIM_MULTS_DEFAULT, channels=args.channels)
                for key in ema_state_dict.keys():
                    if key not in temp_unet.state_dict():
                        is_unet_dict = False
                        break
                if is_unet_dict:
                    unet_weights_to_load = ema_state_dict
                else:
                    print(f"錯誤：在 'ema' 中找到了 state_dict，但無法識別 'ema_model' 或直接匹配 U-Net 結構。")
                    print(f"EMA state_dict keys: {list(ema_state_dict.keys())[:5]}...")
                    return
    elif 'ema_model' in saved_checkpoint: # 有些版本可能直接保存 ema_model 的 state_dict
        unet_weights_to_load = saved_checkpoint['ema_model']
    else:
        # 如果沒有 EMA，嘗試加載 GaussianDiffusion 內的 model (U-Net 非 EMA)
        if 'model' in saved_checkpoint and isinstance(saved_checkpoint['model'], dict):
            # saved_checkpoint['model'] 是 GaussianDiffusion 的 state_dict
            # 我們需要從中提取 U-Net 的 state_dict
            # GaussianDiffusion 的 U-Net 實例名通常是 'model'
            # 所以 U-Net 的權重鍵會帶有 'model.' 前綴
            diffusion_state_dict = saved_checkpoint['model']
            unet_weights_to_load = {}
            for k, v in diffusion_state_dict.items():
                if k.startswith('model.'): # 'model.' 是 GaussianDiffusion.model (U-Net) 的前綴
                    unet_weights_to_load[k.replace('model.', '', 1)] = v
            if not unet_weights_to_load:
                print(f"錯誤：在 'model' (GaussianDiffusion state_dict) 中未找到 U-Net (前綴 'model.') 的權重。")
                return
            print("警告：未找到 EMA 權重，正在嘗試從 GaussianDiffusion 的 'model' 鍵中加載 U-Net (非EMA) 權重。")
        else:
            print(f"錯誤：模型文件 {model_path} 中未找到 'ema' 或 'ema_model' 或 'model' (包含U-Net權重) 鍵。")
            print(f"Checkpoint keys: {list(saved_checkpoint.keys())}")
            return

    try:
        model.load_state_dict(unet_weights_to_load)
        print("已成功加載 U-Net 權重。")
    except RuntimeError as e:
        print(f"錯誤：加載 U-Net state_dict 失敗。可能是模型架構不匹配或權重鍵名問題。")
        print(f"RuntimeError: {e}")
        # 打印一些鍵名幫助調試
        print("\n期望的 U-Net 模型鍵 (部分):")
        for i, key_name in enumerate(model.state_dict().keys()):
            if i < 5: print(f"- {key_name}")
            else: break
        print("\n加載的權重鍵 (部分):")
        for i, key_name in enumerate(unet_weights_to_load.keys()):
            if i < 5: print(f"- {key_name}")
            else: break
        return

    model.eval()
    # --- 3. 執行採樣 ---
    print(f"\n--- 開始 {args.strategy.upper()} 採樣 (步數: {args.sample_steps}" +
          (f", eta: {args.ddim_eta}" if args.strategy == 'ddim' else "") + ") ---")

    # GaussianDiffusion 的 sampling_timesteps 參數決定了實際採樣行為
    # 對於 DDPM，我們希望 sampling_timesteps 等於 timesteps (即 args.train_timesteps)
    # 對於 DDIM，sampling_timesteps 是 args.sample_steps (可以小於 args.train_timesteps)
    
    effective_sampling_timesteps_for_diffusion = args.sample_steps
    if args.strategy == 'ddpm':
        # 為了讓 GaussianDiffusion 執行 p_sample_loop (DDPM)，
        # 其內部的 is_ddim_sampling 必須為 False，
        # 這意味著 sampling_timesteps 參數必須等於其 timesteps 參數。
        effective_sampling_timesteps_for_diffusion = args.train_timesteps
        if args.sample_steps != args.train_timesteps:
            print(f"信息: 為了執行純DDPM採樣，GaussianDiffusion 的 sampling_timesteps 將設置為 {args.train_timesteps} (與訓練時一致)。")


    diffusion_sampler = GaussianDiffusion(
        model,
        image_size=args.image_size,
        timesteps=args.train_timesteps, # U-Net 訓練時的總擴散步長
        sampling_timesteps=effective_sampling_timesteps_for_diffusion, # 實際傳給 GaussianDiffusion 的採樣步數
        ddim_sampling_eta=args.ddim_eta if args.strategy == 'ddim' else 0.0, # eta 只在 DDIM 時相關
        objective=args.objective,
        beta_schedule=args.beta_schedule
    ).to(DEVICE)

    start_time = time.time()
    with torch.no_grad():
        # diffusion_sampler.sample() 會根據 is_ddim_sampling (內部由 timesteps 和 sampling_timesteps 比較得出)
        # 自動選擇 p_sample_loop 或 ddim_sample
        sampled_images = diffusion_sampler.sample(batch_size=args.batch_size)
    end_time = time.time()
    time_taken = end_time - start_time

    print(f"{args.strategy.upper()} 採樣完成。耗時: {time_taken:.4f} 秒")
    
    filename_parts = [
        args.strategy,
        f"steps{args.sample_steps}" # 文件名中仍然使用用戶指定的 sample_steps
    ]
    if args.strategy == 'ddim':
        filename_parts.append(f"eta{args.ddim_eta}")
    
    output_filename = "_".join(filename_parts) + ".png"
    save_path = output_strategy_folder / output_filename
    
    save_image(sampled_images, str(save_path), nrow=int(args.num_samples**0.5))
    print(f"{args.strategy.upper()} 生成圖像已保存到: {save_path}")

    print(f"\n採樣完成！結果保存在: {output_strategy_folder}")
    print(f"生成時間: {time_taken:.4f} 秒")

if __name__ == "__main__":
    main()