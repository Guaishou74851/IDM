import torch, os, glob, cv2, random
import numpy as np
from argparse import ArgumentParser
from model import Net
from utils import *
from skimage.metrics import structural_similarity as ssim
from time import time
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("--epoch", type=int, default=50)
parser.add_argument("--step_number", type=int, default=3)
parser.add_argument("--cs_ratio", type=float, default=0.1)
parser.add_argument("--block_size", type=int, default=32)
parser.add_argument("--model_dir", type=str, default="weight")
parser.add_argument("--data_dir", type=str, default="data")
parser.add_argument("--testset_name", type=str, default="Set11")
parser.add_argument("--result_dir", type=str, default="result")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 2025
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

epoch = args.epoch
T = args.step_number
B = args.block_size
ratio = args.cs_ratio
print("cs ratio =", ratio)

N = B * B
q = int(np.ceil(ratio * N))

U, S, V = torch.linalg.svd(torch.randn(N, N, device=device))
Phi = (U @ V)[:, :q]

from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("sd-legacy/stable-diffusion-v1-5").to(device)
model = Net(T, pipe.unet).to(device)

model_dir = "./%s/R_%.2f_T_%d_B_%d" % (args.model_dir, ratio, T, B)
model.load_state_dict(torch.load("./%s/net_params_%d.pkl" % (model_dir, epoch)))

test_image_paths = glob.glob(os.path.join(args.data_dir, args.testset_name) + "/*")

with torch.no_grad():
    PSNR_list, SSIM_list = [], []
    result_dir = os.path.join(args.result_dir, args.testset_name, str(int(ratio * 100)))
    os.makedirs(result_dir, exist_ok=True)
    for i, path in enumerate(test_image_paths):
        test_image = cv2.cvtColor(cv2.imread(path, 1), cv2.COLOR_BGR2YCrCb)
        img, old_h, old_w, img_pad, new_h, new_w = my_zero_pad(test_image[:,:,0], block_size=B)
        img_pad = img_pad.reshape(1, 1, new_h, new_w) / 255.0
        x = torch.from_numpy(img_pad).to(device).float()
        perm = torch.randperm(new_h * new_w, device=device)
        perm_inv = torch.empty_like(perm)
        perm_inv[perm] = torch.arange(perm.shape[0], device=device)
        A = lambda z: (z.reshape(-1,)[perm].reshape(-1,N) @ Phi)
        AT = lambda z: (z @ Phi.t()).reshape(-1,)[perm_inv].reshape(1,1,new_h,new_w)
        y = A(x)
        x_out = model(y, A, AT, use_amp_=False)[:old_h, :old_w]
        x_out = (x_out.clamp(min=0.0, max=1.0) * 255.0).cpu().numpy().squeeze()
        PSNR = psnr(x_out, img)
        SSIM = ssim(x_out, img, data_range=255)
        PSNR_list.append(PSNR)
        SSIM_list.append(SSIM)
        print("[%d/%d] %s, PSNR: %.2f, SSIM: %.4f" % (i+1, len(test_image_paths), path, PSNR, SSIM))
        test_image[:,:,0] = x_out
        test_image = cv2.cvtColor(test_image, cv2.COLOR_YCrCb2BGR).astype(np.uint8)
        result_path = os.path.join(result_dir, path.split("/")[-1])
        cv2.imwrite("%s_PSNR_%.2f_SSIM_%.4f.png" % (result_path, PSNR, SSIM), test_image)
    print("Average PSNR: %.2f" % np.mean(PSNR_list))
    print("Average SSIM: %.4f" % np.mean(SSIM_list))