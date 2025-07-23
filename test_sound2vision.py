import os
os.environ["OMP_NUM_THREADS"] = "1"
import torch
from torch.optim import *
from torchvision.transforms import *
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import argparse
from model import AVENet
from a2s_dataloader import GetVGGSound
from torchvision.utils import save_image,make_grid
from loss import *
import random

from src.diffusers.pipelines.stable_diffusion.pipeline_stable_unclip_img2img import StableUnCLIPImg2ImgPipeline
import wespeaker
from scipy import signal
import soundfile as sf
import numpy as np

class MAPPING(nn.Module):
    def __init__(self):
        super(MAPPING, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(256, 1024)

    def forward(self, audio):
        x = self.fc1(audio)

        return x


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pool',
        default="avgpool",
        type=str,
        help= 'either vlad or avgpool')
    parser.add_argument(
        '--batch_size',
        default=64,
        type=int,
        help='Batch Size')
    parser.add_argument(
        '--n_classes',
        default=1024,
        type=int,
        help=
        'Number of classes')
    parser.add_argument(
        '--model_depth',
        default=18,
        type=int,
        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument(
        '--resnet_shortcut',
        default='B',
        type=str,
        help='Shortcut type of resnet (A | B)')
    parser.add_argument(
        '--resnet_type',
        default='fc2',
        type=str)

    parser.add_argument('--train', default=True, help='Train or inference')
    parser.add_argument('--warm', default=False, help='Train or inference')
    parser.add_argument("--ckpt_path", type=str, default="/node_data/sung/audio2scene/dataset/a2s2s/vggsound_log/logs/ldm/1126_FACE_FC/ckpt/9.pth",help="path to save trained Sound2Scene")
    parser.add_argument("--input_data", type=str, default="face",help="choose one between env or face")
    parser.add_argument("--wav_path", default="./samples/inference",type=str)
    parser.add_argument("--out_path", default="./samples/output", type=str)

    return parser.parse_args()


def showImage(args,pipe,emb, device):
    layout = torch.strided
    shape = (1, 4, 96, 96)
    rand_device = pipe._execution_device
    dtype = next(pipe.image_encoder.parameters()).dtype
    latents = torch.randn(shape, generator=None, device=rand_device, dtype=dtype, layout=layout).to(device)

    image = pipe(emb.to(device=device, dtype=dtype), latents=latents).images
    return image[0]


def audio2spectrogra(samples, samplerate):
    nperseg = 512  # スペクトログラムに必要な最小の長さ

    # --- ▼▼ ここからが重要な修正 ▼▼ ---
    # もし入力サンプルの長さが、必要な長さ(512)より短い場合
    if len(samples) < nperseg:
        # 512を超えるまで、何回繰り返す必要があるか計算する
        # 例: lenが20なら、512 // 20 + 1 = 25 + 1 = 26回繰り返す
        repeats = (nperseg // len(samples)) + 1
        resamples = np.tile(samples, repeats)
    else:
        # 長さが十分な場合は、そのまま使う
        resamples = samples

    # 元のコードの意図を汲み、長すぎる場合は16万サンプルに制限する
    resamples = resamples[:160000]
    # --- ▲▲ ここまでが重要な修正 ▲▲ ---

    # クリッピング処理（これは元のままでOK）
    resamples[resamples > 1.] = 1.
    resamples[resamples < -1.] = -1.

    # これでresamplesは必ず512以上の長さになるため、エラーは発生しない
    frequencies, times, spectrogram = signal.spectrogram(resamples, samplerate, nperseg=512, noverlap=353)

    spectrogram = np.log(spectrogram + 1e-7)
    mean = np.mean(spectrogram)
    std = np.std(spectrogram)
    spectrogram = np.divide(spectrogram - mean, std + 1e-9)

    spectrogram = torch.from_numpy(spectrogram).unsqueeze(0)
    return spectrogram

import os
import soundfile as sf
import numpy as np # numpyをインポート
from torch.autograd import Variable

def test_env(args, model, pipe, device):
    audio_paths = args.wav_path
    save_path = args.out_path
    os.makedirs(save_path, exist_ok=True)

    for audio_filename in os.listdir(audio_paths):
        audio = os.path.join(audio_paths, audio_filename)

        try:
            samples, samplerate = sf.read(audio)

            # --- ▼▼ これが最後の修正です ▼▼ ---
            # もしデータが2次元（ステレオ）なら、平均をとって1次元（モノラル）に変換
            if samples.ndim == 2:
                samples = np.mean(samples, axis=1)
            # --- ▲▲ ここまで ▲▲ ---

            if len(samples) < 512:
                print(f"SKIPPING: '{audio_filename}' is too short ({len(samples)} samples).")
                continue
        except Exception as e:
            print(f"SKIPPING: Could not read '{audio_filename}'. Reason: {e}")
            continue

        spectrogram = audio2spectrogra(samples, samplerate)

        spectrogram = Variable(spectrogram).to(device)
        _, emb = model(spectrogram.unsqueeze(1).float())

        output = showImage(args, pipe, emb, device)
        save_name = audio.split("/")[-1].split(".")[0]
        save_final = os.path.join(save_path, save_name + ".png")
        output.save(save_final)

def test_face(args, model, aud_model, pipe, device):
    audio_paths = "/local_data/sung/audio2scene/a2s_extension/240410_face_env/face_audio"
    save_path = args.out_path
    os.makedirs(save_path, exist_ok=True)

    for audio in os.listdir(audio_paths):
        audio = os.path.join(audio_paths, audio)
        emb = model(aud_model.extract_embedding(audio).cuda())
        output = showImage(args, pipe, emb.unsqueeze(0), device)
        save_name = audio.split("/")[-1].split(".")[0]
        save_final = os.path.join(save_path, save_name + ".png")
        output.save(save_final)

def main():
    random_seed=1234
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    np.random.seed(random_seed)
    random.seed(random_seed)

    args = get_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # Start the StableUnCLIP Image variations pipeline
    pipe = StableUnCLIPImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip",torch_dtype=torch.float16, variation="fp16")
    pipe = pipe.to("cuda")

    if args.input_data=="env":
        checkpoint = torch.load(args.ckpt_path, map_location=device)
        model = AVENet(args).to(device)
        model.load_state_dict(checkpoint, strict=False)
        model.eval()
        test_env(args, model, pipe, device)

    elif args.input_data=="face":
        face_checkpoint = torch.load(args.ckpt_path, map_location=device)
        model = MAPPING()
        model.load_state_dict(face_checkpoint, strict=False)
        model.cuda()
        aud_model = wespeaker.load_model('english')
        test_face(args, model, aud_model, pipe, device)


if __name__=='__main__':
    main()
