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

    parser.add_argument('--data_path', dest='data_path', default="./samples/training",help='Path of dataset directory for train model')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--resume_epoch',default=0, type=int)

    parser.add_argument('--train', default=True, help='Train or inference')
    parser.add_argument('--warm', default=False, help='Train or inference')
    parser.add_argument('--root_path', default="./checkpoints", help='Train or inference')
    parser.add_argument("--save_path", type=str, default="./samples/output/best.pth",help="path to save trained Sound2Scene")

    return parser.parse_args()

def showImage(args,generator,emb,gt_emb,img):
    output=None
    z = torch.empty(16,generator.dim_z).normal_(mean=0, std=args.z_var)

    emb /= torch.linalg.norm(emb, dim=-1, keepdims=True)
    gt_emb /= torch.linalg.norm(gt_emb, dim=-1, keepdims=True)

    gen_emb = generator(z.cuda(), None, emb[:16])
    gen_emb = torch.clamp(gen_emb,-1., 1.)
    gen_gt = generator(z.cuda(), None, gt_emb[:16])
    gen_gt = torch.clamp(gen_gt, -1., 1.)

    output = torch.cat((img[:8].squeeze(1), gen_gt[:8]),0)
    output = torch.cat((output,gen_emb[:8]),0)

    output = torch.cat((output,img[8:16].squeeze(1)),0)
    output = torch.cat((output, gen_gt[8:16]), 0)
    output = torch.cat((output, gen_emb[8:16]), 0)
    output = make_grid(output, normalize=True, scale_each=True, nrow=8)
    return output


def load(args, device):
    model = AVENet(args).to(device)
    if args.warm:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        for index,child in enumerate(model.children()):
            if index==0:
                num_ftrs=child.fc.in_features
                child.fc = nn.Linear(num_ftrs, 256).to(device)

        print("load_warm_start")
    return model

def load_dataset(args):
    train_dataset = GetVGGSound(args.data_path)
    test_dataset = GetVGGSound(args.data_path)

    return train_dataset, test_dataset



def validate(args, pipe, model, test_loader, device):
    dtype = next(pipe.image_encoder.parameters()).dtype
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = InfoNCE_with_L2(device)
    total_loss = torch.Tensor([0])
    for step, (index, spec, emb_img, orig_img) in enumerate(test_loader):
        with torch.no_grad():
            gt_emb = emb_img.to(device=device, dtype=dtype).squeeze(1)
            gt_emb = pipe.image_encoder(gt_emb).image_embeds

        spec = Variable(spec).to(device)
        _, audio_emb = model(spec.unsqueeze(1).float())
        loss = criterion.loss_fn(audio_emb, gt_emb.float())
        total_loss+=loss.item()
    return torch.mean(total_loss)

def train(args):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load(args, device)
    #define LDM
    ### -- Model -- ###
    # Start the StableUnCLIP Image variations pipeline
    pipe = StableUnCLIPImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variation="fp16")
    pipe = pipe.to("cuda")
    dtype = next(pipe.image_encoder.parameters()).dtype

    criterion = InfoNCE_with_L2_sound2vision(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    #define an dataset
    train_dataset, test_dataset = load_dataset(args)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    min_loss = torch.tensor(100000).detach().cpu()


    for epoch in range(args.epochs):
        model.train()
        epoch_train_loss = torch.tensor(0)
        epoch_iter = 0

        for step, (index, spec, emb_img, orig_img) in enumerate(train_loader):

            with torch.no_grad():
                gt_emb = emb_img.to(device=device, dtype=dtype).squeeze(1)
                gt_emb = pipe.image_encoder(gt_emb).image_embeds

            spec = Variable(spec).to(device)
            _, audio_emb = model(spec.unsqueeze(1).float())
            loss = criterion.loss_fn(audio_emb, gt_emb.float())

            # To delete VAE, you have to remove KLD term and mu, logvar
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_iter = epoch_iter + 1
            epoch_train_loss=epoch_train_loss+loss

            print('[epoch: %d] iter_cosine_loss: %.3f' % (epoch + 1, loss.detach().item()), end='\r')

        model.eval()
        val_loss = validate(args, pipe, model, test_loader, device)
        if val_loss<min_loss:
            min_loss=val_loss
            torch.save(model.state_dict(), args.save_path)
        model.train()

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
    train(args)

if __name__=='__main__':
    main()
