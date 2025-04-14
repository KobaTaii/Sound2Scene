# Sound2Scene (CVPR 2023) and Sound2Vision (Arxiv 2024)

### [Project Page](https://sound2scene.github.io/) | [Paper](https://openaccess.thecvf.com/content/CVPR2023/html/Sung-Bin_Sound_to_Visual_Scene_Generation_by_Audio-to-Visual_Latent_Alignment_CVPR_2023_paper.html) | [Extension Paper](https://arxiv.org/abs/2412.06209)
This repository contains a pytorch implementation for the CVPR 2023 paper, [Sound2Scene: Sound to visual scene generation by audio-to-visual latent alignment](https://openaccess.thecvf.com/content/CVPR2023/html/Sung-Bin_Sound_to_Visual_Scene_Generation_by_Audio-to-Visual_Latent_Alignment_CVPR_2023_paper.html) (V1), and its extended paper, [Sound2Vision](https://arxiv.org/abs/2412.06209) (V2). Sound2Scene and Sound2Vision are sound-to-image generative model which is trained solely from unlabeled videos to generate images from sound.<br><br>

#### Results from Sound2Scene (CVPR 2023)
![teaser1](https://github.com/postech-ami/Sound2Scene/assets/59387731/9c1a2d37-38e0-4525-9dc2-74002ee4c2e2)

#### Results from Sound2Vision (Arxiv 2024) - environmental sound as an input
![image](https://github.com/user-attachments/assets/021707bb-0191-449c-a26d-c1b31f464276)

#### Results from Sound2Vision (Arxiv 2024) - speech as an input
![image](https://github.com/user-attachments/assets/8eeae67a-2cc0-474f-8740-83b411007b00)


## Sound2Scene (CVPR 2023)
### Getting started
This code was developed on Ubuntu 18.04 with Python 3.8, CUDA 11.1 and PyTorch 1.8.0. Later versions should work, but have not been tested.

### Installation 
Create and activate a virtual environment to work in.
```
conda create --name sound2scene python=3.8.8
conda activate sound2scene
```
Install [PyTorch](https://pytorch.org/). For CUDA 11.1, this would look like:
```
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Install the remaining requirements with pip:
```
pip install -r requirements.txt
```

### Download Models
To run Sound2Scene, you need to download image encoder (SWAV), image decoder (BigGAN) and Sound2Scene model.
Download [Sound2Scene](https://drive.google.com/file/d/1MfQo9Y6cBwSo9sYkwj2gG9kNa_1fuaUJ/view?usp=sharing) | [SWAV](https://drive.google.com/file/d/1_DjU6MBZwQTQzNdlktr12eUZszaRvPX5/view?usp=sharing) | [BigGAN](https://drive.google.com/drive/folders/1nlpQ-D2zQNlEWDOKidOV-p4Ny26KHvlb?usp=sharing).

After downloading the models, place them in ./checkpoints.
```
./checkpoints/icgan_biggan_imagenet_res128
./checkpoints/sound2scene.pth
./checkpoints/swav.pth.tar
```

### Highly correlated audio-visual pair dataset
We provide the annotations of the highly correlated audio-visual pairs from the VGGSound dataset.

Download [top1_boxes_top10_moments.json](https://drive.google.com/file/d/1uFht0YV8al9RqMPR2Umn99xWPluOU-UQ/view?usp=drive_link)

The annotation file contains each video name with the corresponding top 10 audio-visually correlated frame numbers.

```
{'9fhhMaXTraI_44000_54000': [47, 46, 45, 23, 42, 9, 44, 56, 27, 17],
'G_JwMzRLRNo_252000_262000': [2, 1, 26, 29, 15, 16, 11, 3, 14, 23], ...}

# 9fhhMaXTraI_44000_54000: video name
# [47, 46, 45, 23, 42, 9, 44, 56, 27, 17]: frame number (e.g., 47th, 46th frame, ...)
# 47th frame is the highest audio-visually correlated frame
```

Please follow the steps below to select a highly correlated audio-visual pair dataset.

**(Step 1)** Download the training dataset from [VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/).

**(Step 2)** Extract the frames of each video in 10 fps.

**(Step 3)** Select the frame that is mentioned in the annotation file.

If you find this dataset helpful, please consider also citing: 
[Less Can Be More: Sound Source Localization With a Classification Model](https://openaccess.thecvf.com/content/WACV2022/html/Senocak_Less_Can_Be_More_Sound_Source_Localization_With_a_Classification_WACV_2022_paper.html).

The VEGAS dataset is available [here](https://drive.google.com/file/d/1ah2s3m96Nz0MUQX9Z9i4pRsqH89rTtu4/view?usp=drive_link).

### Training Sound2Scene
Run below command to train the model.

We provide sample image and audio pairs in **./samples/training**.

The samples are for checking the training code.

For the full dataset, please download the training dataset from [VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/) or VEGAS.

Although we provide the categories which we used ([category_list](https://github.com/postech-ami/Sound2Scene/blob/main/samples/categories.txt)), no category information were used for training.
```
python train.py --data_path [path containing image and audio pairs] --save_path [path for saving the checkpoints]

#or

bash train.sh
```

### Inference
```
bash test.sh
```

### Evaluating Sound2Scene
(1) We used off-the-shelf CLIP model (``Vit-B/32'') to evaluate R@k performance.

(2) We trained the [Inception model](https://drive.google.com/file/d/1GbZ25SShTssQ-G5Ynhzsjwz6QkPWWQNm/view?usp=drive_link) on VGGSound for measuring FID and Inception score.


## Sound2Vision (Arxiv 2024)
### Getting started
This code was developed on Ubuntu 18.04 with Python 3.8, CUDA 11.1 and PyTorch 1.9.1. Later versions should work, but have not been tested.

### Installation 
Create and activate a virtual environment to work in.
```
conda create --name sound2scene python=3.8.8
conda activate sound2scene
```
Install [PyTorch](https://pytorch.org/). For CUDA 11.1, this would look like:
```
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

Install the remaining requirements with pip:
```
pip install -r requirements.txt
pip install openai-clip==1.0.1
pip install transformers==4.28.1
pip install diffusers==0.15.0
pip install git+https://github.com/wenet-e2e/wespeaker.git
```

### Download Models

To run Sound2Vision, you need to download Sound2Vision_env, and Sound2Vision_face.
Sound2Vision_env is the model that is trained on the VGGSound dataset (environmental sound) and Sound2Vision_face is the model that is trained on the CelebV-HQ dataset (face-speech).

Download [Sound2Vision_env](https://drive.google.com/file/d/1npdSAOHINI1MU5I365ylMl151GmmKkxB/view?usp=sharing) | [Sound2Vision_face](https://drive.google.com/file/d/1_klp5z8IWn0eZyh-nqX5KEoQlfLDcVEp/view?usp=sharing)

After downloading the models, place them in ./checkpoints.
```
./checkpoints/sound2vision_env.pth
./checkpoints/sound2vision_face.pth
```

### Training Sound2Vision
Run below command to train the model.

We provide sample image and audio pairs in **./samples/training**.

The samples are for checking the training code.
```
python train_sound2vision.py --data_path [path containing image and audio pairs] --save_path [path for saving the checkpoints]
```

### Inference
```
#for generating images from environmental sound
pyton test_sound2vision.py --ckpt_path ./checkpoints/sound2vision_env.pth --wav_path ./samples/inference --output_path ./samples/output --input_data env

#for generating human face images from speech
pyton test_sound2vision.py --ckpt_path ./checkpoints/sound2vision_face.pth --wav_path ./samples/inference_face --output_path ./samples/output --input_data face
```


## Citation
If you find our code or paper helpful, please consider citing:
````BibTeX
@inproceeding{sung2023sound,
  author    = {Sung-Bin, Kim and Senocak, Arda and Ha, Hyunwoo and Owens, Andrew and Oh, Tae-Hyun},
  title     = {Sound to Visual Scene Generation by Audio-to-Visual Latent Alignment},
  booktitle   = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2023}
}
````

````BibTeX
@article{sung2024sound2vision,
  title={Sound2Vision: Generating Diverse Visuals from Audio through Cross-Modal Latent Alignment},
  author={Sung-Bin, Kim and Senocak, Arda and Ha, Hyunwoo and Oh, Tae-Hyun},
  journal={arXiv preprint arXiv:2412.06209},
  year={2024}
}
````

## Acknowledgment
This work was supported by IITP grant funded by Korea government (MSIT) (No.2021-0-02068, Artificial Intelligence Innovation Hub; No.2022-0-00124, Development of Artificial Intelligence Technology for Self-Improving Competency-Aware Learning Capabilities). The GPU resource was supported by the HPC Support Project, MSIT and NIPA.

The implementation of Sound2Scene borrows most of the codebases from the seminal prior work, [ICGAN](https://github.com/facebookresearch/ic_gan) and [VGGSound](https://github.com/hche11/VGGSound).
We thank the authors of these work who made their code public. Also If you find our work helpful, please consider citing them as well.
