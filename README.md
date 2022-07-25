# Synthesizing facial expression

In this project, we were trying to synthesize different facial expressions (anger, disgust, fear, happy, sad, surprised) on generated identities from [StyleGAN Model](https://github.com/NVlabs/stylegan2.git).

![stargan](https://github.com/sayeh1994/synthesizin_facial_expression/blob/main/images/stargan_affect_output.jpg)

The original code is from the [StarGAN - Official PyTorch Implementation](https://github.com/yunjey/stargan.git) but the model trained with [Affectnet-HQ dataset](https://www.kaggle.com/datasets/tom99763/affectnethq) instead of [RaFD dataset](http://www.socsci.ru.nl:8180/RaFD2/RaFD).

The code is written with tensorflow version 1. For training you can run the following command:

If you are running your code in google colab, you have to put these lines in a block before training:
```bash
%tensorflow_version 1.x
import tensorflow as tf
print(tf.__version__)
```

```bash
python main.py --mode='train' --dataset='RaFD' --c_dim=7 --image_size=256 \
                 --num_iters=200000 --resume_iters=1000 --num_iters_decay=100 --sample_step=1000 --model_save_step=1000 --rafd_image_dir='data/Affectnet-HQ'\
                 --sample_dir='stargan_affectnet/samples' --log_dir='stargan_affectnet/logs' \
                 --model_save_dir='stargan_affectnet/models' --result_dir='stargan_affectnet/results'
```

The pretrained model can be downloaded from [MyDrive](https://drive.google.com/drive/folders/1-3T9AKzD1CSK6UYzcg5xQXUR_Dpyj5zu?usp=sharing).
To test the trained model you either save all the results in one image with the following code by putting the test image in its own class of emotion in `rafd_image_dir='data/test'`:

```bash
python main.py --mode='test' --dataset='RaFD' --c_dim=7 --image_size=256 \
                 --test_iters=200000 --rafd_image_dir='data/test'\
                 --sample_dir='stargan_affectnet/samples' --log_dir='stargan_affectnet/logs' \
                 --model_save_dir='stargan_affectnet/models' --result_dir='stargan_affectnet/results'
```
or you can save the result in separated directory according to its own synthetic expression by the following command:

```bash
python main.py --mode='test_sep' --dataset='RaFD' --c_dim=7 --image_size=256 \
                 --test_iters=200000 --rafd_image_dir='data/test'\
                 --sample_dir='stargan_affectnet/samples' --log_dir='stargan_affectnet/logs' \
                 --model_save_dir='stargan_affectnet/models' --result_dir='stargan_affectnet/results'
```


