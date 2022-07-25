# Synthesizing facial expression

In this project, we were trying to synthesize different facial expressions (anger, disgust, fear, happy, sad, surprised) on generated identities from [StyleGAN Model](https://github.com/NVlabs/stylegan2.git).

![stargan]()

The original code is from the [StarGAN - Official PyTorch Implementation](https://github.com/yunjey/stargan.git) but the model trained with [Affectnet-HQ dataset](https://www.kaggle.com/datasets/tom99763/affectnethq) instead of [RaFD dataset](http://www.socsci.ru.nl:8180/RaFD2/RaFD).

For training you can run the following command:

```bash
python main.py --mode='train' --dataset='RaFD' --c_dim=7 --image_size=256 \
                 --num_iters=200000 --resume_iters=1000 --num_iters_decay=100 --sample_step=1000 --model_save_step=1000 --rafd_image_dir='data/Affectnet-HQ'\
                 --sample_dir='stargan_affectnet/samples' --log_dir='stargan_affectnet/logs' \
                 --model_save_dir='stargan_affectnet/models' --result_dir='stargan_affectnet/results'
```



