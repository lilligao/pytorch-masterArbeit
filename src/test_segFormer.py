from mmseg.apis import init_model

config_path = './src/segformer.b1.512x512.ade.160k.py'
#checkpoint_path = 'checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

# 初始化不带权重的模型
model = init_model(config_path)