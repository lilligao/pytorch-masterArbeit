# from mmseg.models import MixVisionTransformer
import torch
import time
import os.path as osp
import mmcv
from mmengine.config import Config
#from mmseg.utils import collect_env, get_root_logger

# self = MixVisionTransformer(in_channels=1)
# self.eval()
# inputs = torch.rand(1, 1, 32, 32)
# level_outputs = self.forward(inputs)
# for level_out in level_outputs:
#     print(tuple(level_out.shape))


cfg = Config.fromfile('./lib/SegFormer/local_configs/segformer/B1/segformer.b1.512x512.ade.160k.py')

# set cudnn_benchmark
if cfg.get('cudnn_benchmark', False):
    torch.backends.cudnn.benchmark = True

# Set up working dir to save files and logs.
cfg.work_dir = './checkpoints'



# create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
# dump config
cfg.dump(osp.join(cfg.work_dir, osp.basename(cfg.work_dir)))
# init the logger before other steps
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
#logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

# # init the meta dict to record some important information such as
# # environment info and seed, which will be logged
# meta = dict()
# # log env info
# env_info_dict = collect_env()
# env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
# dash_line = '-' * 60 + '\n'
# logger.info('Environment info:\n' + dash_line + env_info + '\n' +
#             dash_line)
# meta['env_info'] = env_info

# # log some basic info
# logger.info(f'Distributed training: {distributed}')
# logger.info(f'Config:\n{cfg.pretty_text}')

# # set random seeds
# if args.seed is not None:
#     logger.info(f'Set random seed to {args.seed}, deterministic: '
#                 f'{args.deterministic}')
#     set_random_seed(args.seed, deterministic=args.deterministic)
# cfg.seed = args.seed
# meta['seed'] = args.seed
# meta['exp_name'] = osp.basename(args.config)

# model = build_segmentor(
#     cfg.model,
#     train_cfg=cfg.get('train_cfg'),
#     test_cfg=cfg.get('test_cfg'))

# logger.info(model)

# datasets = [build_dataset(cfg.data.train)]

# if len(cfg.workflow) == 2:
#     val_dataset = copy.deepcopy(cfg.data.val)
#     val_dataset.pipeline = cfg.data.train.pipeline
#     datasets.append(build_dataset(val_dataset))
# if cfg.checkpoint_config is not None:
#     # save mmseg version, config file content and class names in
#     # checkpoints as meta data
#     cfg.checkpoint_config.meta = dict(
#         mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
#         config=cfg.pretty_text,
#         CLASSES=datasets[0].CLASSES,
#         PALETTE=datasets[0].PALETTE)
# # add an attribute for visualization convenience
# model.CLASSES = datasets[0].CLASSES
# train_segmentor(
#     model,
#     datasets,
#     cfg,
#     distributed=distributed,
#     validate=(not args.no_validate),
#     timestamp=timestamp,
#     meta=meta)
