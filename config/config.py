from yacs.config import CfgNode as CN

__C = CN()
cfg = __C

# basic parameters
__C.out_scale = 0.001
__C.exemplar_sz = 127
__C.instance_sz = 255
__C.context = 0.5

# inference parameters
__C.scale_num = 3
__C.scale_step = 1.0375
__C.scale_lr = 0.59
__C.scale_penalty = 0.9745
__C.window_influence = 0.176
__C.response_sz = 17
__C.response_up = 16
__C.total_stride = 8

# train parameters
__C.epochs = 200
__C.batch_size = 8
__C.eval_freq = 3
__C.num_workers = 10
__C.initial_lr = 1.0e-2
__C.ultimate_lr = 1.0e-5
__C.weight_decay = 5.0e-4
__C.momentum = 0.9
__C.r_pos = 16
__C.r_neg = 0

# Augmentations
# train.template
__C.train = CN()
__C.train.template = CN()
__C.train.template.clahe = True
__C.train.template.flip = 0.3
# train.search
__C.train.search = CN()
__C.train.search.clahe = True
__C.train.search.flip = 0.0

__C.test = CN()
__C.test.template = CN()
__C.test.template.clahe = __C.train.template.clahe
