import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--name', type=str, default="ConvMixer")

# data
parser.add_argument('--batch-size', default=512, type=int)
parser.add_argument('--scale', default=0.75, type=float)
parser.add_argument('--reprob', default=0.25, type=float)
parser.add_argument('--ra-m', default=8, type=int)
parser.add_argument('--ra-n', default=1, type=int)
parser.add_argument('--jitter', default=0.1, type=float)

# arch
parser.add_argument('--hdim', default=256, type=int)
parser.add_argument('--depth', default=8, type=int)
parser.add_argument('--psize', default=2, type=int)
parser.add_argument('--conv-ks', default=5, type=int)
parser.add_argument('--clone_type', type=str, choices=['clone', 'augment', 'no'])
parser.add_argument('--sparsity', type=float, default=0.5)
parser.add_argument('--model_type', type=str, default='regular', choices=['mask', 'regular'])
parser.add_argument('--bn_type', type=str, default='learn', choices=['learn', 'not-learn'])
parser.add_argument('--act', type=str, default='gelu')
parser.add_argument('--MP_RP_ratio', type=float, default=1.0)
parser.add_argument('--MP_RP_mode', type=str, default='copy', choices=['rand', 'copy'])
parser.add_argument('--rescale_var', action='store_true')

# opt
parser.add_argument('--wd', default=0.01, type=float)
parser.add_argument('--clip-norm', action='store_true')
parser.add_argument('--epochs', default=25, type=int)
parser.add_argument('--lr-max', default=0.01, type=float)
parser.add_argument('--workers', default=2, type=int)

# save
parser.add_argument('--save', action='store_true')
parser.add_argument('--save_file', type=str, default='test.csv')
parser.add_argument('--saved_model_path', type=str)
parser.add_argument('--save_start', action="store_true")

# mode
parser.add_argument('--pipe', type=str, choices=['train', 'test'])
parser.add_argument('--parallel', action="store_true")
parser.add_argument('--exp_id', type=int, default=1)

# dataset
parser.add_argument('--dataset', type=str, default='cf10', choices=['cf10', 'cf100', 'tiny'])
parser.add_argument('--num_class', type=int, default=10, choices=[10, 100, 200])

args = parser.parse_args()
