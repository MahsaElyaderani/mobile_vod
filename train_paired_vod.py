from path import parse_paths
from utils.resume import resume_defaults
import argparse
from engine.paired_trainer import add_training_arguments, locked_training
from paired_config import MemoryCorruption


def parse_args(argv=None, *, resume_payload=None):
    parser = argparse.ArgumentParser(description="Fine-tune two MobileNet branches with one shared native LSTM/SSD.")
    add_training_arguments(parser)
    parser.add_argument('--phase', choices=('joint_detection',), default='joint_detection',
                        help='Joint detection training; classification pretraining uses train_joint_pretrain.py.')
    parser.add_argument('--init-source', choices=('standalone_detection', 'joint_pretrain', 'fresh'),
                        default='standalone_detection',
                        help='Use a standalone detector, joint classifier, or fresh LSTM/heads.')
    parser.add_argument('--init-checkpoint', '--big-checkpoint', dest='big_checkpoint',
                        help='Initialization checkpoint; standalone_detection uses existing big stage2 weights.')
    parser.add_argument('--backbone', choices=('timm_mobilenet_v2',))
    parser.add_argument('--big-width-mult', type=float)
    parser.add_argument('--backbone-weights', help=argparse.SUPPRESS)
    parser.add_argument('--random-backbones', action='store_true', help='Fresh initialization only: skip pretrained weights.')
    parser.add_argument("--fast-init", choices=("random", "checkpoint", "standalone", "timm"), default=None)
    parser.add_argument("--fast-checkpoint", help="Backbone state_dict; native detector with --fast-init standalone.")
    parser.add_argument("--fast-width-mult", type=float, default=None)
    parser.add_argument("--fast-image-size", type=int, default=160)
    parser.add_argument("--small-only-state-policy", choices=("update", "hold"), default="update")
    parser.add_argument("--train-bn", action="store_true")
    parser.add_argument("--fast-bn-momentum", type=float,
                        help="Update only fast BN stats when shared/big BN stats are frozen (e.g. 0.01).")
    parser.add_argument("--reset-fast-bn", action="store_true",
                        help="Reset fast BN running stats once on a new run, never on resume.")
    parser.add_argument("--fast-learning-rate", type=float,
                        help="Separate LR for fast backbone and projection; shared/big use --learning-rate.")
    parser.add_argument("--rmsprop-eps", type=float, default=1.)
    parser.add_argument("--diagnostic-validation", action="store_true",
                        help="Also evaluate big-only, small-only, small zero-memory, and fixed without memory.")
    parser.add_argument("--freeze-big-in-joint", action="store_true",
                        help="During joint training freeze big backbone, shared LSTM/SSD and their BN stats; train only fast backbone/adapter on small frames, with only big frames committing memory.")
    parser.add_argument("--memory-corruption-probability", type=float, default=0.)
    parser.add_argument("--memory-dropout", type=float, default=.1)
    parser.add_argument("--memory-noise-std", type=float, default=.1)
    args = resume_defaults(parser, parse_paths(parser, "train_paired_vod", argv), argv, resume_payload)
    del args.phase  # Keep checkpoint arguments compatible with existing detector runs.
    if args.fast_image_size != 160:
        parser.error('The small branch always uses 160x160.')
    initial = None
    if not args.resume and args.big_checkpoint and (args.fast_init is None or args.fast_width_mult is None
            or args.backbone is not None or args.big_width_mult is not None or args.init_source == "joint_pretrain"):
        from utils.runtime import load_checkpoint
        initial = load_checkpoint(args.big_checkpoint)
        cfg = initial.get('model_config', {})
        if args.backbone is not None and args.backbone != cfg.get('backbone'):
            parser.error('--backbone differs from the initialization checkpoint')
        if args.big_width_mult is not None and args.big_width_mult != cfg.get('width_mult'):
            parser.error('--big-width-mult differs from the initialization checkpoint')
        if args.init_source == 'joint_pretrain' and args.fast_width_mult is None:
            args.fast_width_mult = initial.get('paired_config', {}).get('fast_width_mult')
    if args.init_source == 'fresh' and not args.resume and (args.big_checkpoint or args.fast_checkpoint or args.fast_init):
        parser.error('Fresh initialization uses --backbone/--big-width-mult/--fast-width-mult, not detector checkpoints or --fast-init')
    if args.init_source == 'joint_pretrain' and not args.resume and (args.fast_checkpoint or args.fast_init):
        parser.error('joint_pretrain supplies both backbones; do not override --fast-init/--fast-checkpoint')
    if args.random_backbones and args.init_source != 'fresh':
        parser.error('--random-backbones is only valid with --init-source fresh')
    if args.fast_init is None:
        args.fast_init = 'timm'
    if args.fast_width_mult is None:
        args.fast_width_mult = .5
    MemoryCorruption(args.memory_corruption_probability, args.memory_dropout, args.memory_noise_std)
    return args


if __name__ == "__main__":
    locked_training(parse_args())
