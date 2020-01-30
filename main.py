from discover_silhouette import discover_silhouette
from discover_elbow import discover_elbow
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Polyglot toolbox')

    vocab_sub = parser.add_subparsers(help='Commands')

    vocab_parser = vocab_sub.add_parser('vocab', help='build vocabulary')
    vocab_parser.add_argument('input', type=str)
    vocab_parser.set_defaults(func=build_vocab)

    sil_parser = vocab_sub.add_parser('discover-silhouette', help='discover silhouettes')
    sil_parser.add_argument('input', type=str)
    sil_parser.add_argument('embed_model_path', type=str)
    sil_parser.add_argument('prefix', type=str, default='')
    sil_parser.add_argument('max_k', type=int, default=10)
    sil_parser.set_defaults(func=discover_silhouette_temp)

    sil_parser = vocab_sub.add_parser('discover-elbow', help='discover silhouettes')
    sil_parser.add_argument('input', type=str)
    sil_parser.add_argument('embed_model_path', type=str)
    sil_parser.add_argument('prefix', type=str, default='')
    sil_parser.add_argument('max_k', type=int, default=10)
    sil_parser.set_defaults(func=discover_elbow_temp)

    # parser.add_argument(
    #     '--dump-split',
    #     action='store',
    #     dest='dump_split',
    #     type=str,
    #     default=None
    # )

    # parser.add_argument(
    #     '--dump-pred',
    #     action='store',
    #     dest='dump_pred',
    #     type=str,
    #     default=None
    # )

    return parser.parse_args()

def build_vocab(args):
    with open(args.input) as handle:
        for new_line in handle:
            for token in new_line.split():
                sys.stdout.write(token)
                sys.stdout.write(' ')

def discover_silhouette_temp(args):
    discover_silhouette(
        args.input,
        args.embed_model_path,
        args.prefix,
        args.max_k
    )

def discover_elbow_temp(args):
    discover_elbow(
        args.input,
        args.embed_model_path,
        args.prefix,
        args.max_k
    )

if __name__ == '__main__':
    args = parse_args()

    args.func(args)