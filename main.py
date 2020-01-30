from cluster_posts import cluster_posts
from discover_silhouette import discover_silhouette
from discover_elbow import discover_elbow
from dump import dump_split, dump_pred
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Polyglot toolbox')

    subparsers = parser.add_subparsers(help='Commands')

    vocab_parser = subparsers.add_parser('vocab', help='build vocabulary')
    vocab_parser.add_argument('input', type=str)
    vocab_parser.set_defaults(func=build_vocab)

    sil_parser = subparsers.add_parser('discover-silhouette', help='discover k using the silhouettes heuristic')
    sil_parser.add_argument('input', type=str)
    sil_parser.add_argument('embed_model_path', type=str)
    sil_parser.add_argument('prefix', type=str, default='')
    sil_parser.add_argument('max_k', type=int, default=10)
    sil_parser.set_defaults(func=discover_silhouette_temp)

    sil_parser = subparsers.add_parser('discover-elbow', help='discover k using the elbow method')
    sil_parser.add_argument('input', type=str)
    sil_parser.add_argument('embed_model_path', type=str)
    sil_parser.add_argument('prefix', type=str, default='')
    sil_parser.add_argument('max_k', type=int, default=10)
    sil_parser.set_defaults(func=discover_elbow_temp)

    sil_parser = subparsers.add_parser('cluster-documents', help='cluster documents')
    sil_parser.add_argument('input', type=str)
    sil_parser.add_argument('embed_model_path', type=str)
    sil_parser.add_argument('prefix', type=str, default='')
    sil_parser.add_argument('K', type=int)
    sil_parser.set_defaults(func=cluster_temp)
    
    sil_parser = subparsers.add_parser(
        'dump-split', 
        help='assign the input documents to their respective clusters and dump each cluster out'
    )
    sil_parser.add_argument('input', type=str)
    sil_parser.add_argument('embed_model_path', type=str)
    sil_parser.add_argument('cluster_model_path', type=str)
    sil_parser.add_argument('prefix', type=str, default='')
    sil_parser.set_defaults(func=dump_split_tmp)

    sil_parser = subparsers.add_parser(
        'dump-pred', 
        help='assign the input documents to their respective clusters and produce 1 document with the cluster labels'
    )
    sil_parser.add_argument('input', type=str)
    sil_parser.add_argument('embed_model_path', type=str)
    sil_parser.add_argument('cluster_model_path', type=str)
    sil_parser.add_argument('dest', type=str, default='')
    sil_parser.set_defaults(func=dump_pred_tmp)

    return parser.parse_args()

def dump_pred_tmp(args):
    dump_pred(
        args.input,
        args.embed_model_path,
        args.cluster_model_path,
        args.dest
    )

def dump_split_tmp(args):
    dump_split(
        args.input,
        args.embed_model_path,
        args.cluster_model_path,
        args.prefix
    )

def cluster_temp(args):
    cluster_posts(
        args.input,
        args.embed_model_path,
        args.prefix,
        args.K
    )

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