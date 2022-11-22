import argparse
import numpy as np
np.random.seed(0)
from ruamel.yaml import YAML
import os
from models import MGDCR

def get_args(model_name, dataset, custom_key="", yaml_path=None) -> argparse.Namespace:
    yaml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "args.yaml")
    custom_key = custom_key.split("+")[0]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=model_name)
    parser.add_argument("--custom-key", default=custom_key)
    parser.add_argument("--dataset", default=dataset)
    parser.add_argument('--cfg', type=int, default=[512, 128], help='hidden dimension')
    parser.add_argument('--sc', type=float, default=3.0, help='GCN self connection')
    parser.add_argument('--sparse', type=bool, default=True, help='sparse adjacency matrix')
    parser.add_argument('--iterater', type=int, default=10, help='iterater')
    parser.add_argument('--use_pretrain', type=bool, default=True, help='use_pretrain')
    parser.add_argument('--nb_epochs', type=int, default=1500, help='the number of epochs')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--patience', type=int, default=20, help='patience for early stopping')
    parser.add_argument('--gpu_num', type=int, default=5, help='the id of gpu to use')
    parser.add_argument('--seed', type=int, default=0, help='the seed to use')
    parser.add_argument('--test_epo', type=int, default=50, help='test_epo')
    parser.add_argument('--test_lr', type=int, default=0.3, help='test_lr')
    parser.add_argument('--dropout', type=int, default=0.2, help='dropout')
    parser.add_argument('--save_root', type=str, default="./saved_model", help='root for saving the model')
    parser.add_argument('--isSemi', action='store_true', default=False)
    parser.add_argument('--lambdintra', type=int, default=0.01, help='lamdaintra')
    parser.add_argument('--lambdinter', type=int, default=0.0001, help='lamdainter')
    parser.add_argument('--w_intra', type=int, default=1, help='w_intra')
    parser.add_argument('--w_inter', type=int, default=1, help='w_inter')

    with open(yaml_path) as args_file:
        args = parser.parse_args()
        args_key = "-".join([args.model_name, args.dataset, args.custom_key])
        try:
            parser.set_defaults(**dict(YAML().load(args_file)[args_key].items()))
        except KeyError:
            raise AssertionError("KeyError: there's no {} in yamls".format(args_key), "red")

    # Update params from .yamls
    args = parser.parse_args()
    return args


def printConfig(args):
    arg2value = {}
    for arg in vars(args):
        arg2value[arg] = getattr(args, arg)
    print(arg2value)

def main():
    args = get_args(
        model_name="MGDCR",
        dataset="acm",  # acm imdb
        custom_key="Node",  # Node: node classification  Clu: clustering   Sim: similarity    SemiNode: semi-supervised learning
    )
    if args.dataset == "imdb" or args.dataset == "acm" :
        args.length = 2
    else:
        args.length = 3
    if args.custom_key == 'SemiNode':
        args.isSemi = True
    printConfig(args)
    embedder = MGDCR(args)
    # macro_f1s, micro_f1s, k1, st = embedder.training()

    filePath = "log"
    exp_ID = 0
    for filename in os.listdir(filePath):
        file_info = filename.split("_")
        file_dataname = file_info[0]
        if file_dataname == args.dataset:
            exp_ID = max(int(file_info[1]), exp_ID)
    exp_name = args.dataset + "_" + str(exp_ID + 1)
    exp_name = os.path.join(filePath, exp_name)
    os.makedirs(exp_name)
    arg_file = open(os.path.join(exp_name, "arg.txt"), "a")
    for k, v in sorted(args.__dict__.items()):
        arg_file.write("\n- {}: {}".format(k, v))

    macro_f1s, micro_f1s, k1, st = embedder.training()
    os.rename(exp_name,
              exp_name + "_" + '%.4f' % np.mean(macro_f1s) + "+_" + '%.4f' % np.std(macro_f1s) + "_" + '%.4f' % np.mean(
                  micro_f1s) + "+_" + '%.4f' % np.std(micro_f1s))
    arg_file.write(
        "\n- macro_f1s:{},std:{}, micro_f1s:{},std:{},k1:{},std:{}, similarity:{}".format(
            np.mean(macro_f1s), np.std(macro_f1s),
            np.mean(micro_f1s), np.std(micro_f1s),
            np.mean(k1[0]), np.std(k1), st))
    arg_file.close()
    return macro_f1s, micro_f1s, k1, st


if __name__ == '__main__':
    macro_f1s, micro_f1s, k1, st = main()
