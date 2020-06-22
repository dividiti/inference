import argparse
import torch
import random
import numpy as np
import pickle
import time
import json


def parse_args():
    parser = argparse.ArgumentParser(description='dumptool')
    parser.add_argument("dump", default=None,
                        type=str, help="The dump to view")
    parser.add_argument("--json", type=str,
                        help='dump file to json')
    parser.add_argument("--raw", type=str,
                        help='dump file to raw')
    parser.add_argument("--truncate", type=str,
                        help='truncate sequence to size')
    return parser.parse_args()


def main(args):

    xi, h0c0, xo, hNcN = torch.load(args.dump);

    print("input dims %s " % str(list(xi.size())))
    print("output dims %s " % str(list(xo.size())))

    if args.json:
        with open(args.json, "w") as outfile:
            dictionary = {}
            dictionary["input"] = xi.tolist()
            #dictionary["h_0_c_0"] = h0c0.tolist()
            dictionary["output"] = xo.tolist()
            #dictionary["h_n_c_n"] = h0c0.tolist()
            json.dump(dictionary, outfile)


    if args.raw:
        foot = open("in_" + args.raw, "wb")
        fool = list(xi.size())
        xdim = fool[0]
        if args.truncate:
            xdim = xdim if xdim <= int(args.truncate) else int(args.truncate)
        for k in range(xdim):
            for j in range(fool[1]):
                for i in range(fool[2]):
                    foot.write(xi.numpy()[k][j][i])
        foot.close()

        foot = open("out_" + args.raw, "wb")
        fool = list(xo.size())
        xdim = fool[0]
        if args.truncate:
            xdim = xdim if xdim <= int(args.truncate) else int(args.truncate)
        for k in range(xdim):
            for j in range(fool[1]):
                for i in range(fool[2]):
                    foot.write(xo.numpy()[k][j][i])
        foot.close()


if __name__ == "__main__":
    args = parse_args()

    main(args)
