'''
Train network on datafeed

usage: python train_network.py <params_file>

<params_file> : JSON file
'''

from run.network_train import run_train, run_train_threaded
import json
import sys


def main():
    if len(sys.argv) < 2:
        print("Must provide JSON parameters file.")
        return

    for fname in sys.argv[1:]:
        with open(fname) as fp:
            params = json.load(fp)

        print(f"Starting training on network {params['clf_name']}")
        threaded = params['threaded'] if 'threaded' in params else False
        if threaded:
            run_train_threaded(params)
        else:
            run_train(params)


if __name__ == '__main__':
    main()
