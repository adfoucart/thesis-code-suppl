'''
Generate SNOW datasets

Usage:
python generate_snow.py <input_dir> <output_dir> <pRemove> <sigmaR> <f> <doBB> <doLA>
'''

from data import SNOWGenerator
import sys


def main():
    if len(sys.argv) < 8:
        print("Not enough arguments.")
        return

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    pRemove = float(sys.argv[3])
    sigmaR = int(sys.argv[4])
    f = int(sys.argv[5])
    doBB = sys.argv[6] == 'y'
    doLA = sys.argv[7] == 'y'

    SNOWGenerator(input_dir, output_dir, pRemove, sigmaR, f, doBB, doLA, True)


if __name__ == '__main__':
    main()
