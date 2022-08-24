'''
Run the artefact detector, either as a one-shot or as a thread

Usage: python artefact_detector.py <input_dir> <output_dir> <network_path> <bgDetection> <as_thread>
'''

from run.artefact_detector import artefact_detector
import sys
import threading


def main():
    if len(sys.argv) < 5:
        print("Not enough arguments")
        return

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    network_path = sys.argv[3]
    bgDetection = sys.argv[4] in ['true', 'True', '1']
    asThread = sys.argv[5] in ['true', 'True', '1']

    if asThread:
        t = threading.Thread(target=artefact_detector,
                             args=(input_dir, output_dir, network_path, asThread, None, bgDetection, True))
        t.start()
    else:
        artefact_detector(input_dir, output_dir, network_path, asThread, None, bgDetection, True)


if __name__ == '__main__':
    main()
