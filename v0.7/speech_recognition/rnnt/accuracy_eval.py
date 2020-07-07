#!/usr/bin/env python

import argparse
import array
import json
import sys
import os

from QSL import AudioQSL

sys.path.insert(0, os.path.join(os.getcwd(), "pytorch"))
from helpers import process_evaluation_epoch, __gather_predictions
from parts.manifest import Manifest


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--instr", action="store_true", help="enable instrumentation", default=False)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    labels = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]
    qsl = AudioQSL(args.dataset_dir, args.manifest, labels)
    manifest = qsl.manifest
    with open(os.path.join(args.log_dir, "mlperf_log_accuracy.json")) as fh:
        results = json.load(fh)
    hypotheses = []
    references = []
    instrumentation = []
    for result in results:
        hypotheses.append(array.array('q', bytes.fromhex(result["data"])).tolist())
        references.append(manifest[result["qsl_idx"]]["transcript"])
        if args.instr:
            blob = {}
            blob['hypothesis'] = hypotheses[-1]
            blob['reference'] = references[-1]
            blob['result'] = result
            instrumentation.append(blob)
    hypotheses = __gather_predictions([hypotheses], labels=labels)
    references = __gather_predictions([references], labels=labels)
    d = dict(predictions=hypotheses,
             transcripts=references)

    wer = process_evaluation_epoch(d)
    print("Word Error Rate:", wer)

    if args.instr:
        with open('accuracy_instr.json', 'w') as save_file:
            json.dump({'wer': wer, 'samples': instrumentation}, save_file, indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
