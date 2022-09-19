import sys
import os
from easy_latent_seq.modules.lightning import JointProb
from easy_latent_seq.preprocess.preprocess import Preprocess
from easy_latent_seq.util.preprocess_util import Utils, Vocab


def main():
    from argparse import ArgumentParser
    from modules.util import Utils, Vocab
    from glob import glob
    from tqdm import tqdm
    from os.path import basename, exists

    # torch.autograd.set_detect_anomaly(True)
    parser = ArgumentParser()
    parser.add_argument("--checkpoint-path", default="./")
    parser.add_argument("--npz-path", default="./")
    parser.add_argument("--decoded-prefix", default="./")
    parser.add_argument("--force", action="store_true")
    parser = PreprocessNPZT9.add_argparse_args(parser)
    parser = JointProb.add_model_specific_args(parser)
    args = parser.parse_args()

    print(f"serialize prefix: {args.serialize_prefix}")

    Vocab.load(args.vocab_mapping)
    args.vocab_size = Vocab.size()
    args.bos, args.eos, args.pad = Utils.lookup_control(args.bos, args.eos, args.pad)

    model = JointProb.load_from_checkpoint(
        args.checkpoint_path, args=args, strict=False
    ).to("cpu")
    model.eval()

    filenames = glob(f"{args.npz_path}/*.npz")
    marks = ""
    for fname in tqdm(filenames, disable=None):
        bfname = basename(fname)
        decoded_fname = f"{args.decoded_prefix}/{bfname}.decoded"
        if args.force or not exists(decoded_fname):
            try:
                prob, mark = model.decode_from_npz(
                    fname, vocab_size=Vocab.size(), pad=args.pad
                )
                with open(decoded_fname, mode="w") as fh:
                    fh.write(f"{prob}\n")
                tokens = [Vocab.r_lookup(_) for _ in mark.tolist()]
                print(tokens)
            except Exception as e:
                print(f"cannot decode {fname}: {e}")
        else:
            print("already decoded, add toggle force to decode again")
    # decoded_string_fname = f'{args.decoded_prefix}/input_output.txt'
    # with open(decoded_string_fname, 'w') as f:
    #     f.write(marks)
    return


if __name__ == "__main__":
    main()
