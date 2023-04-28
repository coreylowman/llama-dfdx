import argparse
import os
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="root directory")
    parser.add_argument("dst", help="output directory")
    args = parser.parse_args()

    for f in os.listdir(args.src):
        if not f.endswith(".bin"):
            continue
        sd = torch.load(os.path.join(args.src, f))
        for key, tensor in sd.items():
            print(key, tensor.shape, tensor.dtype)
            path = os.path.sep.join(key.split("."))
            os.makedirs(os.path.join(args.dst, os.path.dirname(path)), exist_ok=True)
            with open(os.path.join(args.dst, path), "w") as fp:
                tensor.numpy().tofile(fp)
        del sd


if __name__ == "__main__":
    main()
