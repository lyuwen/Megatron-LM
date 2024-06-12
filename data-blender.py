import argparse
import numpy as np


def process_input(files_weights: list[str]) -> tuple[list[float], list[str]]:
    if len(files_weights) % 2 != 0:
        raise ValueError("Input files_weights is not the correct length.")
    files_weights = np.array(files_weights).reshape((-1, 2))
    weights = files_weights[:, 0].astype(float)
    files = files_weights[:, 1]
    return weights, files



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files_weights", nargs="+", help="Files and their weights")
    parser.add_argument("--expected-total", type=int, required=True, help="Expected total number of tokens")
    parser.add_argument("--equal", action="store_true", help="Treat as if every file has the same number of tokens when creating the blend")
    parser.add_argument("--output", "--out", "-o", type=argparse.FileType('w'), default='-', help="Where to write output")

    args = parser.parse_args()

    weights, files = process_input(args.files_weights)

    # print(args.files_weights, file=args.output)

    datalist = []
    total_weight = np.sum(weights)
    for i, file in enumerate(files):
        with open(file, "r") as f:
            dswt, dataset = process_input(f.read().strip().split())
        ds_total_weight = np.sum(dswt)
        if args.equal:
            new_dswt = dswt * weights / total_weight
        else:
            num_samples = weights[i] / total_weight * args.expected_total
            new_dswt = dswt / ds_total_weight * num_samples
        new_dswt = np.round(new_dswt).astype(int)
        for i, wt in enumerate(new_dswt):
            if wt > 0:
                datalist.append(f"{wt:>12d} {dataset[i]}")
    print("\n".join(datalist), file=args.output)

