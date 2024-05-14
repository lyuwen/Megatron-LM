import os
import csv
import tqdm
import glob
import argparse
import smart_open
import numpy as np
import multiprocessing
from functools import partial
from tqdm.contrib.concurrent import process_map

from indexed_dataset import _IndexReader, _IndexWriter, DType


def process_single(path, dtype, out=None, reset_document_id=False):
  out = path.replace(".csv.gz", ".idx")
  with _IndexWriter(out, dtype) as writer, smart_open.open(path) as fin:
    reader = csv.reader(fin)
    sequence_lengths: List[int] = []
    document_indices: List[int] = []
    for i, row in enumerate(reader):
      start, end, sid, fname, fid = row
      sequence_lengths.append(int(end) - int(start))
      if reset_document_id:
        document_indices.append(i)
      else:
        document_indices.append(fid)
    document_indices.append(len(document_indices))
    writer.write(sequence_lengths, None, document_indices)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("files", nargs="+", help="Metadata files to process.")
  parser.add_argument("--output-dir", dest="output_dir", default=None, help="Output directory, by default create the idx file in the same location as the metadata file.")
  parser.add_argument("--dtype", dest="dtype", default="uint16", help="Tokens data formath.")
  parser.add_argument("--num-process", "-P", dest="num_process", type=int, default=os.cpu_count(), help="Number of lines to process in parallel.")
  parser.add_argument("--reset-document-id", "-R", dest="reset_document_id", action="store_true", help="Reset ID post-shuffle rather than use the original document order.")

  args = parser.parse_args()

  files = [f for file in args.files for f in glob.glob(file, recursive=True)]
  dtype = np.dtype(args.dtype).type

  #  for file in files:
  #    output = file.replace(".csv.gz", ".idx")
  #    process_single(file, dtype)

  fn = partial(process_single, dtype=np.uint32, reset_document_id=args.reset_document_id)
  num_process = min(len(files), args.num_process)

  with multiprocessing.Pool(processes=num_process) as pool:
      with tqdm.tqdm(total=len(files)) as pbar:
          for _ in pool.imap_unordered(fn, files):
              pbar.update()


