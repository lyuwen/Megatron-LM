import argparse

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("file", type=str, help="File to process.")
  args = parser.parse_args()
  total = 0
  with open(args.file, "r") as f:
    for l in f:
      if l.startswith("#") or not l.strip():
        continue
      total += int(l.split()[0])
  print(total)
