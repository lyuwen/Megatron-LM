import os
import yaml
import argparse


checkpoint_index = "latest_checkpointed_iteration.txt"


def sum_column(file, column_index=0):
  with open(file, "r") as f:
    total = 0
    for l in f:
      if l.startswith("#"):
        continue
      total += int(l.split()[column_index])
  return total


class Config:

  def __init__(self, save_path):
    self.save_path = save_path
    self._config = dict()


  def __contains__(self, key):
    return key in self._config


  def __getitem__(self, key):
    return self._config[key]


  def __setitem__(self, key, value):
    self._config[key] = value
    return self


  def __enter__(self):
    if os.path.exists(self.save_path):
      with open(self.save_path, "r") as f:
        self._config = yaml.safe_load(f.read())
    return self


  def __exit__(self, *exc):
    with open(self.save_path, "w") as f:
      f.write(yaml.dump(self._config) + "\n")


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Restart status checker for batched pretraining")
  parser.add_argument("--last-batch-checkpoint", "--lbc", type=str, required=True, help="Checkpoint path of the last batch.")
  parser.add_argument("--current-batch-checkpoint", "--cbc", type=str, required=True, help="Checkpoint path of the last batch.")
  parser.add_argument("--batch-datalist", "--dl", type=str, required=True, help="Datalist file of current batch.")
  parser.add_argument("--status-save-path", "--save", "-s", type=str, required=True, help="Path to save the batched training status.")
  parser.add_argument("--global-batch-size", "--gbs", type=int, required=True, help="Global batch size.")
  parser.add_argument("--seq-length", "--seq", type=int, required=True, help="Sequence length.")
  parser.add_argument("--train-sample-ratio", "--tsr", type=float, default=1.0, help="Training sample ratio.")
  parser.add_argument("--reset-iterations", "--ri", action="store_true", help="Reset iteration count at the new batch.")

  args = parser.parse_args()

  statements = []
  status_key = f"{args.batch_datalist}-{args.current_batch_checkpoint}-{args.last_batch_checkpoint}"

  with Config(args.status_save_path) as config:
    if (status_key in config) and os.path.exists(os.path.join(args.current_batch_checkpoint, checkpoint_index)):
      # existing run
      statements.append(f"LOAD_CHECKPOINT_PATH=\"{args.current_batch_checkpoint}\"")
      statements.append(f"RESET_DATALOADER=\"\"")
    else:
      # new run
      config[status_key] = {}
      sample_size = int(sum_column(args.batch_datalist, column_index=0) * args.train_sample_ratio) // args.seq_length
      sample_iters = sample_size // args.global_batch_size
      config[status_key]['sample_size'] = sample_size
      config[status_key]['sample_iters'] = sample_iters
      with open(os.path.join(args.last_batch_checkpoint, checkpoint_index), "r") as f:
        config[status_key]['seen_steps'] = int(f.read())
      reset_arguments = "--reset-dataloader --override-opt_param-scheduler"
      if args.reset_iterations:
        reset_arguments += " --reset-iterations"
      #
      statements.append(f"LOAD_CHECKPOINT_PATH=\"{args.last_batch_checkpoint}\"")
      statements.append(f"RESET_DATALOADER=\"{reset_arguments}\"")
    statements.append(f"SEEN_STEPS=\"{config[status_key]['seen_steps']}\"")
    statements.append(f"SAMPLE_SIZE=\"{config[status_key]['sample_size']}\"")
    statements.append(f"SAMPLE_ITERS=\"{config[status_key]['sample_iters']}\"")
    statements.append(f"TRAIN_STEPS=\"{config[status_key]['sample_iters'] + config[status_key]['seen_steps']}\"")



  print("export " + " ".join(statements))


