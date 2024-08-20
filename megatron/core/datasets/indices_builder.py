import numpy as np


def build_blend_indices(
    weights: list[float] | np.ndarray[float],
    num_datasets: int,
    size: int,
    ) -> tuple[np.ndarray[np.int16], np.ndarray[np.int64]]:
  """
  Given multiple datasets and a weighting array, build samples
  such that it follows those wieghts.
  """
  fsamples = np.array(weights) * size
  isamples = fsamples.astype(int)
  errors = fsamples - isamples
  error_ranks = np.argsort(errors)
  add = size - sum(isamples)
  isamples[error_ranks[-add:]] += 1
  #
  dataset_index = np.repeat(np.arange(num_datasets, dtype=np.int16)[error_ranks], isamples[error_ranks])
  c = 0
  dataset_sample_index = np.zeros(size, dtype=np.int64)
  maxrange = np.arange(max(isamples))
  for i in error_ranks:
      isample = isamples[i]
      dataset_sample_index[c:c+isample] = maxrange[0:isample]
      c += isample
  return dataset_index, dataset_sample_index
