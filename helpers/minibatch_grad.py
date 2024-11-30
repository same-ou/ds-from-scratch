from typing import TypeVar, List, Iterator
import random

T = TypeVar('T')

def minibatches(dataset: List[T], batch_size: int, shuffle: bool = True) -> Iterator[List[T]]:
    """Generates a batch generator for a dataset."""
  # start indexes 0, batch_size, 2 * batch_size, ...
    batch_starts = [start for start in range(0, len(dataset), batch_size)]
    if shuffle: random.shuffle(batch_starts)
    
    # shuffle the batches
    for start in batch_starts:
        end = start + batch_size
        yield dataset[start:end]