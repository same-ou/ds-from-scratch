from typing import List, NamedTuple
from collections import Counter

from helpers.vectors import distance, Vector

def raw_majority_vote(labels: List[str]) -> str:
    votes = Counter(labels)
    winner, _ = votes.most_common(1)[0]
    return winner

def majority_vote(labels: List[str]) -> str:
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count 
                       for count in vote_counts.values()
                       if count == winner_count])
    if num_winners == 1:
        return winner
    else:
        return majority_vote(labels[:-1])
    
class LabelPoint(NamedTuple):
    point: Vector
    label: str
    
def knn_classify(k: int, labled_points: List[LabelPoint], new_point: Vector) -> str:
    by_distance = sorted(labled_points, 
                         key = lambda lp: distance(lp.point, new_point))
    
    k_nearest_labels = [lp.label for lp in by_distance[:k]]
    
    return majority_vote(k_nearest_labels)


    
