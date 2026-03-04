"""Baseline placement strategies for comparison with FGD.

Implements BestFit, FirstFit, Random, DotProd, GpuPacking, and GpuClustering
placement as described in the FGD paper (ATC'23). Each strategy decides which
node to place a task on, given the current cluster state.
"""

import math
import random
from typing import List, Optional, Tuple

from fgd import Node, Task


class BaselinePlacer:
    """Base class for placement strategies."""

    def place(self, task: Task, nodes: List[Node]) -> Tuple[Optional[Node], Optional[List[int]]]:
        raise NotImplementedError


class BestFitPlacer(BaselinePlacer):
    """BestFit: place task on the node with the least remaining GPU capacity
    that can still fit the task. Minimizes wasted space per-node."""

    def place(self, task: Task, nodes: List[Node]) -> Tuple[Optional[Node], Optional[List[int]]]:
        best_node = None
        best_gpus = None
        best_remaining = float('inf')

        for node in nodes:
            if not node.can_fit_task(task):
                continue
            gpu_indices = node.find_suitable_gpus(task)
            if gpu_indices is None:
                continue

            # For partial GPU, pick the GPU with smallest sufficient capacity
            if 0 < task.gpu_request < 1:
                candidates = [(idx, node.gpus[idx]) for idx in gpu_indices]
                candidates.sort(key=lambda x: x[1])
                gpu_idx, remaining = candidates[0]
                remaining_after = remaining - task.gpu_request
                if remaining_after < best_remaining:
                    best_remaining = remaining_after
                    best_node = node
                    best_gpus = [gpu_idx]
            else:
                # For full GPUs, score by total remaining GPU capacity
                remaining = sum(node.gpus) - task.gpu_request
                if remaining < best_remaining:
                    best_remaining = remaining
                    best_node = node
                    best_gpus = gpu_indices

        return best_node, best_gpus


class FirstFitPlacer(BaselinePlacer):
    """FirstFit: place task on the first node that can accommodate it.
    Simple and fast but may cause fragmentation."""

    def place(self, task: Task, nodes: List[Node]) -> Tuple[Optional[Node], Optional[List[int]]]:
        for node in nodes:
            if not node.can_fit_task(task):
                continue
            gpu_indices = node.find_suitable_gpus(task)
            if gpu_indices is None:
                continue

            if 0 < task.gpu_request < 1:
                return node, [gpu_indices[0]]
            return node, gpu_indices

        return None, None


class RandomPlacer(BaselinePlacer):
    """Random: place task on a randomly chosen node that can fit it."""

    def __init__(self, seed=42):
        self.rng = random.Random(seed)

    def place(self, task: Task, nodes: List[Node]) -> Tuple[Optional[Node], Optional[List[int]]]:
        candidates = []
        for node in nodes:
            if not node.can_fit_task(task):
                continue
            gpu_indices = node.find_suitable_gpus(task)
            if gpu_indices is None:
                continue

            if 0 < task.gpu_request < 1:
                candidates.append((node, [gpu_indices[0]]))
            else:
                candidates.append((node, gpu_indices))

        if not candidates:
            return None, None

        return self.rng.choice(candidates)


class DotProdPlacer(BaselinePlacer):
    """DotProd: score nodes by cosine similarity between requested and available
    resource vectors [cpu, gpu, memory]. Picks the node with the highest score,
    encouraging balanced resource consumption."""

    def place(self, task: Task, nodes: List[Node]) -> Tuple[Optional[Node], Optional[List[int]]]:
        best_node = None
        best_gpus = None
        best_score = -float('inf')

        req = [task.cpu_request, task.gpu_request, task.memory_request]
        req_mag = math.sqrt(sum(r * r for r in req))
        if req_mag == 0:
            return None, None

        for node in nodes:
            if not node.can_fit_task(task):
                continue
            gpu_indices = node.find_suitable_gpus(task)
            if gpu_indices is None:
                continue

            avail = [node.available_cpu, sum(node.gpus), node.available_memory]
            avail_mag = math.sqrt(sum(a * a for a in avail))
            if avail_mag == 0:
                continue

            dot = sum(r * a for r, a in zip(req, avail))
            score = dot / (req_mag * avail_mag)

            if score > best_score:
                best_score = score
                best_node = node
                if 0 < task.gpu_request < 1:
                    best_gpus = [gpu_indices[0]]
                else:
                    best_gpus = gpu_indices

        return best_node, best_gpus


class GpuPackingPlacer(BaselinePlacer):
    """GpuPacking: place task on the node with the fewest available GPUs
    (most packed). Consolidates GPU usage onto fewer nodes."""

    def place(self, task: Task, nodes: List[Node]) -> Tuple[Optional[Node], Optional[List[int]]]:
        best_node = None
        best_gpus = None
        best_score = float('inf')

        for node in nodes:
            if not node.can_fit_task(task):
                continue
            gpu_indices = node.find_suitable_gpus(task)
            if gpu_indices is None:
                continue

            score = sum(node.gpus)  # total available GPU capacity
            if score < best_score:
                best_score = score
                best_node = node
                if 0 < task.gpu_request < 1:
                    best_gpus = [gpu_indices[0]]
                else:
                    best_gpus = gpu_indices

        return best_node, best_gpus


class GpuClusteringPlacer(BaselinePlacer):
    """GpuClustering: place task on the node with the most free GPUs.
    Spreads GPU usage to preserve large contiguous blocks."""

    def place(self, task: Task, nodes: List[Node]) -> Tuple[Optional[Node], Optional[List[int]]]:
        best_node = None
        best_gpus = None
        best_score = -float('inf')

        for node in nodes:
            if not node.can_fit_task(task):
                continue
            gpu_indices = node.find_suitable_gpus(task)
            if gpu_indices is None:
                continue

            score = sum(node.gpus)  # total available GPU capacity
            if score > best_score:
                best_score = score
                best_node = node
                if 0 < task.gpu_request < 1:
                    best_gpus = [gpu_indices[0]]
                else:
                    best_gpus = gpu_indices

        return best_node, best_gpus
