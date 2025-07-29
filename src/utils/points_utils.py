import os
import tqdm
import asyncio
import numpy as np
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

class AsyncMultiQueue:
    """
    Asynchronous key-value queue

    Didn't end up using this for point management, but may be otherwise useful
    """
    def __init__(self):
        self._values  = defaultdict(deque)
        self._waiters = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def get(self, key):
        async with self._lock:
            if self._values[key]:
                return self._values[key].popleft()
            fut = asyncio.get_running_loop().create_future()
            self._waiters[key].append(fut)
        return await fut
    
    def put(self, key, value):
        async def _deliver():
            async with self._lock:
                if self._waiters[key]:
                    print("giving to waiter")
                    self._waiters[key].popleft().set_result(value)
                else:
                    print("appending")
                    self._values[key].append(value)
        asyncio.create_task(_deliver())

    def put_threadsafe(self, key, value, loop):
        loop.call_soon_threadsafe(self.put, key, value)


class PointsManager:
    """
    Object for memory-efficiently reordering numpy memmap
    """
    def __init__(self, points_memmap_file, shape, dtype):
        
        self.points_memmap_file = points_memmap_file
        self.shape = shape
        self.dtype = dtype

    def _sub_reorder(self, arr:np.memmap, indices, tail_value:np.ndarray):
        """
        Single thread call of split cyclic reorder - generalizes to closed loop or loop fragment
        """
        for i in range(len(indices) - 2):
            arr[indices[i]] = arr[indices[i + 1]]

        arr[indices[-2]] = tail_value

    def _process_cycle(self, cycle, approx_fragment_size, n_threads):
        """
        Split `cycle` into sub-cycles of size `approx_fragment_size`, reorder over `n_threads` threads
        """
        arr = np.memmap(self.points_memmap_file, mode="r+", dtype=self.dtype, shape=self.shape)

        sub_cycles = self.split_cycle(cycle, approx_fragment_size=approx_fragment_size)

        tails = [arr[sub[-1]].copy() for sub in sub_cycles]

        with ThreadPoolExecutor(max_workers=n_threads) as tp:
            tp.map(lambda args: self._sub_reorder(arr, *args), zip(sub_cycles, tails))

        if isinstance(arr, np.memmap):
            arr.flush()

    def split_cycle(self, cycle, approx_fragment_size):
        """
        Split cycle into connected subcycles
        """
        if len(cycle) <= (approx_fragment_size + 1):
            return [cycle, ]

        # Split into proper number of subcycles
        n = len(cycle) // (approx_fragment_size - 1) + 1
        splits = np.linspace(0, len(cycle), n).astype(int)
        subs = [cycle[splits[i]:splits[i+1]] for i in range(n - 1)]

        # Add in connectors
        for i in range(len(subs) - 1):
            subs[i].append(subs[i + 1][0])

        return subs

    def reorder(self, start, end, order, approx_fragment=256, num_workers=16, n_threads=4):
        """
        Form index closed loops (https://en.wikipedia.org/wiki/100_prisoners_problem), process
        each closed cycle in its own process, split cycles into subcycles and distribute over threads
        """
        # Find cycles
        visited = np.zeros(len(order), dtype=bool)
        cycles = []
        for start_ind in range(start, end):
            if visited[start_ind - start] or order[start_ind - start] == start_ind:
                continue
            cycle = [start_ind, ]

            while True:
                visited[cycle[-1] - start] = True

                cycle.append(
                    int(order[cycle[-1] - start])
                )

                # Cycle closed, use buffer
                if cycle[0] == cycle[-1]:
                    cycles.append(cycle)
                    break

        procs = min(num_workers, os.cpu_count() or 1, len(cycles))        

        with ProcessPoolExecutor(max_workers=procs) as px:
            futures = []
            for cycle in cycles:
                futures.append(
                    px.submit(
                        self._process_cycle,
                        cycle,
                        approx_fragment,
                        n_threads
                    )
                )
            for f in futures:
                f.result()

        