import os
import tqdm
import asyncio
import numpy as np
from collections import defaultdict, deque
from reorder import batch_cyclic_reorder
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

        self.ro_points = np.memmap(self.points_memmap_file, mode="readonly", dtype=self.dtype, shape=self.shape)

    def __len__(self):
        return self.shape[0]
    
    def __getitem__(self, ind):
        return self.ro_points[ind]
    
    def __setitem__(self, ind, val):
        write = np.memmap(self.points_memmap_file, mode="w+", dtype=self.dtype, shape=self.shape)
        write[ind] = val
        write.flush()
        del write
        
    def get_points(self, start, end):
        """
        Return memmap from readonly points map
        """
        return self.ro_points[start:end]
    
    def reorder_inplace_blocked(self, start, end, order, block_rows=65536):
        D = self.shape[1]
        N = end - start

        points = np.memmap(self.points_memmap_file, mode="r+", dtype=self.dtype, shape=self.shape)

        for i in range(0, N, block_rows):
            j = min(i + block_rows, N)

            ii = i + start
            jj = j + start


            tmp = points[ii:jj].copy()
            src = order[i:j]

            inside  = (src >= ii) & (src < jj)
            outside = ~inside


            points[ii:jj][outside] = points[src[outside]]
            rel = src[inside] - i
            points[ii:jj][inside] = tmp[rel]

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

    @staticmethod
    def split_cycle(cycle, fragment_size):
        """
        Split cycle into connected subcycles
        """

        subs = []

        for i in range(0, len(cycle), fragment_size - 1):
            subs.append(cycle[i:min(i + fragment_size, len(cycle))])

        if len(subs[-1]) in [0, 1]: subs.pop()
        if len(subs[-1]) < fragment_size:
            subs[-1].extend([-1, ] * (fragment_size - len(subs[-1])))

        return subs
    
    @staticmethod
    def get_cycles(start, end, order):
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

        return cycles
    
    @staticmethod
    def get_split_cycles(start, end, order, fragment_size=256):
        cycles = PointsManager.get_cycles(start, end, order)

        count = 0
        i = 0

        while count < len(cycles):
            if len(cycles[i]) > fragment_size:
                cycles.extend(PointsManager.split_cycle(cycles.pop(i), fragment_size))
            else:
                cycles[i].extend([-1, ] * (fragment_size - len(cycles[i])))
                i += 1

            count += 1

        return cycles

    def reorder(self, start, end, order, fragment=256, num_workers=16, n_threads=4):
        """
        Form index closed loops (https://en.wikipedia.org/wiki/100_prisoners_problem), process
        each closed cycle in its own process, split cycles into subcycles and distribute over threads
        """
        subcycles = np.array(self.get_split_cycles(start, end, order, fragment)).astype(np.int32)

        arr = np.memmap(self.points_memmap_file, mode="r+", dtype=self.dtype, shape=self.shape).view(np.uint16)

        for i in range(len(subcycles)):
            if i == 0: tails = []
            for j in range(1, len(subcycles[0])):
                if subcycles[i, -j] != -1:
                    tails.append(arr[subcycles[i, -j]].copy())
                    break

        batch_cyclic_reorder(arr, subcycles, np.array(tails))
        del arr

        return

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
                        fragment,
                        n_threads
                    )
                )
            for f in futures:
                f.result()

        