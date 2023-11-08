# istarmap.py for Python 3.8+
# source https://stackoverflow.com/questions/57354700/starmap-combined-with-tqdm
import multiprocessing.pool as mpp


def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap"""
    self._check_running()
    if chunksize < 1:
        raise ValueError("Chunksize must be 1+, not {0:n}".format(chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)  # type: ignore
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job, mpp.starmapstar, task_batches),  # type: ignore
            result._set_length,  # type: ignore
        )
    )
    return (item for chunk in result for item in chunk)


mpp.Pool.istarmap = istarmap  # type: ignore
