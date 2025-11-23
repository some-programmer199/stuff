import lmdb
import sys
env=lmdb.open('./lmdb_data',map_size=2**30)
with env.begin(write=True) as txn:
    txn.put(b'key', b'value')
def worker():
    with env.begin() as txn:
        value = txn.get(b'key')
        print(value)
import multiprocessing
p = multiprocessing.Process(target=worker)
p.start()
p.join()
    