import json
import logging
import os
import pickle
import sys

def pickle_dump_large_file(obj, filepath):
    max_bytes = 2**31-1
    bytes_out = pickle.dumps(obj)
    total_bytes = sys.getsizeof(bytes_out)
    with open(filepath, 'wb') as f:
        for idx in range(0,total_bytes,max_bytes):
            f.write(bytes_out[idx:idx+max_bytes])

def save(filepath,obj,message=None):
    if message is not None:
        logging.info("Saving {}...".format(message))
    pickle_dump_large_file(obj, filepath)
