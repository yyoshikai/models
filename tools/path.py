import sys, os
import re
import shutil
from glob import glob
import subprocess
import pickle
from datetime import datetime
import pandas as pd

def get_root_path():
    dirnames = re.split("/", os.path.abspath(__file__))
    root_path = ""
    for dirname in dirnames[1:len(dirnames)-3]:
        root_path =root_path+'/'+dirname
    return root_path

def cleardir(dirname, exist_ok=None):
    _cleardir(dirname)
    os.makedirs(dirname)

def _cleardir(dirname):
    for path in glob(os.path.join(dirname, '*')):
        if os.path.isdir(path):
            _cleardir(path)
        else:
            os.remove(path)
    if os.path.exists(dirname):
        os.rmdir(dirname)

def find_file_s(file_s, prefix):
    if type(file_s) == str:
        return find_file_s([file_s], prefix)[0]
    founds = []
    for file in file_s:
        if os.path.exists(file):
            founds.append(file)
        elif os.path.exists(prefix+file):
            founds.append(prefix+file)
        elif os.path.exists(prefix+"/"+file):
            founds.append(prefix+"/"+file)
        else:
            raise FileNotFoundError(f'Neither "{file}", "{prefix}{file}" nor "{prefix}/{file}" was found.')
    return founds

def make_pardir(path):
    path_dir = os.path.dirname(path)
    if len(path_dir) > 0:
        os.makedirs(path_dir, exist_ok=True)

def find_file(file, prefix):
    if os.path.exists(file):
        return file
    elif os.path.exists(prefix+file):
        return prefix+file
    elif os.path.exists(prefix+"/"+file):
        return prefix+"/"+file
    else:
        return None

def read_csv_tsv(file, **kwargs):
    ext = file.split('.')[-1]
    if ext == 'csv':
        df = pd.read_csv(file, keep_default_na=False, **kwargs)
    elif ext in ['tsv', 'txt']:
        df = pd.read_csv(file, sep='\t', keep_default_na=False, **kwargs)
    else:
        raise ValueError(f"Unsupported type of file extension: {file}")
    return df


def read_table2(path: str, sep: str=None, args: dict={}, **kwargs):
    if len(kwargs) > 0:
        print(f"[WARNING] Unknown kwargs: {kwargs.keys()}", file=sys.stderr)
    if sep is None:
        ext = path.split('.')[-1]
        if ext in ['txt', 'tsv']:
            sep = '\t'
        elif ext == 'csv':
            sep = ','
        else:
            raise ValueError("Unknown format of table. Please specify config.sep")
    return pd.read_csv(path, sep=sep, **args)
def read_table(config):
    print("Usage of read_table is depricated. use read_table2 (from args) instead.")
    return read_table2(**config)

def read_array(config, df_cache=None):
    if isinstance(config, dict):
        config = config.copy()
        col = config.pop('col')
        ext = config.path.split('.')[-1]
        if ext in ['csv', 'tsv', 'txt']:
            if df_cache is not None:
                assert isinstance(df_cache, dict)
                if config.path not in df_cache:
                    df_cache[config.path] = read_table2(**config)
                df = df_cache[config.path]
            else:
                df = read_table2(**config)
            array = df[col].values
        elif ext == 'pkl':
            with open(config.path, 'rb') as f:
                array = pickle.load(f)
        else:
            raise ValueError(f"Unsupported type of path in read_array: {config.path}")
        return array
    else:
        return config

def make_result_dir(result_dir=None, dirname=None, duplicate=None):
    """
    互換性のためresult_dirとdirname両方残している。
    """
    if (result_dir is None) == (dirname is None):
        raise ValueError(f"Please specify either result_dir({result_dir}) XOR dirname({dirname})")
    if result_dir is not None:
        print(f"from make_result_dir: usage of 'result_dir' is deprecated. Use 'dirname' instead.")
        dirname = result_dir
    if os.path.exists(dirname):
        if duplicate == 'error':
            raise FileExistsError(f"'{dirname}' already exists.")
        elif duplicate == 'ask':
            answer = None
            while answer not in ['y', 'n']:
                answer = input(f"'{dirname}' already exists. Will you overwrite? (y/n)")
            if answer == 'n':
                return
        elif duplicate in {'overwrite', 'merge'}:
            pass
        else:
            raise ValueError(f"Unsupported config.result_dir.duplicate: {duplicate}")
    if duplicate == 'merge':
        os.makedirs(dirname, exist_ok=True)
    else:
        cleardir(dirname)
    return dirname

def timestamp():
    dt_now = datetime.now()
    return f"{dt_now.year%100:02}{dt_now.month:02}{dt_now.day:02}"