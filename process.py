"""
240402 CallProcess.__init__のoutput=0のときoutputを無視する

"""
import importlib

import torch
import numpy as np
import torch.nn as nn

from .models2 import function_config2func, PRINT_PROCESS


class Process:
    def __init__(self):
        pass
    def __call__(self, model, batch):
        raise NotImplementedError

class CallProcess(Process):
    def __init__(self, input, output=None, **kwargs):
        self.input = input
        self.output = output
        if self.output is None:
            self.output = self.input
        self.kwargs = kwargs
    def __call__(self, model, batch):
        callable_ = self.get_callable(model)
        if self.input is None:
            output = callable_(**self.kwargs)
        elif isinstance(self.input, str):
            output = callable_(batch[self.input], **self.kwargs)
        elif isinstance(self.input, list):
            output = callable_(*[batch[i] for i in self.input], **self.kwargs)
        elif isinstance(self.input, dict):
            output = callable_(**{name: batch[i] for name, i in self.input.items()}, **self.kwargs)
        else:
            raise ValueError(f'Unsupported type of input: {self.input}')
        if isinstance(self.output, str):
            batch[self.output] = output
        elif isinstance(self.output, list):
            for oname, o in zip(self.output, output):
                batch[oname] = o
        elif self.output == 0:
            pass
        else:
            raise ValueError(f'Unsupported type of output: {self.output}')
    def get_callable(self, model):
        raise NotImplementedError
class ForwardProcess(CallProcess):
    def __init__(self, module, input, output=None, **kwargs):
        """
        Parameters
        ----------
        module: str
            Name of module.
        input: str, list[str] or dict[str, str]
            Name of input(s) in the batch to the module.
        output: str, list[str], or None
            Name of output(s) in the batch from the module.
            If None, input is used as output (inplace process)
        kwargs: dict
            Other kwargs are directly send to module
        """
        super().__init__(input, output, **kwargs)
        self.module = module
    def get_callable(self, model):
        return  model[self.module]
    def __str__(self):
        return f"ForwardProcess(input={self.input}, output={self.output}, kwargs={self.kwargs}, module={self.module})"
class FunctionProcess(CallProcess):
    def __init__(self, function, input, output=None, **kwargs):
        """
        Parameters
        ----------
        function: dict
            Input for function_config2func
        input: str, list[str] or dict[str, str]
            Name of input(s) in the batch to the module.
        output: str, list[str], or None
            Name of output(s) in the batch from the module.
            If None, input is used as output (inplace process) 
        kwargs: dict
            他のパラメータはモジュールに直接渡される。    
        """
        super().__init__(input, output, **kwargs)
        self.function = function_config2func(function)
    def get_callable(self, model):
        return self.function
    def __str__(self):
        return f"ForwardProcess(input={self.input}, output={self.output}, kwargs={self.kwargs}, function={self.function})"

class IterateProcess(Process):
    def __init__(self, length, processes, i_name='iterate_i'):
        """
        Parameters
        ----------
        length: int | str
            Length of iteration | name of it in batch.
        processes: list[dict]
            Parameters for processes to iterate
        i_name: str
            Name of index of iteration in batch        
        """
        self.length = length
        self.processes = [get_process(**process) for process in processes]
        self.i_name = i_name
    def __call__(self, model, batch):
        if isinstance(self.length, int): length = self.length
        else: length = batch[self.length]
        for i in range(length):
            batch[self.i_name] = i
            for i, process in enumerate(self.processes):

                if PRINT_PROCESS:
                    # Show parameters
                    print(f"---process {i}---")
                    for key, value in batch.items():
                        if isinstance(value, (torch.Tensor, np.ndarray)):
                            print(f"  {key}: {list(value.shape)}")
                        else:
                            print(f"  {key}: {type(value).__name__}")
                

                process(model, batch)

process_type2class = {
    'forward': ForwardProcess,
    'function': FunctionProcess,
    'iterate': IterateProcess,
}
def get_process(type='forward', **kwargs):
    return process_type2class[type](**kwargs)
def get_process_from_config(config):
    return get_process(**config)

N_LOOP_MODULE = 0
def build_processes(configs, train_configs=None):
    if isinstance(configs, list):
        if train_configs is not None:
            configs = train_configs + configs
        processes = [get_process(**process) for process in configs]
    elif isinstance(processes, dict):
        if 'path' in configs: # 関数を指定する場合
            loop_module = importlib.import_module(name=f'loop_module{N_LOOP_MODULE}', package=processes.path)
            N_LOOP_MODULE = N_LOOP_MODULE+1
            processes = loop_module.__getattribute__(processes.function)
        else:
            processes = list(processes.values())
            if trconfig.val_loop_add_train: 
                processes += list(trconfig.train_loop)
            val_times = [process.pop('order') for process in processes]
            processes = [processes[i] for i in np.argsort(val_times)]
            processes = [get_process(**process) for process in processes]
    else:
        raise ValueError
    return processes