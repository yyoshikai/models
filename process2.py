
import torch.nn as nn
from .models import function_config2func

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

process_type2class = {
    'forward': ForwardProcess,
    'function': FunctionProcess,
}
def get_process(type=None, **kwargs):
    if type is None:
        if 'module' in kwargs: type = 'forward'
        elif 'function' in kwargs: type = 'function'
        else: raise ValueError
    return process_type2class[type](**kwargs)

class SequentialProcess(Process):
    def __init__(self, processes, process_dict):
        self.processes = []
        for p in processes:
            if isinstance(p, str):
                self.processes.append(process_dict[p])
            else:
                self.processes.append(get_process(**p))
    
    def __call__(self, model, batch):
        for p in self.processes:
            p(model, batch)

def get_processes(config: dict):

    process_dict = {}
    for key, processes in config:
        process = SequentialProcess(processes, process_dict)
        process_dict[key] = process
    return process_dict


