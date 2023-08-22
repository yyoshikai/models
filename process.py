from .models2 import function_config2func

class CallProcess:
    def __init__(self, input, output=None):
        self.input = input
        self.output = output
        if self.output is None:
            self.output = self.input
    def __call__(self, model, batch):
        callable_ = self.get_callable(model)
        if self.input is None:
            output = callable_()
        if isinstance(self.input, str):
            output = callable_(batch[self.input])
        elif isinstance(self.input, list):
            output = callable_(*[batch[i] for i in self.input])
        elif isinstance(self.input, dict):
            output = callable_(**{name: batch[i] for name, i in self.input.items()})
        else:
            raise ValueError(f'Unsupported type of input: {self.input}')
        if isinstance(self.output, str):
            batch[self.output] = output
        elif isinstance(self.output, list):
            for oname, o in zip(self.output, output):
                batch[oname] = o
        else:
            raise ValueError(f'Unsupported type of output: {self.output}')
    def get_callable(self, model):
        raise NotImplementedError
class ForwardProcess(CallProcess):
    def __init__(self, module, input, output=None):
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
        """
        super().__init__(input, output)
        self.module = module
    def get_callable(self, model):
        return  model[self.module]
class FunctionProcess(CallProcess):
    def __init__(self, function, input, output=None):
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
        """
        super().__init__(input, output)
        self.function = function_config2func(function)
    def get_callable(self, model):
        return self.function
class IterateProcess:
    def __init__(self, length, processes, i_name='iterate_i'):
        """
        Parameters
        ----------
        length: str
            Name of length of iteration in batch.
        processes: list[dict]
            Parameters for processes to iterate
        i_name: str
            Name of index of iteration in batch        
        """
        self.length = length
        self.processes = [get_process(**process) for process in processes]
        self.i_name = i_name
    def __call__(self, model, batch):
        for i in range(batch[self.length]):
            batch[self.i_name] = i
            for process in process:
                process(model, batch)

process_type2class = {
    'forward': ForwardProcess,
    'function': FunctionProcess,
    'iterate': IterateProcess
}
def get_process(type, **kwargs):
    return process_type2class[type](**kwargs)
def get_process_from_config(config):
    return get_process(**config)