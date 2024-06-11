import os, sys
import argparse
from typing import Callable
import inspect
import yaml
import pandas as pd
from addict import Dict
from collections import defaultdict

def default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--notice", action='store_true')
    return parser

def save_args(args, file):
    arg_names = []
    arg_types = []
    arg_values = []
    for arg_name in dir(args):
        if not arg_name[0] == '_':
            arg = eval(f"args.{arg_name}")
            arg_names.append(arg_name)
            arg_types.append(type(arg).__name__)
            arg_values.append(arg)
    arg_df = pd.DataFrame({'name':arg_names, 'type':arg_types, 'value': arg_values})
    arg_df.to_csv(file, index=False)

def args_from_df(file):
    arg_df = pd.read_csv(file, keep_default_na=False)
    args = argparse.Namespace()
    for i, arg_row in arg_df.iterrows():
        if arg_row['type'].lower() == 'nonetype':
            arg_value = None
        elif arg_row['type'] == 'bool':
            arg_value = arg_row['value'] == 'True'
        elif arg_row['type'] == 'list':
            arg_value = eval(f"{arg_row['type']}(ast.literal_eval(arg_row['value']))")
        else:
            arg_value = eval(f"{arg_row['type']}(arg_row['value'])")
        exec(f"args.{arg_row['name']} = arg_value")
    return args

# configを2回書き換えた場合警告を出す。
def update_with_check(origin, after, path, modifieds, filename):
    if isinstance(origin, dict) and isinstance(after, dict):
        for key, value in after.items():
            if key not in origin:
                origin[key] = value
            else:
                origin[key], modifieds = update_with_check(
                    origin[key], value, path+(key,), modifieds, filename)
        return origin, modifieds
    else:
        if origin != after:
            if origin != {}:
                for mpath in modifieds.keys():
                    if path[:len(mpath)] == mpath or mpath[:len(path)] == path:
                        print(f"WARNING: {'.'.join(path)} was overwritten for multiple times:", 
                            file=sys.stderr)
                        for filename0 in modifieds[mpath]:
                            print("  ", filename0, file=sys.stderr)
                        print("  ", filename, file=sys.stderr)
                        
                modifieds[path].append(filename)
        return after, modifieds


def search_args(config):
    args = []
    types = []
    if isinstance(config, dict):
        for child in config.values():
            arg_c, types_c = search_args(child)
            args += arg_c
            types += types_c
    elif isinstance(config, list):
        for child in config:
            args_l, types_l = search_args(child)
            args += args_l
            types += types_l
    elif type(config) == str:
        if len(config) > 2 and config[:2] == '--':
            configs = config.split(':')
            args = [configs[0]]
            types = [str] if len(configs) == 1 else [eval(configs[1])]
    return args, types

def gather_args(config):
    args = []
    if isinstance(config, dict):
        if 'argname' in config:
            arg_args = {}
            for key, value in config.items():
                if key == 'argname':
                    continue
                elif key == 'type':
                    arg_args['type'] = eval(config.type)
                else:
                    arg_args[key] = value
            args = [(config.argname, arg_args)]
        else:
            for child in config.values():
                args += gather_args(child)
    elif isinstance(config, list):
        for child in config:
            args += gather_args(child)
    return args

def fill_args(config, args):
    if isinstance(config, dict):
        if 'argname' in config:
            return args[config.argname]
        else:
            for label, child in config.items():
                config[label] = fill_args(child, args)
            return config
    elif isinstance(config, list):
        config = [fill_args(child, args) for child in config]
        return config
    else:
        return config

def subs_vars(config, vars):
    if isinstance(config, str):
        if config in vars:
            return vars[config]
        for key, value in vars.items():
            config = config.replace(key, str(value))
        return config
    elif isinstance(config, dict):
        return Dict({label: subs_vars(child, vars) for label, child in config.items()})
    elif isinstance(config, list):
        return [subs_vars(child, vars) for child in config]
    else:
        return config

def delete_args(config):
    if isinstance(config, dict):
        new_config = Dict()
        for key, value in config.items():
            if value == '$delete':
                continue
            elif isinstance(value, (dict, list)):
                value = delete_args(value)
            new_config[key] = value
        return new_config
    elif isinstance(config, list):
        return [delete_args(child) for child in config]
    else:
        return config

def load_config2(config_dir, default_configs, remove_variables=False):
    return load_config3(sys.argv[1:], config_dir, default_configs, remove_variables)

# get args from parameter(argv)
def load_config3(argv, config_dir, default_configs, remove_variables=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs='*', default=[])
    args = parser.parse_known_args(argv)[0]
    config = None
    modifieds = defaultdict(list)
    for file in default_configs+args.config:
        if os.path.isfile(file):
            with open(file, 'r') as f:
                aconfig = Dict(yaml.load(f, yaml.Loader))
        else:
            with open(os.path.join(config_dir, file)+".yaml", 'r') as f:
                aconfig = Dict(yaml.load(f, yaml.Loader))
        # config.update(aconfig)
        if config is None: 
            config = aconfig
        else:
            config, modifieds = update_with_check(config, aconfig, tuple(), modifieds, file)
    config = Dict(config)
    args = gather_args(config)
    for arg_name, arg_args in args:
        parser.add_argument(f"--{arg_name}", **arg_args)
    with open("config.yaml", 'w') as f:
        yaml.dump(config.to_dict(), f, sort_keys=False)
    args = vars(parser.parse_args(argv))
    config = fill_args(config, args)
    if 'variables' in config.keys():
        variables = config['variables']
        var_names = list(variables.keys())
        for var_name in var_names:
            var_value = variables.pop(var_name)
            config = subs_vars(config, {var_name: var_value})
            variables = subs_vars(variables, {var_name: var_value})
        if remove_variables:
            del config['variables']
    config = delete_args(config)
    return Dict(config)

def clip_config(config: dict, func: Callable):
    sig = inspect.signature(func)
    fkeys = sig.parameters.keys()
    for ckey in config:
        if ckey not in fkeys:
            del config[ckey]
    for fkey, fparam in sig.parameters.items():
        if fkey not in config and fparam.default != inspect.Parameter.empty:
            config[fkey] = fparam.default
    return config