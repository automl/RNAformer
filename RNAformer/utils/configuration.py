import re
import yaml
import pathlib
import copy
from typing import Dict
from functools import reduce
import operator


def getFromDict(dataDict, mapList):
    return reduce(operator.getitem, mapList, dataDict)


def setInDict(dataDict, mapList, value):
    getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value


def convert_string_value(value):
    if value in ('false', 'False'):
        value = False
    elif value in ('true', 'True'):
        value = True
    else:
        try:
            value = int(value)
        except:
            try:
                value = float(value)
            except:
                pass
    return value


def read_unknown_args(unknown_args, config_dict):
    for arg in unknown_args:
        if '=' in arg:
            keys = arg.split('=')[0].split('.')
            value = convert_string_value(arg.split('=')[1])
            print(keys, value)
            setInDict(config_dict, keys, value)
        else:
            raise UserWarning(f"argument unknown: {arg}")
    return config_dict


class SimpleNestedNamespace(Dict):
    def __init__(self, *args, **kwargs):

        super().__init__(**kwargs)

        for k, v in kwargs.items():
            if isinstance(v, Dict):
                kwargs[k] = SimpleNestedNamespace(**v)

        self.__dict__.update(kwargs)

    def __repr__(self):
        keys = sorted(self.__dict__)
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __str__(self):
        return self.__dict__.__str__()


class Config(SimpleNestedNamespace):

    def __init__(self, config_file=None, config_dict=None):

        if config_file is None and config_dict is None:
            raise UserWarning("ConfigHandler: config_file and config_dict is None")

        elif config_file is not None and config_dict is None:
            with open(config_file, 'r') as f:
                config_dict = yaml.load(f, Loader=yaml.Loader)

        def convert_exponential_string(s):
            pattern = r"([+-]?\d+\.\d+|[+-]?\d+)[eE]([+-]?\d+)"
            match = re.search(pattern, s)
            if match:
                mantissa = float(match.group(1))
                exponent = int(match.group(2))
                result = mantissa * (10 ** exponent)
                if result.is_integer():
                    return int(result)
                else:
                    return result
            else:
                return s

        def get_attr_by_link(obj, links):
            attr = obj[links[0]]
            if isinstance(attr, Dict) and len(links) > 1:
                return get_attr_by_link(attr, links[1:])
            return attr

        def replace_linker(dictionary):
            for k, v in dictionary.items():
                if isinstance(v, str):
                    dictionary[k] = convert_exponential_string(v)
                if isinstance(v, Dict):
                    replace_linker(v)
                if isinstance(v, str) and len(v) > 3 and v[0] == '$' and v[1] == '{' and v[-1] == '}':
                    links = v[2:-1].split('.')
                    dictionary[k] = get_attr_by_link(config_dict, links)

        replace_linker(config_dict)

        super().__init__(**config_dict)

    def get_dict(self):
        def resolve_namespace(dictionary):
            for k, v in dictionary.items():
                if isinstance(v, SimpleNestedNamespace):
                    dictionary[k] = resolve_namespace(v.__dict__)
            return dictionary

        dictionary = copy.deepcopy(self.__dict__)
        return resolve_namespace(dictionary)

    def save_config(self, directory, file_name="config.yml"):
        dir = pathlib.Path(directory)
        dir.mkdir(parents=True, exist_ok=True)
        with open(dir / file_name, 'w+') as f:
            config_dict = self.get_dict()
            yaml.dump(config_dict, f, default_flow_style=False, encoding='utf-8')
        return dir / file_name
