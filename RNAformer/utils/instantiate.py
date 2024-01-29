from typing import Dict, List
import importlib
import inspect
from RNAformer.utils.configuration import Config, SimpleNestedNamespace


def get_class(target: str):
    target_class = target.split('.')
    module_name = '.'.join(target_class[:-1])
    class_name = target_class[-1]
    module = importlib.import_module(module_name)
    instance = getattr(module, class_name)
    return instance


def instantiate(config, *args, instance=None, **kwargs):
    """
    Instantiates a class by selecting the required args from a ConfigHandler. Omits wrong kargs
    @param config:      ConfigHandler object contains class args
    @param instance:    class
    @param kwargs:      kwargs besides/replacing ConfigHandler args
    @return:            class object
    """

    if isinstance(config, Config) or isinstance(config, SimpleNestedNamespace):
        config_dict = config.__dict__
    elif isinstance(config, Dict):
        config_dict = config
    elif isinstance(config, List):
        config_dict = {}
        for sub_conf in config:
            if isinstance(sub_conf, Config) or isinstance(sub_conf, SimpleNestedNamespace):
                config_dict.update(sub_conf.__dict__)
            elif isinstance(sub_conf, Dict):
                config_dict.update(sub_conf)
    else:
        raise UserWarning(
            f"cinit: Unknown config type. config must be Dict, AttributeDict or ConfigHandler but is {type(config)}")

    if instance is None and '_target_' not in config_dict:
        raise UserWarning(f"instantiate: keys missing instance or _target_")
    if instance is None:
        instance = get_class(config_dict['_target_'].__str__())
        del config_dict['_target_']

    if isinstance(instance, type):
        instance_args = inspect.signature(instance.__init__)
        instance_keys = list(instance_args.parameters.keys())
        instance_keys.remove("self")
    else:
        instance_keys = inspect.getfullargspec(instance).args

    init_dict = {}
    for name, arg in kwargs.items():
        if name in instance_keys:
            init_dict[name] = arg

    for name, arg in config_dict.items():
        if name in instance_keys and name not in init_dict.keys():
            init_dict[name] = arg

    init_keys = list(init_dict.keys())
    missing_keys = list(set(instance_keys) - set(init_keys))
    if len(missing_keys) > 0:
        print(f"instantiate: keys missing {missing_keys}")

    return instance(*args, **init_dict)
