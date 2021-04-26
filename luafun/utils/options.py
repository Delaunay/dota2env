import copy
import os
import json


def select(a, b):
    if a is not None:
        return a
    return b


def flatten(dictionary):
    """Turn all nested dict keys into a {key}.{subkey} format"""
    def _flatten(dictionary):
        if dictionary == {}:
            return dictionary

        key, value = dictionary.popitem()
        if not isinstance(value, dict) or not value:
            new_dictionary = {key: value}
            new_dictionary.update(_flatten(dictionary))
            return new_dictionary

        flat_sub_dictionary = _flatten(value)
        for flat_sub_key in list(flat_sub_dictionary.keys()):
            flat_key = key + '.' + flat_sub_key
            flat_sub_dictionary[flat_key] = flat_sub_dictionary.pop(flat_sub_key)

        new_dictionary = flat_sub_dictionary
        new_dictionary.update(_flatten(dictionary))
        return new_dictionary

    return _flatten(copy.deepcopy(dictionary))


# Loaded options
_options = {}

# Options that are currently in use
_active_options = {}


def load_configuration(file_name):
    # loading a config does not make active
    # a config only become active when it is actually used somewhere
    global _options

    options = json.load(open(file_name, 'r'))
    _options = flatten(options)


def set_option(name, value):
    global _options
    _options[name] = value


def get_active_options():
    global _active_options
    return _active_options


def fetch_option(name, default, type=str) -> str:
    """Look for an option locally and using the environment variables
    Environment variables are use as the ultimate overrides
    """
    global _active_options

    env_name = name.upper().replace('.', '_')
    value = os.getenv(f'LUAFUN_{env_name}', None)

    if value is None:
        value = _options.get(name, default)

    if value is None:
        return value

    return type(value)


def option(name, default, type=str) -> str:
    global _active_options

    value = fetch_option(name, default, type)
    _active_options[name] = value

    return value


def datapath(folder):
    """Returns the data path, defaults to luafun/../data"""
    dirname = os.path.dirname(__file__)
    default_data_dir = os.path.join(dirname, '..', '..', 'data')

    datadir = option('data.path', default_data_dir, type=str)
    return os.path.join(datadir, folder)


def datafile(folder, name):
    return os.path.join(datapath(folder), name)
