from glob import glob
import logging
import os

log = logging.getLogger(__name__)


def fetch_directory_factories(base_module, base_file_name, attr_name='builders'):
    factories = {}
    module_path = os.path.dirname(os.path.abspath(base_file_name))
    for module_path in glob(os.path.join(module_path, '[A-Za-z]*.py')):
        module_file = module_path.split(os.sep)[-1]

        if module_file == base_file_name:
            continue

        module_name = module_file.split(".py")[0]

        try:
            module = __import__(".".join([base_module, module_name]), fromlist=[''])
        except ImportError as e:
            log.warning(f'Could not import {module_name} from {base_file_name} because of {e}')
            continue

        if hasattr(module, attr_name):
            builders = getattr(module, attr_name)
            if not isinstance(builders, dict):
                builders = {module_name: builders}

            for key, builder in builders.items():
                factories[key] = builder

    return factories


