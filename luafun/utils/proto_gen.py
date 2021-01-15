import logging

from luafun.utils.proto_parser import ProtoParser

log = logging.getLogger(__name__)


class ProtoGenerator:
    # Generate nice python code from a proto file
    # the main purpose is auto complete when looking for some data
    def __init__(self, filename):
        self.parser = ProtoParser(filename)
        self.parser.parse()

        # for k, v in self.parser.path.items():
        #     print(f'{k:>40} {v}')
        self.object = [
            'from dataclasses import dataclass',
            'from enum import IntEnum',
            'from typing import Optional, List',
            '',
        ]


    def get_type(self, name):
        if name in self.parser.path:
            return self.parser.path[name]
        
        if name == 'int32' or name == 'uint32':
            return 'int'

        if name == 'string':
            return 'str'
        
        return name

    def generate(self, filename):
        for name in self.parser.type_array:
            for message in self.parser.messages:
                _, tname, _ = message

                if name == tname:
                    print(name, tname)
                    self.generate_entity(message)

        with open(filename, 'w') as f:
            f.write('\n'.join(self.object))
        

    def generate_entity(self, m):
        kind, name, fields = m

        if kind == 'M':
            return self.generate_message(name, fields)

        if kind == 'E':
            return self.genererate_enum(name, fields)

        if kind == 'O':
            return self.generate_oneof(name, fields)

        print(kind)

    def genererate_enum(self, name, fields):
        path = self.parser.old.get(name)
        elems = [
            f'# {path}',
            f'class {name}(IntEnum):'
        ]
    
        for field in fields:
            name, value = field
            elems.append(f'    {name} = {value}')

        elems.append('')
        self.object.append('\n'.join(elems))
        
    def generate_field(self, typeid, name, type_name, qualifier, union=False):
        type = self.get_type(type_name)
        if qualifier == 'repeated':
            type = f'List[{type}]'
        if qualifier == 'optional' or union:
            type = f'Optional[{type}]'
        return f'    {name}: {type}'

    def generate_oneof(self, name, fields):
        elems = [
            '# @union',
            '@dataclass',
            f'class {name}:'
        ]

        for field in fields:
            if field[0] == 'F':
                elems.append(self.generate_field(*field[1:], union=True))

        elems.append('')
        self.object.append('\n'.join(elems))

        var_name = name[0].lower() + name[1:]
        return f'    {var_name}: {name}'

    def generate_message(self, name, fields):
        path = self.parser.old.get(name)

        elems = [
            f'# {path}',
            '@dataclass',
            f'class {name}:'
        ]
    
        for field in fields:
            if field[0] == 'F':
                elems.append(self.generate_field(*field[1:]))

            elif field[0] == 'O':
                elems.append(self.generate_oneof(*field[1:]))

        if len(fields) == 0:
            elems.append('    pass')

        elems.append('')
        self.object.append('\n'.join(elems))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    p = ProtoGenerator('C:/Users/Newton/work/luafun/luafun/game/dota2/dota_gcmessages_common_bot_script.proto')
    p.generate('C:/Users/Newton/work/luafun/luafun/game/dota2/state_types.py')

