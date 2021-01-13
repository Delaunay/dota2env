import logging

log = logging.getLogger(__name__)


class ProtoParser:
    def __init__(self, filename):
        with open(filename, 'r') as f:
            data = f.read()

        lines = data.split('\n')
        tokens = []
        for line in lines:
            if line.startswith('//'):
                continue

            if line.startswith('syntax'):
                continue

            if line.startswith('option'):
                continue

            line = line.replace('\t', ' ')
            tokens.extend(line.split(' '))

        self.tokens = []
        for t in tokens:
            if t not in ('', ' '):
                self.tokens.append(t)

        self.nested_message = []
        self.path = dict()
        self.messages = []
        self.pos = 0

    def next(self):
        self.pos += 1
        return self.token()

    def token(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def has_tokens(self):
        return self.pos < len(self.tokens)

    def parse(self):
        while self.has_tokens():
            tok = self.token()

            if tok == 'message':
                self.parse_message()

            elif tok == 'enum':
                self.parse_enum()

            else:
                print(f'dont know how to parse `{tok}`')

    def parse_oneof(self):
        self.expect('oneof')
        name = self.next()

        self.next(), self.expect('{')
        fields = []

        tok = self.next()
        while tok != '}':
            fields.append(self.parse_field())
            tok = self.token()

        self.expect('}'), self.next()
        return ('O', name, fields)

    def parse_field(self):
        qualifier = ''
        tok = self.token()

        if tok in ('optional', 'required', 'repeated'):
            qualifier = tok
            field_type = self.next() 
        elif tok == 'oneof':
            return self.parse_oneof()
        else:
            field_type = tok
        
        field_name = self.next()

        self.next(), self.expect('=')
        field_id = self.next() 

        # ; might be included in the id
        tok = self.next()
        if tok == ';':
            self.next()
        # start of the [default ....]; stuff
        elif tok[0] == '[':
            tok = self.token()
            while tok[-1] != ';':
                tok = self.next()
            self.next()

        elif field_id[-1] == ';':
            field_id = field_id[:-1]

        log.debug(f'parse field {qualifier} {field_name}: {field_type} = {field_id}')
        return field_id, field_name, field_type, qualifier

    def parse_enum_field(self):
        name = self.token()
        self.next(), self.expect('=')
        value = self.next()

        # ; might be included in the id
        tok = self.next()
        if tok == ';':
            self.next()
        elif value[-1] == ';':
            value = value[:-1]

        return name, value

    def expect(self, c):
        t = self.token()
        assert t == c, f'Expected `{c}` got `{t}`'

    def parse_enum(self):
        self.expect('enum')
        name = self.next()

        log.debug(f'>> parsing enum {name}')
        self.path['.'.join(self.nested_message)] = name

        tok = self.next(), self.expect('{')
        tok = self.next()

        fields = []
        while tok != '}':
            fname, fvalue = self.parse_enum_field()
            fields.append((fname, fvalue))
            tok = self.token()

        self.expect('}'), self.next()
        self.messages.append(('E', name, fields))
        log.debug(f'<< {name}')

    def parse_message(self):
        self.expect('message')
        name = self.next()

        log.debug(f'>> parsing message {name}')
        self.nested_message.append(name)
        self.path['.'.join(self.nested_message)] = name

        self.next(), self.expect('{')
        tok = self.next()

        fields = []
        while tok != '}':
            if tok == 'message':
                self.parse_message()
                tok = self.token()
                continue

            if tok == 'enum':
                self.parse_enum()
                tok = self.token()
                continue

            fields.append(self.parse_field())
            tok = self.token()
        
        self.expect('}'), self.next()
        self.messages.append(('M', name, fields))
        log.debug(f'<< {name}')



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    p = ProtoParser('C:/Users/Newton/work/luafun/luafun/game/dota2/dota_gcmessages_common_bot_script.proto')
    p.parse()
    print(p.messages)
