from luafun.utils.ring import RingBuffer
import os
import re


FUNCTION = re.compile(r'define (?P<rtype>.*) @(?P<fname>.*)\((?P<args>.*)\) (.*)')


# Have you ever dealt with file so big you needed
# to reimplement tail | head | grep
class ReadHugeFile:
    def __init__(self, filename, prev=50, lines=101):
        self.prev = prev
        self.previous = RingBuffer(prev, None)
        self.filename = filename
        self.lines = prev
        self.max = max(lines, prev + 1)
        self.fname = None

    def print_previous(self):
        if self.previous is None:
            return

        for p in self.previous:
            print(p, end='')

    def find(self, regex, max_match=None):
        print(os.path.getsize(self.filename))
        regex = re.compile(regex)

        print_next = False
        match_count = 0
        with open(self.filename, 'r') as f:
            for i, line in enumerate(f.readlines()):
                result = FUNCTION.match(line)

                if result:
                    msg = result.groupdict()
                    self.fname = msg.get('fname')

                if self.lines >= self.max:
                    self.lines = self.prev
                    print_next = False
                    print('<<<<<<<<')

                    if max_match is not None and match_count >= max_match:
                        break

                if print_next:
                    print(line, end='')
                    self.lines += 1
                    continue

                if regex.search(line) is not None:
                    match_count += 1
                    print(f'>>>>>>> Match (line: {i}) (function: {self.fname})')
                    self.print_previous()
                    print('>>>')
                    print(line, end='')
                    print('<<<')
                    self.previous = RingBuffer(self.prev, None)
                    print_next = True
                    continue

                self.previous.append(line)


if __name__ == '__main__':
    filename = '/media/setepenre/local/SteamLibraryLinux/steamapps/common/dota 2 beta/game/dota/bin/linuxsteamrt64/libserver.so.ll'

    obj = ReadHugeFile(filename)
    # obj.find(r'%911 = call i64 @function_3498330')

    obj.find('dec_label_pc_34a8412')

    # obj.find('@global_var_40035c5', max_match=4)

