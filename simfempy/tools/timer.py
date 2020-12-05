import time

#=================================================================#
class Timer():
    def __init__(self, name='', verbose=False):
        self.name = name
        self.verbose = verbose
        self.tlast = time.time()
        self.data = {}
        self.counter = 0
    def __repr__(self):
        tall = sum(self.data.values())
        repr = f"\nTimer({self.name:}) total = {tall:8.2e}\n"
        for name, t in self.data.items():
            repr += f"\t{name:12s}: {100*t/tall:5.1f}%  ({t:8.2e})\n"
        return repr

    def items(self): return self.data.items()
    def add(self, name=None):
        if name is None: name = str(self.counter); self.counter += 1
        t = time.time()
        if name not in self.data: self.data[name] = 0
        self.data[name] += t - self.tlast
        self.tlast = t

    def __del__(self):
        if self.verbose: print(self)

    def reset(self, name):
        self.data[name] = 0
    def reset_all(self):
        for name in self.data: self.data[name] = 0
        self.tlast = time.time()
