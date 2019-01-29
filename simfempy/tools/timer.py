import time

#=================================================================#
class Timer():
    def __init__(self, name=''):
        self.name = name
        self.tlast = time.time()
        self.data = {}
    def add(self, name):
        t = time.time()
        self.data[name] = t - self.tlast
        self.tlast = t
    def __del__(self):
        tall = sum(self.data.values())
        print("Timer {:12s} {:12.2e}".format(self.name, tall))
        for name, t in self.data.items():
            print("{:12s}: {:5.1f}%  ({:8.2e})".format(name, 100*t/tall, t))
