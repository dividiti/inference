import time

class Logging():

    def __init__(self):
        self.buffer = [None]*6

    def log_pre_start(self):
        self.buffer[0] = time.time()

    def log_pre_end(self):
        self.buffer[1] = time.time()

    def log_post_start(self):
        self.buffer[2] = time.time()

    def log_post_end(self):
        self.buffer[3] = time.time()

    def log_dec_init(self):
        self.buffer[5] = 0.0

    def log_dec_start(self):
        self.buffer[4] = time.time()

    def log_dec_end(self):
        self.buffer[5] += time.time() - self.buffer[4]

    def get_timings(self):
        return self.buffer[1] - self.buffer[0], \
               self.buffer[3] - self.buffer[2], \
               self.buffer[5]

class DummyLogging():

    def log_pre_start(self):
        pass

    def log_pre_end(self):
        pass

    def log_post_start(self):
        pass

    def log_post_end(self):
        pass

    def log_dec_init(self):
        pass

    def log_dec_start(self):
        pass

    def log_dec_end(self):
        pass

    def get_timings(self):
        return 0.0, 0.0, 0.0



