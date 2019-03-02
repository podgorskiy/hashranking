import time


def timer(f):
    """Decorator for timeing method execution time"""
    def __wrapper(*args, **kw):
        time_start = time.time()
        result = f(*args, **kw)
        time_end = time.time()
        print('func:%r  took: %2.4f sec' % (f.__name__, time_end - time_start))
        __wrapper.time = time_end - time_start
        return result
    return __wrapper

