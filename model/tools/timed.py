import time
import functools

def timed(method):
    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = method(*args, **kwargs)
        end = time.time()
        if (end - start) > 0.0001:
            print(f"[TIMED] {method.__qualname__} took {(end - start):.6f}s")
        return result
    return wrapper

class TimedBase:
    def __init_subclass__(cls):
        super().__init_subclass__()
        for attr_name in dir(cls):
            if attr_name.startswith("_"):
                continue

            raw_attr = getattr(cls, attr_name)
            if isinstance(raw_attr, staticmethod):
                func = raw_attr.__func__
                wrapped = staticmethod(timed(func))
                setattr(cls, attr_name, wrapped)

            elif isinstance(raw_attr, classmethod):
                func = raw_attr.__func__
                wrapped = classmethod(timed(func))
                setattr(cls, attr_name, wrapped)

            elif callable(raw_attr):
                setattr(cls, attr_name, timed(raw_attr))