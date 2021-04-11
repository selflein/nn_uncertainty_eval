import functools


def partialclass(cls, *args, **kwds):
    """Partial init function application for classes based on
    https://stackoverflow.com/questions/38911146/python-equivalent-of-functools-partial-for-a-class-constructor
    """

    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwds)

    return NewCls
