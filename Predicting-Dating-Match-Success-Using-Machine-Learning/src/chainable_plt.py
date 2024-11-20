class ChainablePlt:
    def __init__(self, plt):
        self.plt = plt
        self._wrap_functions()

    def _wrap_functions(self):
        for name in dir(self.plt):
            attr = getattr(self.plt, name)
            if callable(attr):
                self._create_chainable_function(name, attr)

    def _create_chainable_function(self, name, func):
        def chainable_func(*args, **kwargs):
            _ = func(*args, **kwargs)
            if name == "show":
                return
            return self

        setattr(self, name, chainable_func)

    def __getattr__(self, name):
        return getattr(self.plt, name)

    def txylabel(self, t, x, y):
        self.plt.title(t)
        self.plt.xlabel(x)
        self.plt.ylabel(y)
        return self.plt

    def xylim(self, x1, x2, y1=None, y2=None):
        self.plt.xlim(x1, x2)
        if y1 and y2:
            self.plt.ylim(y1, y2)
        return self.plt
