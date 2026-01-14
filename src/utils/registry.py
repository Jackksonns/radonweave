# Radonweave (https://github.com/Jackksonns/radonweave): low-dose CT reconstruction/denoising toolkit.
REG = {
    "reconstructor": {},
    "dataset": {},
    "optim": {},
    "loss": {},
    "metric": {},
}

def register(kind, name):
    def deco(obj):
        REG[kind][name] = obj
        return obj
    return deco

def build(kind, name, **kw):
    try:
        return REG[kind][name](**kw)
    except KeyError as e:
        raise KeyError(f"{kind}:{name} not found. choices={list(REG[kind])}") from e
