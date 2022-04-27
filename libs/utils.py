import numpy as np


class DictObj:
    def __init__(self, in_dict: dict):
        assert isinstance(in_dict, dict)

        self.in_dict = in_dict

        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
                setattr(self, key, [DictObj(x) if isinstance(x, dict) else x for x in val])
            else:
                setattr(self, key, DictObj(val) if isinstance(val, dict) else val)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, DictObj):
            return False

        for key, value in vars(self).items():
            if callable(value):
                continue
            if o.__getattribute__(key) != value:
                return False
        return True

    def to_json(self):
        return {}


def list_of_ndarray_tolist(l):
    """
    Convert a list of ndarray's to a plain Python list

    This is useful if you want to dump the list via JSON or similar.

    @param l The list to convert
    @return l as a list of same dimensions but without any ndarray in it
    """
    if isinstance(l, list) and (
            len(l) > 0 and (isinstance(l[0], list) or isinstance(l[0], float) or isinstance(l[0], int))):
        ret_unsafe = l
    else:
        # Convert to a list first, this list may contain int64 and is therefore not safe for json
        ret_unsafe = list(map(lambda x: x.tolist(), l)) if len(l) > 0 else []

    ret = []
    for x in ret_unsafe:
        if type(x) == type(np.int64):
            ret.append(int(x))
        else:
            ret.append(x)
    return ret
