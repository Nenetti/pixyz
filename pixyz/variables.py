import torch
import sympy
from IPython.display import Math
import pixyz
from collections import UserDict


class Variables:

    def __init__(self, **kwargs):
        """
        Args:
            # **kwargs (dict[str, torch.Tensor]): variables

        """
        self._x_dict = kwargs

    def get_variables(self, keys):
        """
        Get variables from `Variables` specified by `keys`.

        Args:
            keys (str or list[str]):

        Returns:
            Variables: keys and values

        """
        if isinstance(keys, str):
            return Variables(**{keys: self._x_dict[keys]})
        else:
            return Variables(**dict((key, self._x_dict[key]) for key in keys if key in self._x_dict.keys()))

    def get_values(self, keys):
        """
        Get values from `Variables` specified by `keys`.

        Args:
            keys (str or list[str]):

        Returns:
            list: only values

        """
        if isinstance(keys, str):
            return [self._x_dict[keys]]
        elif isinstance(keys, list):
            return [self._x_dict[key] for key in keys if key in self._x_dict.keys()]
        else:
            ValueError()

    def get_value(self, key):
        """
        Get values from `Variables` specified by `keys`.

        Args:
            keys (str or list[str]):

        Returns:
            list: only values

        """
        if isinstance(key, str):
            return self._x_dict[key]
        elif isinstance(key, list) and len(key) == 1:
            return self._x_dict[key[0]]
        else:
            ValueError()

    def delete_dict_values(self, keys):
        """
        Delete values from `dicts` specified by `keys`.

        Args:
            keys (str or list[str]):

        Returns
            Variables:

        """
        if isinstance(keys, str):
            return Variables(**dict((key, value) for key, value in self._x_dict.items() if key != keys))
        else:
            return Variables(**dict((key, value) for key, value in self._x_dict.items() if key not in keys))

    def detach_dict(self):
        """
        Detach all values.

        Returns
            Variables:

        """
        return Variables(**{key: value.detach() for key, value in self._x_dict.items()})

    def replace_dict_keys(self, replace_keys):
        """
        Replace values in `dicts` according to `replace_list_dict`.

        Args:
            replace_keys (dict[str,str]): Dictionary.

        Returns:
            Variables:

        """
        return Variables(**dict((replace_keys[key], value) if key in replace_keys else (key, value) for key, value in self._x_dict.items()))

    def replace_dict_keys_split(self, replace_keys):
        """
        Replace values in `dicts` according to :attr:`replace_list_dict`.
        Replaced dict is splitted by :attr:`replaced_dict` and :attr:`remain_dict`.

        Args:
            replace_keys (dict[str,str]): Dictionary.

        Returns:
            Variables:

        """
        replaced_dict = Variables(**dict((replace_keys[key], value) for key, value in self._x_dict.items() if key in replace_keys))
        remain_dict = Variables(**dict((key, value) for key, value in self._x_dict.items() if key not in replace_keys))

        return replaced_dict, remain_dict

    def update(self, variables):
        self._x_dict.update(variables._x_dict)

    def keys(self):
        return self._x_dict.keys()

    def clear(self):
        self._x_dict.clear()

    def __getitem__(self, key):
        return self._x_dict[key]
