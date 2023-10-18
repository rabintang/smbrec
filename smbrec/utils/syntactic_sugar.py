#  Copyright (c) 2021, The SmbRec Authors.  All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import collections
import functools
import sys
from tensorflow.python.util import tf_decorator


SMBREC_API_NAME = 'smbrec'

_Attributes = collections.namedtuple(
    'ExportedApiAttributes', ['names', 'constants'])

# Attribute values must be unique to each API.
API_ATTRS = {
    SMBREC_API_NAME: _Attributes(
        '_smbrec_api_names',
        '_smbrec_api_constants')
}


class SymbolAlreadyExposedError(Exception):
    """Raised when adding API names to symbol that already has API names."""
    pass


class InvalidSymbolNameError(Exception):
    """Raised when trying to export symbol as an invalid or unallowed name."""
    pass


_NAME_TO_SYMBOL_MAPPING = dict()


class api_export(object):  # pylint: disable=invalid-name
    """Provides ways to export symbols to the TensorFlow API."""

    def __init__(self, *args, **kwargs):  # pylint: disable=g-docs-args
        """Export under the names *args (first one is considered canonical).

        Args:
          *args: API names in dot delimited format.
          **kwargs: Optional keyed arguments.
            overrides: List of symbols that this is overriding
              (those overrided api exports will be removed). Note: passing overrides
              has no effect on exporting a constant.
            api_name: Name of the API you want to generate (e.g. `tensorflow` or
              `estimator`). Default is `tensorflow`.
            allow_multiple_exports: Allow symbol to be exported multiple time under
              different names.
        """
        self._names = args
        self._api_name = kwargs.get('api_name', SMBREC_API_NAME)
        self._overrides = kwargs.get('overrides', [])
        self._allow_multiple_exports = kwargs.get('allow_multiple_exports', False)
        self._validate_symbol_names()

    def _validate_symbol_names(self):
        """Validate you are exporting symbols under an allowed package.

        We need to ensure things exported by tf_export, estimator_export, etc.
        export symbols under disjoint top-level package names.

        For TensorFlow, we check that it does not export anything under subpackage
        names used by components (estimator, keras, etc.).

        For each component, we check that it exports everything under its own
        subpackage.

        Raises:
          InvalidSymbolNameError: If you try to export symbol under disallowed name.
        """
        all_symbol_names = set(self._names)
        if not all(n.startswith(self._api_name) for n in all_symbol_names):
            raise InvalidSymbolNameError(
                'Can only export symbols under package name of component. '
                'e.g. tensorflow_estimator must export all symbols under '
                'tf.estimator')

    def __call__(self, func):
        """Calls this decorator.

        Args:
          func: decorated symbol (function or class).

        Returns:
          The input function with _tf_api_names attribute set.

        Raises:
          SymbolAlreadyExposedError: Raised when a symbol already has API names
            and kwarg `allow_multiple_exports` not set.
        """
        api_names_attr = API_ATTRS[self._api_name].names
        # Undecorate overridden names
        for f in self._overrides:
            _, undecorated_f = tf_decorator.unwrap(f)
            delattr(undecorated_f, api_names_attr)

        _, undecorated_func = tf_decorator.unwrap(func)
        self.set_attr(undecorated_func, api_names_attr, self._names)

        for name in self._names:
            _NAME_TO_SYMBOL_MAPPING[name] = func
        return func

    def set_attr(self, func, api_names_attr, names):
        # Check for an existing api. We check if attribute name is in
        # __dict__ instead of using hasattr to verify that subclasses have
        # their own _tf_api_names as opposed to just inheriting it.
        if api_names_attr in func.__dict__:
            if not self._allow_multiple_exports:
                raise SymbolAlreadyExposedError(
                    'Symbol %s is already exposed as %s.' %
                    (func.__name__, getattr(func, api_names_attr)))  # pylint: disable=protected-access
        setattr(func, api_names_attr, names)

    def export_constant(self, module_name, name):
        """Store export information for constants/string literals.

        Export information is stored in the module where constants/string literals
        are defined.

        e.g.
        ```python
        foo = 1
        bar = 2
        tf_export("consts.foo").export_constant(__name__, 'foo')
        tf_export("consts.bar").export_constant(__name__, 'bar')
        ```

        Args:
          module_name: (string) Name of the module to store constant at.
          name: (string) Current constant name.
        """
        module = sys.modules[module_name]
        api_constants_attr = API_ATTRS[self._api_name].constants

        if not hasattr(module, api_constants_attr):
            setattr(module, api_constants_attr, [])
        # pylint: disable=protected-access
        getattr(module, api_constants_attr).append(
            (self._names, name))


smbrec_export = functools.partial(api_export, api_name=SMBREC_API_NAME)
