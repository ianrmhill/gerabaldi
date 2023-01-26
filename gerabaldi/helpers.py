"""Internal helper functions used within Gerabaldi to streamline the package."""

import importlib


def _on_demand_import(module: str, pypi_name: str = None):
    try:
        mod = importlib.import_module(module)
        return mod
    except ImportError as e:
        # Module name and pypi package name do not always match, we want to tell the user the package to install
        if not pypi_name:
            pypi_name = module
        hint = f"Trying to use a feature that requires the optional {module} module. " \
               f"Please install package '{pypi_name}' first."

        class FailedImport():
            """By returning a class that raises an error when used, we can try to import modules at the top of each file
            and only raise errors if we try to use methods of modules that failed to import"""
            def __getattr__(self, attr):
                raise ImportError(hint)
        return FailedImport()
