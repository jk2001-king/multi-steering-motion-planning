"""Load trusted hospital helpers by file, independent of embedded import caches.

Only these helpers' private builtins resolve hospital_* siblings explicitly.
Standard-library/third-party imports retain the host's normal importer. No
sys.path, sys.meta_path, sys.modules or global builtins mutation is required,
even on error. Helpers retain private dependency namespaces for later calls.
This loader does not execute a trial or modify control/physics parameters.
"""

import builtins
from pathlib import Path
from types import ModuleType


def load(path):
    path=Path(path).resolve()
    if not path.is_file():raise FileNotFoundError(path)
    folder=path.parent;modules={};normal_import=builtins.__import__

    def private_import(name,globals=None,locals=None,fromlist=(),level=0):
        # The host may intercept bare project imports and ignore a newly added
        # sys.path entry. Read only trusted siblings, without consulting it.
        if level==0 and name.startswith("hospital_") and name.isidentifier():
            sibling=folder/(name+".py")
            if sibling.is_file():return load_module(name,sibling)
        return normal_import(name,globals,locals,fromlist,level)

    def load_module(name,source):
        if name in modules:return modules[name]
        module=ModuleType(name)
        namespace=module.__dict__
        namespace.update(__file__=str(source),__package__="",__cached__=None,
            __builtins__={**vars(builtins),"__import__":private_import})
        modules[name]=module
        try:
            exec(compile(source.read_text(encoding="utf-8"),str(source),"exec"),namespace)
        except BaseException:
            modules.pop(name,None)
            raise
        return module

    return load_module(path.stem,path).__dict__
