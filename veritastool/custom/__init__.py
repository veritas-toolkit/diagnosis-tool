# from os.path import dirname, basename, isfile, join
# import glob
# modules = glob.glob(join(dirname(__file__), "*.py"))
# __all__ = [basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]

# #
# for py in [basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]:
#     mod = __import__('.'.join(['custom', py]))
#     classes = [getattr(mod, x) for x in dir(mod) if isinstance(getattr(mod, x), type)]
#     # print(classes)
#     for cls in classes:
#         setattr(sys.modules[__name__], cls.__name__, cls)

# print(__all__)

from .LRwrapper import LRwrapper
from .CMmodelwrapper import CMmodelwrapper
from .modelwrapper import ModelWrapper
from .newmetric import NewMetric
from .newmetric_child import NewMetricChild
from .newmetric_child_perf import PerfNewMetricChild
