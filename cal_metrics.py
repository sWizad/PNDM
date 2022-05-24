from metrics import metric_main
from metrics import metric_utils
import dnnlib
## copy from https://github.com/autonomousvision/stylegan_xl/blob/main/calc_metrics.py

#dataset_kwargs = {'path':'pretrained/data/cifar10'}
dataset_kwargs = dnnlib.EasyDict(class_name='dataset.dataset.ImageFolderDataset', path='pretrained/data/cifar10')
metric = 'is50k'
result_dict = metric_main.calc_metric(metric=metric,dataset_kwargs=dataset_kwargs)

print("OK")