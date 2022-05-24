import torchvision.datasets as dset
import torchvision.transforms as transforms
from typing import Callable, Optional, Tuple, Union
import pdb
import tarfile
import pickle
import numpy as np
from pathlib import Path
import os, io
from tqdm import tqdm
import PIL.Image
import json
### copy from https://github.com/autonomousvision/stylegan_xl/blob/aa6531372d3517cfe3157631093191e8cfea2aaf/dataset_tool.py
'''
def _data_transforms_cifar10():
    """Get data transforms for cifar10."""

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    return train_transform, valid_transform

train_transform = transforms.Compose([
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
root = 'pretrained/data'

train_data = dset.CIFAR10(
            root=root, train=True, download=True, transform=train_transform)

for (x, _ ) in train_data:
    pdb.set_trace()
'''
def maybe_min(a: int, b: Optional[int]) -> int:
    if b is not None:
        return min(a, b)
    return a

def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]

def open_cifar10(tarball: str, *, max_images: Optional[int]):
    images = []
    labels = []

    with tarfile.open(tarball, 'r:gz') as tar:
        for batch in range(1, 6):
            member = tar.getmember(f'cifar-10-batches-py/data_batch_{batch}')
            with tar.extractfile(member) as file:
                data = pickle.load(file, encoding='latin1')
            images.append(data['data'].reshape(-1, 3, 32, 32))
            labels.append(data['labels'])

    images = np.concatenate(images)
    labels = np.concatenate(labels)
    images = images.transpose([0, 2, 3, 1]) # NCHW -> NHWC
    assert images.shape == (50000, 32, 32, 3) and images.dtype == np.uint8
    assert labels.shape == (50000,) and labels.dtype in [np.int32, np.int64]
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 0 and np.max(labels) == 9

    max_idx = maybe_min(len(images), max_images)

    def iterate_images():
        for idx, img in enumerate(images):
            yield dict(img=img, label=int(labels[idx]))
            if idx >= max_idx-1:
                break

    return max_idx, iterate_images()

def open_dest(dest: str) -> Tuple[str, Callable[[str, Union[bytes, str]], None], Callable[[], None]]:
    dest_ext = file_ext(dest)

    if dest_ext == 'zip':
        if os.path.dirname(dest) != '':
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        zf = zipfile.ZipFile(file=dest, mode='w', compression=zipfile.ZIP_STORED)
        def zip_write_bytes(fname: str, data: Union[bytes, str]):
            zf.writestr(fname, data)
        return '', zip_write_bytes, zf.close
    else:
        # If the output folder already exists, check that is is
        # empty.
        #
        # Note: creating the output directory is not strictly
        # necessary as folder_write_bytes() also mkdirs, but it's better
        # to give an error message earlier in case the dest folder
        # somehow cannot be created.
        if os.path.isdir(dest) and len(os.listdir(dest)) != 0:
            error('--dest folder must be empty')
        os.makedirs(dest, exist_ok=True)

        def folder_write_bytes(fname: str, data: Union[bytes, str]):
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            with open(fname, 'wb') as fout:
                if isinstance(data, str):
                    data = data.encode('utf8')
                fout.write(data)
        return dest, folder_write_bytes, lambda: None



def compute_is(opts, num_gen, num_splits):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(no_output_bias=True) # Match the original implementation by not applying bias in the softmax layer.

    gen_probs = metric_utils.compute_feature_stats_for_generator(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        capture_all=True, max_items=num_gen).get_all()

    if opts.rank != 0:
        return float('nan'), float('nan')

    scores = []
    for i in range(num_splits):
        part = gen_probs[i * num_gen // num_splits : (i + 1) * num_gen // num_splits]
        kl = part * (np.log(part) - np.log(np.mean(part, axis=0, keepdims=True)))
        kl = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl))
    return float(np.mean(scores)), float(np.std(scores))


source = 'pretrained/data/cifar-10-python.tar.gz'
dest = 'pretrained/data/cifar10'
PIL.Image.init()
num_files, input_iter = open_cifar10(source, max_images = None)
archive_root_dir, save_bytes, close_dest = open_dest(dest)

dataset_attrs = None

labels = []

for idx, image in tqdm(enumerate(input_iter), total=num_files):
    idx_str = f'{idx:08d}'
    archive_fname = f'{idx_str[:5]}/img{idx_str}.png'

    img = (image['img'])

    if img is None:
        continue

    channels = img.shape[2] if img.ndim == 3 else 1
    cur_image_attrs = {
        'width': img.shape[1],
        'height': img.shape[0],
        'channels': channels
    }
    if dataset_attrs is None:
        dataset_attrs = cur_image_attrs
        width = dataset_attrs['width']
        height = dataset_attrs['height']
        if width != height:
            error(f'Image dimensions after scale and crop are required to be square.  Got {width}x{height}')
        if dataset_attrs['channels'] not in [1, 3]:
            error('Input images must be stored as RGB or grayscale')
        if width != 2 ** int(np.floor(np.log2(width))):
            error('Image width/height after scale and crop are required to be power-of-two')
    elif dataset_attrs != cur_image_attrs:
        err = [f'  dataset {k}/cur image {k}: {dataset_attrs[k]}/{cur_image_attrs[k]}' for k in dataset_attrs.keys()] # pylint: disable=unsubscriptable-object
        error(f'Image {archive_fname} attributes must be equal across all images of the dataset.  Got:\n' + '\n'.join(err))

    # Save the image as an uncompressed PNG.
    img = PIL.Image.fromarray(img, { 1: 'L', 3: 'RGB' }[channels])
    image_bits = io.BytesIO()
    img.save(image_bits, format='png', compress_level=0, optimize=False)
    save_bytes(os.path.join(archive_root_dir, archive_fname), image_bits.getbuffer())
    labels.append([archive_fname, image['label']] if image['label'] is not None else None)

metadata = { 'labels': labels if all(x is not None for x in labels) else None }
save_bytes(os.path.join(archive_root_dir, 'dataset.json'), json.dumps(metadata))
close_dest()

print("OK")