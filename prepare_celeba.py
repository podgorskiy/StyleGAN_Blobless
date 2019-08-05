import zipfile
import tqdm

from dlutils import download

from scipy import misc
from net import *
import numpy as np
import pickle
import random
import os
from dlutils.pytorch.cuda_helper import *

im_size = 128

directory = 'celeba'

os.makedirs(directory, exist_ok=True)

corrupted = [
    '195995.jpg',
    '131065.jpg',
    '118355.jpg',
    '080480.jpg',
    '039459.jpg',
    '153323.jpg',
    '011793.jpg',
    '156817.jpg',
    '121050.jpg',
    '198603.jpg',
    '041897.jpg',
    '131899.jpg',
    '048286.jpg',
    '179577.jpg',
    '024184.jpg',
    '016530.jpg',
]

download.from_google_drive("0B7EVK8r0v71pZjFTYXZWM3FlRnM", directory=directory)


def center_crop(x, crop_h=128, crop_w=None, resize_w=128):
    # crop the images to [crop_h,crop_w,3] then resize to [resize_h,resize_w,3]
    if crop_w is None:
        crop_w = crop_h # the width and height after cropped
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.)) + 15
    i = int(round((w - crop_w)/2.))
    return misc.imresize(x[j:j+crop_h, i:i+crop_w], [resize_w, resize_w])


archive = zipfile.ZipFile(os.path.join(directory, 'img_align_celeba.zip'), 'r')

names = archive.namelist()

names = [x for x in names if x[-4:] == '.jpg']

count = len(names)
print("Count: %d" % count)

names = [x for x in names if x[-10:] not in corrupted]

random.shuffle(names)

folds = 8
celeba_folds = [[] for _ in range(folds)]

spread_identiteis_across_folds = True


if spread_identiteis_across_folds:
    # Reading indetities
    # Has format of
    # 000001.jpg 2880
    # 000002.jpg 2937
    with open("identity_CelebA.txt") as f:
        lineList = f.readlines()

    lineList = [x[:-1].split(' ') for x in lineList]

    identity_map = {}
    for x in lineList:
        identity_map[x[0]] = int(x[1])

    names = [(identity_map[x.split('/')[1]], x) for x in names]

    class_bins = {}

    for x in names:
        if x[0] not in class_bins:
            class_bins[x[0]] = []
        img_file_name = x[1]
        class_bins[x[0]].append((x[0], img_file_name))

    left_overs = []

    for _class, filenames in class_bins.items():
        count = len(filenames)
        print("Class %d count: %d" % (_class, count))

        count_per_fold = count // folds

        for i in range(folds):
            celeba_folds[i] += filenames[i * count_per_fold: (i + 1) * count_per_fold]

        left_overs += filenames[folds * count_per_fold:]

    leftover_per_fold = len(left_overs) // folds
    for i in range(folds):
        celeba_folds[i] += left_overs[i * leftover_per_fold: (i + 1) * leftover_per_fold]

    for i in range(folds):
        random.shuffle(celeba_folds[i])

    # strip ids
    for i in range(folds):
        celeba_folds[i] = [x[1] for x in celeba_folds[i]]

    print("Folds sizes:")
    for i in range(len(celeba_folds)):
        print(len(celeba_folds[i]))
else:
    count_per_fold = count // folds
    for i in range(folds):
        celeba_folds[i] += names[i * count_per_fold: (i + 1) * count_per_fold]


for i in range(folds):
    images = []
    for x in tqdm.tqdm(celeba_folds[i]):
        imgfile = archive.open(x)
        image = center_crop(misc.imread(imgfile))
        images.append(image)

    output = open(os.path.join(directory, 'data_fold_%d_lod_5.pkl' % i), 'wb')
    pickle.dump(images, output)
    output.close()

    for j in range(5):
        images_down = []

        for image in tqdm.tqdm(images):
            h = image.shape[0]
            w = image.shape[1]
            image = torch.tensor(np.asarray(image, dtype=np.float32).transpose((2, 0, 1))).view(1, 3, h, w)

            image_down = F.avg_pool2d(image, 2, 2).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8)

            image_down = image_down.view(w // 2, h // 2, 3).numpy()
            images_down.append(image_down)

        with open(os.path.join(directory, 'data_fold_%d_lod_%d.pkl' % (i, 5 - j - 1)), 'wb') as pkl:
            pickle.dump(images_down, pkl)
        images = images_down
