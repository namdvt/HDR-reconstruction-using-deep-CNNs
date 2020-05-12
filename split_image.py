import cv2
import os
import pandas as pd
from tqdm import tqdm


def split_ldr(root, size):
    image = cv2.imread(root + '/input_2_aligned.tif')

    height = image.shape[0]
    width = image.shape[1]

    for y in range(0, height, size):
        for x in range(0, width, size):
            tiles = image[y:y + size, x:x + size]
            if (tiles.shape[0] == size) and (tiles.shape[1] == size):
                cv2.imwrite(root + '/' + str(x) + '_' + str(y) + ".png", tiles)
    os.remove(root + '/input_2_aligned.tif')


def split_hdr(root, size):
    image = cv2.imread(root + '/ref_hdr_aligned.hdr', cv2.IMREAD_ANYDEPTH)

    height = image.shape[0]
    width = image.shape[1]

    for y in range(0, height, size):
        for x in range(0, width, size):
            tiles = image[y:y + size, x:x + size]
            if (tiles.shape[0] == size) and (tiles.shape[1] == size):
                cv2.imwrite(root + '/' + str(x) + '_' + str(y) + ".hdr", tiles)
    os.remove(root + '/ref_hdr_aligned.hdr')


if __name__ == '__main__':
    for folder in tqdm(os.listdir('data/train')):
        split_ldr(os.path.join('data/train', folder), size=256)
        split_hdr(os.path.join('data/train', folder), size=256)

    df = pd.DataFrame()
    f = open('data/train/annotations.txt', 'w+')
    for folder in os.listdir('data/train'):
        path = os.path.join('data/train', folder)
        if os.path.isfile(path):
            continue

        name_list = list()
        for file in os.listdir(path):
            name = file.split('.')[0]
            if name not in name_list:
                f.write(path + '/' + name + '.png' + '\t' + path + '/' + name + '.hdr\n')
                name_list.append(name)
    f.close()