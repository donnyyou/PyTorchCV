import os
import cv2
import numpy

LABEL_PATH = "/home/mark/Dataset/CityScapes/gtFine/val/"
RESULT_PATH = "./results/"
SAVE_PATH = "./diff/fp/"
color_list = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
color_map = [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153),
             (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
             (255,  0,  0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)]


def visualize_fn():
    folders = os.listdir(LABEL_PATH)
    for f in folders:
        label_path = LABEL_PATH + f + "/"
        pre_path = RESULT_PATH + f + "/"
        save_path = SAVE_PATH + f + "/"
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        names = os.listdir(pre_path)
        for n in names:
            print n
            read_name = pre_path + n
            label_name = label_path + n[:-15] + "gtFine_labelIds.png"

            label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
            pre = cv2.imread(read_name, cv2.IMREAD_GRAYSCALE)
            result = numpy.zeros(shape=(pre.shape[0], pre.shape[1], 3), dtype=numpy.uint8)

            for i, c in enumerate(color_list):
                mask0 = numpy.zeros_like(label, dtype=numpy.uint8)
                mask1 = numpy.zeros_like(label, dtype=numpy.uint8)
                mask0[label[:] == c] += 1
                mask0[pre[:] == c] += 1
                mask1[pre[:] == c] += 1
                result[mask0[:] == 1] = color_map[i]
                result[mask1[:] == 1] = (0, 0, 0)
            result[label[:] < 7] = (0, 0, 0)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            cv2.imwrite(save_path + n, result)
            cv2.waitKey(1)


def visualize_fp():
    folders = os.listdir(LABEL_PATH)
    for f in folders:
        label_path = LABEL_PATH + f + "/"
        pre_path = RESULT_PATH + f + "/"
        save_path = SAVE_PATH + f + "/"
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        names = os.listdir(pre_path)
        for n in names:
            print n
            read_name = pre_path + n
            label_name = label_path + n[:-15] + "gtFine_labelIds.png"

            label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
            pre = cv2.imread(read_name, cv2.IMREAD_GRAYSCALE)
            result = numpy.zeros(shape=(pre.shape[0], pre.shape[1], 3), dtype=numpy.uint8)

            for i, c in enumerate(color_list):
                mask0 = numpy.zeros_like(label, dtype=numpy.uint8)
                mask1 = numpy.zeros_like(label, dtype=numpy.uint8)
                mask0[label[:] == c] += 1
                mask0[pre[:] == c] += 1
                mask1[label[:] == c] += 1
                result[mask0[:] == 1] = color_map[i]
                result[mask1[:] == 1] = (0, 0, 0)
            result[label[:] < 7] = (0, 0, 0)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            cv2.imwrite(save_path + n, result)
            cv2.waitKey(1)

if __name__ == "__main__":
    # visualize_fn()
    visualize_fp()
