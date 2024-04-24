from . import utils
import cv2

# DATA_PATH = "/Users/daniel/Documents/code/python/AutoDriveTasks/ColorCorrection/dataset_2024_04_19_East"
# DATA_PATH = "/Users/daniel/Documents/code/python/AutoDriveTasks/ColorCorrection/dataset_2024_04_19_West"
# DATA_PATH = "/Users/daniel/Documents/code/python/AutoDriveTasks/ColorCorrection/dataset_2024_04_19_North"
DATA_PATH = "/Users/daniel/Documents/code/python/AutoDriveTasks/ColorCorrection/dataset_2024_04_19_South"

if __name__ == '__main__':
    imgs, pcds, imus, calib = utils.load_data(DATA_PATH)
    for img, pcd in zip(imgs, pcds):
        # projected_img = utils.draw_lidar_on_image(img, pcd, calib)
        cv2.imshow('img', img)
        cv2.waitKey(0)

        img = utils.auto_exposure(img, pcd)
        cv2.imshow('img', img)
        cv2.waitKey(0)

        img = utils.auto_white_balancing(img)
        cv2.imshow('img', img)
        cv2.waitKey(0)