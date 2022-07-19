from model import MultiViewStereo
from model import visutils
from model.utils import check_2d_position, check_3d_position
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from torch_geometric.transforms import Delaunay


def gaussian(theta, theta0=5, sigma1=1, sigma2=10):
    if theta <= theta0:
        return torch.exp(-(theta-theta0)**2/(2*sigma1**2))
    else:
        return torch.exp(-(theta-theta0)**2/(2*sigma2**2))


def calculate_score(m, p=torch.tensor([[10], [10], [10]])):
    for i in range(m.n):
        for j in range(i+1, m.n):
            theta = (180 / np.pi) * torch.arccos(torch.dot(m.cameras[i].c.flatten() - m.cameras[i].project(p)[:, 0].flatten(),
                                                           m.cameras[j].c.flatten() - m.cameras[i].project(p)[:, 0].flatten()))
            print(torch.dot(m.cameras[i].project(p)[:, 0].flatten(), m.cameras[i].c.flatten()))
            print(f"View {i}, {j}: {theta}, {gaussian(theta)}")


if __name__ == '__main__':
    model = MultiViewStereo(num_img=49,
                            available_list=[3,4,5,7,8,10,11,12,15,38,41,48],
                            data_dir='./model/dtu_data')

    # model.data_preprocess(convert_heic=True)
    # model.generate_params()

    # model.generate_depth_map()
    model.load_cameras()
    model.generate_point_cloud()
    # model.mesh_cleaning()
    model.triangulation()
    model.saving_cloud()
    # print(model.depth[0].shape)

    # torch.save(model.point_clouds[0], 'point_cloud.txt')
    # torch.save(model.meshes[0], 'mesh.txt')
    # print(model.cameras[0].c, model.cameras[0].f)
    # print(model.depth[0])

    # objp = torch.zeros((7*7, 3))
    # objp[:, :2] = torch.from_numpy(2.8*np.mgrid[0:7, 0:7].T.reshape(-1, 2))
    # check_2d_position(model, objp.T)
    # check_3d_position(model, objp.T)


    # import shutil

    # for i in range(49):
    # print()

    #     img = cv2.imread(f"/home/chen_zhang/Desktop/{i}.png")
    #     img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_NEAREST)

    #     if j <= 9:
    #         shutil.copyfile(f"/home/chen_zhang/PythonProjects/3DVNet-CS117/model/dtu_data/cam/{j}_cam.txt",
    #                         f"/media/chen_zhang/AzirZhang/data/dtu/Cameras/0000000{j}_cam.txt")
    #     else:
    #         shutil.copyfile(f"/home/chen_zhang/PythonProjects/3DVNet-CS117/model/dtu_data/cam/{j}_cam.txt",
    #                         f"/media/chen_zhang/AzirZhang/data/dtu/Cameras/000000{j}_cam.txt")
    # import os
    # for idx in [2]:
    #     for j in range(49):
    #         if j < 9:
    #             for k in [0, 1, 2, 4, 5, 6]:
    #                 os.remove(f'./model/dtu_data/object/scan{idx}/rect_00{j+1}_{k}_r5000.png')
    #             os.remove(f'./model/dtu_data/object/scan{idx}/rect_00{j+1}_max.png')
    #             os.rename(f'./model/dtu_data/object/scan{idx}/rect_00{j+1}_3_r5000.png',
    #                       f'./model/dtu_data/object/scan{idx}/{j}.png')
    #         else:
    #             for k in [0, 1, 2, 4, 5, 6]:
    #                 os.remove(f'./model/dtu_data/object/scan{idx}/rect_0{j+1}_{k}_r5000.png')
    #             os.remove(f'./model/dtu_data/object/scan{idx}/rect_0{j+1}_max.png')
    #             os.rename(f'./model/dtu_data/object/scan{idx}/rect_0{j+1}_3_r5000.png',
    #                       f'./model/dtu_data/object/scan{idx}/{j}.png')

    # calculate_score(model)

