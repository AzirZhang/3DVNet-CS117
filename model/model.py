from .preprocess import heic_to_png, remove_green_screen, calibrate, remove_rd, crop_and_resize
from .meshutils import writeply

import torch
import numpy as np
import scipy
import time
import os
from scipy.optimize import leastsq
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import re
import os.path as osp
from torch_geometric.data import Data
from torch_geometric.transforms import Delaunay



class Camera:

    def __init__(self, f, c, R, t):
        self.f = f
        self.c = c
        self.R = R
        self.t = t
        torch.set_default_dtype(torch.float64)

    def project(self, pts3):

        assert (pts3.shape[0] == 3)

        # print(self.R.float().dtype)
        # print(pts3.dtype)
        # print(self.t.dtype)
        pts2 = self.R.T @ (pts3 - self.t)
        pts2 = (pts2 / pts2[-1])[:-1] * self.f + self.c

        assert (pts2.shape[1] == pts3.shape[1])
        assert (pts2.shape[0] == 2)

        return pts2

    def update_extrinsics(self, params):
        # print(params)
        self.R = make_rotation(params[0], params[1], params[2])
        # print(type(params))
        self.t = torch.from_numpy(params[3:]).reshape(3, 1)


def make_rotation(rx, ry, rz):
    x, y, z = torch.tensor(rx * np.pi / 180), \
              torch.tensor(ry * np.pi / 180), \
              torch.tensor(rz * np.pi / 180)

    Rx = torch.tensor([[1, 0, 0],
                       [0, torch.cos(x), -torch.sin(x)],
                       [0, torch.sin(x), torch.cos(x)]])
    Ry = torch.tensor([[torch.cos(y), 0, -torch.sin(y)],
                       [0, 1, 0],
                       [torch.sin(y), 0, torch.cos(y)]])
    Rz = torch.tensor([[torch.cos(z), -torch.sin(z), 0],
                       [torch.sin(z), torch.cos(z), 0],
                       [0, 0, 1]])

    return Rz @ Ry @ Rx


def chess_line_rot90(pts2, num, w=7, h=7):
    for _ in range(num):
        pts2 = np.hstack([pts2[:, [w*(h-j-1)+i for j in range(h)]] for i in range(w)])
    return pts2


def residuals(pts3, pts2, cam, params):
    cam.update_extrinsics(params)
    # print(type(cam.project(pts3)))
    # print(type(pts2))
    return (pts2 - cam.project(pts3)).numpy()


class MultiViewStereo:
    def __init__(self, num_img=5, available_list=[0, 1, 2, 3, 4], data_dir='./model/data', raw_data_dir='./model/raw_data'):
        self.n = num_img
        self.available_list = available_list
        self.data_dir = data_dir
        self.raw_data_dir = raw_data_dir
        self.f = 0.
        self.c = torch.tensor([[0.], [0.]])
        self.cameras = [Camera(f=self.f, c=self.c, R=..., t=...) for i in range(self.n)]
        self.calib = {}
        # self.images = []
        # self.mask = []
        self.depth = []
        self.point_clouds = []
        self.meshes = []
        self.params_init = torch.load('model/params_init.txt')
        self.chess_rotate_init = torch.load('model/chess_rotate_init.txt')
        # print(self.params_init)

    def data_preprocess(self, convert_heic=False, resize=True):
        if convert_heic:
            for i in range(self.n):
                heic_to_png(f'{self.raw_data_dir}/chessboard/{i}.HEIC',
                            f'{self.data_dir}/chessboard/{i}.png')
                remove_green_screen(f'{self.data_dir}/chessboard/{i}.png',
                                    f'{self.data_dir}/mask_chessboard/{i}.png', resize)
                heic_to_png(f'{self.raw_data_dir}/object/{i}.HEIC',
                            f'{self.data_dir}/object/{i}.png')
                remove_green_screen(f'{self.data_dir}/object/{i}.png',
                                    f'{self.data_dir}/mask/{i}.png', resize)
        calib = calibrate(f'{self.data_dir}/chessboard/*.png')
        remove_rd(f'{self.data_dir}/chessboard/*.png', f'{self.data_dir}/chessboard/*.png', calib)
        remove_rd(f'{self.data_dir}/object/*.png', f'{self.data_dir}/object/*.png', calib)
        remove_rd(f'{self.data_dir}/mask/*.png', f'{self.data_dir}/mask/*.png', calib)
        self.f = calib['f']
        self.c = calib['c']
        self.calib = calib
        self.cameras = [Camera(self.f, self.c, ..., ...) for _ in range(self.n)]

    def calibrate_pose(self, pts3, pts2, i):
        def func(p):
            norm = np.linalg.norm(residuals(pts3, pts2, self.cameras[i], p), axis=0) ** 2
            # print(p)
            return norm

        # print(self.params_init[i])
        # print(self.params_init[i].dtype)
        param = leastsq(func, self.params_init[i])
        # print(param)
        self.cameras[i].update_extrinsics(param[0])

    def find_extrinsic(self, i):
        img = cv2.imread(f'{self.data_dir}/chessboard/{i}.png')
        ret, cornersL = cv2.findChessboardCorners(img, (7, 7), None)
        if ret:
            pts2 = cornersL.squeeze().T
            n = self.chess_rotate_init[i][0]
            # print(n)
            pts2 = torch.from_numpy(chess_line_rot90(pts2, n))
            # print(pts2.shape)
            # plt.imshow(img)
            # plt.plot(pts2[0, [0, 16]], pts2[1, [0, 16]], 'b')
            # plt.plot(pts2[0, :], pts2[1, :], 'r')
            # plt.show()

            pts3 = torch.zeros((3, 7*7))
            yy, xx = torch.meshgrid(torch.arange(7), torch.arange(7))
            pts3[0, :] = 2.8 * xx.reshape(1, -1)
            pts3[1, :] = 2.8 * yy.reshape(1, -1)

            self.calibrate_pose(pts3, pts2, i)
        else:
            print(i)

    def generate_params(self):
        for i in range(self.n):
            self.find_extrinsic(i)
            # if ret:
            cam = self.cameras[i]
            extrinsic = torch.hstack((cam.R, cam.t))
            extrinsic = torch.vstack((extrinsic, torch.tensor([0., 0., 0., 1.])))
            # print(extrinsic)
            e_str = str(extrinsic)
            # print(e_str)
            e_str = re.sub(' +', '', e_str[8:-2])
            e_str = e_str.replace(',', ' ')
            e_str = e_str.replace('[', '')
            e_str = e_str.replace(']', '')

            intrinsic = torch.vstack((torch.hstack((torch.eye(2)*cam.f, cam.c)), torch.tensor([0., 0., 1.])))
            # print(intrinsic)
            i_str = str(intrinsic)
            i_str = re.sub(' +', '', i_str[8:-2])
            i_str = i_str.replace(',', ' ')
            i_str = i_str.replace('[', '')
            i_str = i_str.replace(']', '')
            # else:
                # e_str = i_str = '\n\n\n\n\n\n'
            param = f"extrinsic\n{e_str}\n\nintrinsic\n{i_str}\n\n"
            with open(f'{self.data_dir}/cam/{i}_cam.txt', 'w') as f:
                f.write(param)
                f.close()

    def generate_depth_map(self):
        os.system('python pointmvsnet/test.py ' +
                  '--cfg pointmvsnet/configs/dtu_wde3.yaml ' +
                  'TEST.WEIGHT pointmvsnet/outputs/dtu_wde3/pretrained.pth')

    def load_cameras(self):
        for i in range(self.n):
            with open(f'{self.data_dir}/cam/{i}_cam.txt', 'r') as f:
                words = f.read().split()
                extrinsic = torch.tensor(list(map(float, words[1:13]))).reshape(3, 4)

                self.cameras[i].f = torch.tensor([float(words[18])]).T.cuda()
                self.cameras[i].c = torch.tensor([[float(words[20]), float(words[23])]]).T.cuda()
                self.cameras[i].R = extrinsic[:, :3].cuda()
                self.cameras[i].t = extrinsic[:, 3].reshape(3, 1).cuda()

    def generate_point_cloud(self):
        for i in self.available_list:
            depth = torch.load(f'{self.data_dir}/flow/{i}.txt')
            mask = torch.from_numpy((np.dot(plt.imread(f'{self.data_dir}/mask/{i}.png'),
                                            np.array([0.2989, 0.5870, 0.1140])) > 0).astype(int)).cuda()
            pts3 = torch.zeros(3, 640*480, device='cuda:0')
            xx, yy = torch.meshgrid(torch.arange(480), torch.arange(640))

            pts3[0, :] = (yy.reshape(1, -1).cuda() - 320)#  * self.cameras[i].c[0]/320
            pts3[1, :] = (xx.reshape(1, -1).cuda() - 240)#  * self.cameras[i].c[1]/240
            pts3[2, :] = self.cameras[i].f/2.5

            depth = depth.reshape(1, 640*480)[:, mask.flatten().bool()]

            img = torch.from_numpy(cv2.imread(f"{self.data_dir}/object/{i}.png")).cuda()
            img = (img.reshape(640*480, 3).T)[:, mask.flatten().bool()]
            # self.images.append(img.cuda())
            # self.mask.append(mask.cuda())

            # print(depth.shape)
            # print(pts3.shape)
            # print(mask)
            pts3 = pts3[:, mask.flatten().bool()]
            a = depth * pts3/2.5/425
            depth = torch.vstack((pts3[0:2, :], depth, img))
            self.depth.append(depth)
            # print(depth.shape)
            # print(pts3.shape)
            # print(a.shape)
            self.point_clouds.append(a.reshape(3, depth.shape[1]))

    def triangulate(self, i):
        data = Data(pos=self.depth[i][:2, :].T)
        delaunay = Delaunay()
        self.meshes.append(delaunay(data).face)

    def triangulation(self):
        for i in range(len(self.available_list)):
            data = Data(pos=self.depth[i][:2, :].T)
            delaunay = Delaunay()
            self.meshes.append(delaunay(data).face)
            # return

    def mesh_cleaning(self):
        self.triangulation()
        trithresh = 100
        boxlimits = np.array([-10000, 10000, -10000, 10000, -10000, 10000])
        t = time.time()
        for i in range(len(self.available_list)):
            pts2 = self.depth[i]
            pts3 = self.point_clouds[i]
            pts2_kept = pts2[:,(pts3[0] >= boxlimits[0]) & (pts3[0] <= boxlimits[1]) &
                               (pts3[1] >= boxlimits[2]) & (pts3[1] <= boxlimits[3]) &
                               (pts3[2] >= boxlimits[4]) & (pts3[2] <= boxlimits[5])]
            pts3_kept = pts3[:,(pts3[0] >= boxlimits[0]) & (pts3[0] <= boxlimits[1]) &
                               (pts3[1] >= boxlimits[2]) & (pts3[1] <= boxlimits[3]) &
                               (pts3[2] >= boxlimits[4]) & (pts3[2] <= boxlimits[5])]
            self.depth[i] = pts2_kept
            self.point_clouds[i] = pts3_kept
            self.triangulate(i)
            simplices = self.meshes[i].T
            pts3 = self.point_clouds[i]
            # print(simplices.shape, pts3.shape)
            simplices = simplices[(torch.linalg.norm(pts3[:, simplices[:, 0]] - pts3[:, simplices[:, 1]], axis=0) <= trithresh) &
                                  (torch.linalg.norm(pts3[:, simplices[:, 1]] - pts3[:, simplices[:, 2]], axis=0) <= trithresh) &
                                  (torch.linalg.norm(pts3[:, simplices[:, 2]] - pts3[:, simplices[:, 0]], axis=0) <= trithresh)]

            # connected_pts = set(simplices.flatten())
            # print(1)
            a = simplices.flatten()[torch.searchsorted(simplices.flatten(), pts3)]
            # print(2)
            isolated_pts = pts3[a != pts3]
            # print(3)
            connected_pts = pts3[a == pts3]
            # print(isolated_pts, connected_pts)
            # isolated_pts = torch.from_numpy(np.array(list(set(np.arange(pts3.shape[1])) - connected_pts))).cuda()
            index_fixing = torch.zeros((simplices.shape[0], 3), device='cuda:0')

            for idx in isolated_pts:
                index_fixing += (simplices > idx).int()
            self.depth[i] = self.depth[i][:, connected_pts]
            self.depth[i] = self.depth[i][:, connected_pts]
            pts3_new = pts3[:, connected_pts]
            simplices_new = simplices - index_fixing
            self.meshes[i] = simplices_new.T
            self.point_clouds[i] = pts3_new
            print(f'Mesh Cleaning: Iteration {i} Finished, Time: {time.time()-t}')
            t = time.time()

    def saving_cloud(self):
        for i in range(len(self.available_list)):
            pts3 = self.point_clouds[i].detach().cpu().numpy()
            color = self.depth[i][3:].detach().cpu().numpy()/255
            tri = self.meshes[i].detach().cpu().numpy()
            writeply(pts3, color, tri.T, f"./model/dtu_data/clean_cloud/{i}.ply")
            # torch.save(self.depth[i], f"./model/dtu_data/depth/{i}.txt")
        # return

    def load_cloud(self):
        self.depth = []
        for i in range(len(self.available_list)):
            pcd = o3d.io.read_point_cloud(f"./dtu_data/cloud/{i}.ply")
            # pcd.points = o3d.utility.Vector3dVector(self.point_clouds[i].T)
            self.depth.append(torch.load(f"./dtu_data/depth/{i}.txt"))


    # def assemble(self, point_pairs):
    #     for i in range(len(self.available_list)-1):