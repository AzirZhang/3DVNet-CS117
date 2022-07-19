import torch
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from model import camutils, visutils
import cv2

# [0,1,2]
# d = []
# n = 4
# start = 3
# for i in range(start, start+n):
#     d.append(torch.load(f"model/dtu_data/flow/{i}.txt").cpu())
#
# d = torch.cat(d).cpu()
# print(d.shape)
#
# for i in range(0, n):
#     image_idx = i
#
#     plt.imshow(d[image_idx][0], cmap='jet')
#
#     plt.colorbar()
#     plt.title('code')
#     plt.savefig(f'project_img/{i}.png')
#     plt.show()


# t = t[0]
# c = camutils.Camera(
#     # f=(2892.33+2883.18)/2,
#     # ,
#     f=(361.54125+360.3975)/2,
#     c=np.array([82.900625, 66.383875]).reshape(2, 1),
#     R=np.array([[0.970263, 0.00747983, 0.241939],
#                 [-0.0147429, 0.999493, 0.0282234],
#                 [-0.241605, -0.030951, 0.969881]]),
#     t=np.array([-191.02, 3.28832, 22.5401]).reshape(3, 1)
# )

# point_cloud = torch.load("point_cloud.txt")
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(point_cloud.T.detach().cpu().numpy())
# o3d.io.write_point_cloud(f"./model/dtu_data/cloud/0.ply", pcd)

# mesh = torch.load("mesh.txt")

# depth = torch.load("./model/dtu_data/depth/0.txt")
# img = depth[3:].detach().cpu().numpy()
# plt.imshow(img)
# print(depth)

flow = torch.load("./model/dtu_data/depth/scan1/0.txt")
img = flow.reshape(480, 640).detach().cpu().numpy()
plt.imshow(img, cmap='jet')
plt.savefig(f'project_img/4.png')
plt.show()

# fig1 = plt.figure()
# ax = fig1.add_subplot(111, projection='3d')
# # ax.plot(point_cloud[0].detach().cpu(), point_cloud[1].detach().cpu(), point_cloud[2].detach().cpu(), 'r.')
# ax.plot_trisurf(point_cloud[0, :].detach().cpu(),
#                 point_cloud[1, :].detach().cpu(),
#                 point_cloud[2, :].detach().cpu(),
#                 triangles=mesh.detach().cpu(),
#                 antialiased=False)
# ax.view_init(azim=-120, elev=-135)  #set the camera viewpointn
# visutils.set_axes_equal_3d(ax)
# visutils.label_axes(ax)
# plt.title('Final Mesh View 1')
# plt.savefig(f'project_img/mesh.png')
# # plt.gca().set_xlim(plt.gca().get_xlim()[::-1])
# # plt.gca().set_ylim(plt.gca().get_ylim()[::-1])
# plt.show()

# plt.imshow(d[image_idx][0], cmap='jet')
# plt.colorbar()
# plt.title('code')
# plt.show()
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.plot(t[0, 0, :], t[0, 1, :], t[0, 2, :], 'b.')
# visutils.set_axes_equal_3d(ax)
# visutils.label_axes(ax)
# plt.title('reconstruction overlay')
# plt.show()