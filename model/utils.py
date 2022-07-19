import torch
from matplotlib import pyplot as plt
import model.visutils as visutils
import cv2


def check_2d_position(model, chessboard):
    cameras = model.cameras
    n = len(cameras)
    colors = (['r', 'g', 'b']*(n//3+1))[:n]
    plt.rcParams['figure.figsize'] = [15, 15]
    for i in range(len(cameras)):
        cam = cameras[i]

        projected_board = cam.project(chessboard)
        # print(projected_board)
        img = cv2.imread(f"{model.data_dir}/chessboard/{i}.png")
        plt.imshow(img)
        plt.plot(projected_board[0, :], projected_board[1, :], colors[i])
        direction = torch.tensor([[0, 5.6], [0., 8.4], [0., 0.]])
        direction = cam.project(direction)
        # print(direction)

        plt.plot(direction[0], direction[1], colors[-1-i])
        plt.show()


def check_3d_position(model, chessboard):
    cameras = model.cameras
    n = len(cameras)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    colors = (['r', 'g', 'b']*(n//3+1))[:n]
    plt.rcParams['figure.figsize'] = [15, 15]
    ax.plot(chessboard[0, :], chessboard[1, :], chessboard[2, :])
    ax.plot(chessboard[0, [0, 16]], chessboard[1, [0, 16]], chessboard[2, [0, 16]])
    for i in range(n):
        cam = cameras[i]
        direction = torch.tensor([[0.], [0.], [5.]])
        direction = cam.R@direction + cam.t
        direction = torch.hstack((cam.t, direction))
        print(cam.t)
        print(direction)

        # projected_board = cam.project(chessboard)
        # img = cv2.imread(f"{model.data_dir}/chessboard/{i}.png")
        # plt.imshow(img)
        # plt.plot(projected_board[0, :], projected_board[1, :], colors[i])
        # plt.show()

        # ax.plot(projected_board[0, :], projected_board[1, :], projected_board[2, :], f'{colors[i]}x')
        ax.plot(cam.t[0], cam.t[1], cam.t[2], f'{colors[i]}o')
        ax.plot(direction[0, :], direction[1, :], direction[2, :], f'{colors[-i-1]}')
        visutils.set_axes_equal_3d(ax)
        visutils.label_axes(ax)
    plt.title('scene 3D view')
    plt.gca().set_ylim(plt.gca().get_ylim()[::-1])
    plt.show()
    return
