import torch

params_init = torch.tensor([[-120., 0., 0., 30., 40., 60.],
                            [-160., 0., 0., 5., 30., 50.],
                            [-120., 0., 0., 5., 20., 40.],
                            [-160., 0., 0., -10., 40., 80.],
                            [-135., -40., 0., -30., 40., 30.]])

torch.save(params_init, './model/params_init.txt')

chess_rotate_init = torch.tensor([[0],
                                  [0],
                                  [0],
                                  [1],
                                  [1]])

torch.save(chess_rotate_init, './model/chess_rotate_init.txt')
