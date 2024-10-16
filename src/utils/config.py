# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===========================================================================

"""
    Define the common options that are used in both training and test.
"""

import argparse
import ast


def get_args():
    '''
        get args.
    '''
    parser = argparse.ArgumentParser(description='Pix2Pix Model')

    # 添加命令行参数以指定使用的设备类型
    parser.add_argument('--device_target', type=str, default='Ascend', choices=('Ascend', 'GPU'),
                        help='运行代码的设备 (默认: Ascend)')

    # 添加命令行参数以指定是否使用分布式训练
    parser.add_argument('--run_distribute', type=int, default=0, help='分布式训练，默认为 0')

    # 添加命令行参数以指定设备数量
    parser.add_argument('--device_num', type=int, default=1, help='设备数量，默认为 1')

    # 添加命令行参数以指定设备 ID
    parser.add_argument('--device_id', type=int, default=6, help='设备 ID，默认为 0')

    # 添加命令行参数以指定是否保存图
    parser.add_argument('--save_graphs', type=ast.literal_eval, default=False,
                        help='是否保存图，默认为 False')

    # 添加命令行参数以指定网络初始化类型
    parser.add_argument('--init_type', type=str, default='normal', help='网络初始化类型，默认为 normal')

    # 添加命令行参数以指定初始化缩放因子
    parser.add_argument('--init_gain', type=float, default=0.02,
                        help='normal, xavier 和 orthogonal 初始化的缩放因子，默认为 0.02')

    # 添加命令行参数以指定填充模式
    parser.add_argument('--pad_mode', type=str, default='CONSTANT', choices=('CONSTANT', 'REFLECT', 'SYMMETRIC'),
                        help='图像缩放模式，默认为 CONSTANT')

    # 添加命令行参数以指定图像加载大小
    parser.add_argument('--load_size', type=int, default=286, help='图像缩放大小，默认为 286')

    # 添加命令行参数以指定批量大小
    parser.add_argument('--batch_size', type=int, default=1, help='批量大小，默认为 1')

    # 添加命令行参数以指定判别器损失权重
    parser.add_argument('--LAMBDA_Dis', type=float, default=0.5, help='判别器损失权重，默认为 0.5')

    # 添加命令行参数以指定 GAN 损失权重
    parser.add_argument('--LAMBDA_GAN', type=int, default=1, help='GAN 损失权重，默认为 1')

    # 添加命令行参数以指定 L1 损失权重
    parser.add_argument('--LAMBDA_L1', type=int, default=100, help='L1 损失权重，默认为 100')

    # 添加命令行参数以指定 Adam 优化器的 beta1 参数
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam 优化器的 beta1 参数，默认为 0.5')

    # 添加命令行参数以指定 Adam 优化器的 beta2 参数
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam 优化器的 beta2 参数，默认为 0.999')

    # 添加命令行参数以指定初始学习率
    parser.add_argument('--lr', type=float, default=0.0002, help='初始学习率，默认为 0.0002')

    # 添加命令行参数以指定学习率策略
    parser.add_argument('--lr_policy', type=str, default='linear', help='学习率策略，默认为 linear')

    # 添加命令行参数以指定训练的总轮数
    parser.add_argument('--epoch_num', type=int, default=200, help='训练轮数，默认为 200')

    # 添加命令行参数以指定使用初始学习率的轮数
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='使用初始学习率的轮数，默认为 100')

    # 添加命令行参数以指定使用动态学习率的轮数
    parser.add_argument('--n_epochs_decay', type=int, default=100,
                        help='使用动态学习率的轮数，默认为 100')

    # 添加命令行参数以指定数据集大小
    parser.add_argument('--dataset_size', type=int, default=400, choices=(400, 1096),
                        help='Facade 数据集大小为 400；Maps 数据集大小为 1096')

    # 输入和输出数据的位置
    # 添加命令行参数以指定训练数据文件路径
    parser.add_argument('--train_data_dir', type=str, default=None, help='训练期间的输入数据文件路径')

    # 添加命令行参数以指定验证数据文件路径
    parser.add_argument('--val_data_dir', type=str, default=None, help='验证期间的输入数据文件路径')

    # 添加命令行参数以指定训练期间存储假图像的文件路径
    parser.add_argument('--train_fakeimg_dir', type=str, default='./results/fake_img/',
                        help='训练期间存储假图像的文件路径')

    # 添加命令行参数以指定训练期间存储损失图像的文件路径
    parser.add_argument('--loss_show_dir', type=str, default='./results/loss_show',
                        help='训练期间存储损失图像的文件路径')

    # 添加命令行参数以指定训练期间存储检查点的文件路径
    parser.add_argument('--ckpt_dir', type=str, default='./results/ckpt/',
                        help='训练期间存储检查点的文件路径')

    # 添加命令行参数以指定验证期间使用的检查点文件路径
    parser.add_argument('--ckpt', type=str, default='./results/ckpt/Generator_150', help='验证期间使用的检查点文件路径')

    # 添加命令行参数以指定验证期间生成图像的文件路径
    parser.add_argument('--predict_dir', type=str, default='./results/predict/',
                        help='验证期间生成图像的文件路径')

    args = parser.parse_args()
    return args


def get_args_eval(file_path):
    '''
        get args.
    '''
    parser = argparse.ArgumentParser(description='Pix2Pix Model')

    # 添加命令行参数以指定使用的设备类型
    parser.add_argument('--device_target', type=str, default='Ascend', choices=('Ascend', 'GPU'),
                        help='运行代码的设备 (默认: Ascend)')

    # 添加命令行参数以指定是否使用分布式训练
    parser.add_argument('--run_distribute', type=int, default=0, help='分布式训练，默认为 0')

    # 添加命令行参数以指定设备数量
    parser.add_argument('--device_num', type=int, default=1, help='设备数量，默认为 1')

    # 添加命令行参数以指定设备 ID
    parser.add_argument('--device_id', type=int, default=6, help='设备 ID，默认为 0')

    # 添加命令行参数以指定是否保存图
    parser.add_argument('--save_graphs', type=ast.literal_eval, default=False,
                        help='是否保存图，默认为 False')

    # 添加命令行参数以指定网络初始化类型
    parser.add_argument('--init_type', type=str, default='normal', help='网络初始化类型，默认为 normal')

    # 添加命令行参数以指定初始化缩放因子
    parser.add_argument('--init_gain', type=float, default=0.02,
                        help='normal, xavier 和 orthogonal 初始化的缩放因子，默认为 0.02')

    # 添加命令行参数以指定填充模式
    parser.add_argument('--pad_mode', type=str, default='CONSTANT', choices=('CONSTANT', 'REFLECT', 'SYMMETRIC'),
                        help='图像缩放模式，默认为 CONSTANT')

    # 添加命令行参数以指定图像加载大小
    parser.add_argument('--load_size', type=int, default=286, help='图像缩放大小，默认为 286')

    # 添加命令行参数以指定批量大小
    parser.add_argument('--batch_size', type=int, default=1, help='批量大小，默认为 1')

    # 添加命令行参数以指定判别器损失权重
    parser.add_argument('--LAMBDA_Dis', type=float, default=0.5, help='判别器损失权重，默认为 0.5')

    # 添加命令行参数以指定 GAN 损失权重
    parser.add_argument('--LAMBDA_GAN', type=int, default=1, help='GAN 损失权重，默认为 1')

    # 添加命令行参数以指定 L1 损失权重
    parser.add_argument('--LAMBDA_L1', type=int, default=100, help='L1 损失权重，默认为 100')

    # 添加命令行参数以指定 Adam 优化器的 beta1 参数
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam 优化器的 beta1 参数，默认为 0.5')

    # 添加命令行参数以指定 Adam 优化器的 beta2 参数
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam 优化器的 beta2 参数，默认为 0.999')

    # 添加命令行参数以指定初始学习率
    parser.add_argument('--lr', type=float, default=0.0002, help='初始学习率，默认为 0.0002')

    # 添加命令行参数以指定学习率策略
    parser.add_argument('--lr_policy', type=str, default='linear', help='学习率策略，默认为 linear')

    # 添加命令行参数以指定训练的总轮数
    parser.add_argument('--epoch_num', type=int, default=200, help='训练轮数，默认为 200')

    # 添加命令行参数以指定使用初始学习率的轮数
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='使用初始学习率的轮数，默认为 100')

    # 添加命令行参数以指定使用动态学习率的轮数
    parser.add_argument('--n_epochs_decay', type=int, default=100,
                        help='使用动态学习率的轮数，默认为 100')

    # 添加命令行参数以指定数据集大小
    parser.add_argument('--dataset_size', type=int, default=400, choices=(400, 1096),
                        help='Facade 数据集大小为 400；Maps 数据集大小为 1096')

    # 输入和输出数据的位置
    # 添加命令行参数以指定训练数据文件路径
    parser.add_argument('--train_data_dir', type=str, default=None, help='训练期间的输入数据文件路径')

    # 添加命令行参数以指定验证数据文件路径
    parser.add_argument('--val_data_dir', type=str, default=file_path, help='验证期间的输入数据文件路径')

    # 添加命令行参数以指定训练期间存储假图像的文件路径
    parser.add_argument('--train_fakeimg_dir', type=str, default='./results/fake_img/',
                        help='训练期间存储假图像的文件路径')

    # 添加命令行参数以指定训练期间存储损失图像的文件路径
    parser.add_argument('--loss_show_dir', type=str, default='./results/loss_show',
                        help='训练期间存储损失图像的文件路径')

    # 添加命令行参数以指定训练期间存储检查点的文件路径
    parser.add_argument('--ckpt_dir', type=str, default='./results/ckpt/',
                        help='训练期间存储检查点的文件路径')

    # 添加命令行参数以指定验证期间使用的检查点文件路径
    parser.add_argument('--ckpt', type=str, default='./results/ckpt/Generator_150.ckpt', help='验证期间使用的检查点文件路径')

    # 添加命令行参数以指定验证期间生成图像的文件路径
    parser.add_argument('--predict_dir', type=str, default='./results/predict/',
                        help='验证期间生成图像的文件路径')

    args = parser.parse_args()
    return args