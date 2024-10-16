import gradio as gr
import argparse
import ast

def get_args_from_dict(args_dict):
    '''
    Get args from a dictionary.
    '''
    parser = argparse.ArgumentParser(description='Pix2Pix Model')

    # parameters
    parser.add_argument('--device_target', type=str, default='Ascend', choices=('Ascend', 'GPU'),
                        help='device where the code will be implemented (default: Ascend)')
    parser.add_argument('--run_distribute', type=int, default=0, help='distributed training, default is 0.')
    parser.add_argument('--device_num', type=int, default=1, help='device num, default is 1.')
    parser.add_argument('--device_id', type=int, default=6, help='device id, default is 0.')
    parser.add_argument('--save_graphs', type=ast.literal_eval, default=False,
                        help='whether save graphs, default is False.')
    parser.add_argument('--init_type', type=str, default='normal', help='network initialization, default is normal.')
    parser.add_argument('--init_gain', type=float, default=0.02,
                        help='scaling factor for normal, xavier and orthogonal, default is 0.02.')
    parser.add_argument('--pad_mode', type=str, default='CONSTANT', choices=('CONSTANT', 'REFLECT', 'SYMMETRIC'),
                        help='scale images to this size, default is CONSTANT.')
    parser.add_argument('--load_size', type=int, default=286, help='scale images to this size, default is 286.')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size, default is 1.')
    parser.add_argument('--LAMBDA_Dis', type=float, default=0.5, help='weight for Discriminator Loss, default is 0.5.')
    parser.add_argument('--LAMBDA_GAN', type=int, default=1, help='weight for GAN Loss, default is 1.')
    parser.add_argument('--LAMBDA_L1', type=int, default=100, help='weight for L1 Loss, default is 100.')
    parser.add_argument('--beta1', type=float, default=0.5, help='adam beta1, default is 0.5.')
    parser.add_argument('--beta2', type=float, default=0.999, help='adam beta2, default is 0.999.')
    parser.add_argument('--lr', type=float, default=0.0002, help='the initial learning rate, default is 0.0002.')
    parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy, default is linear.')
    parser.add_argument('--epoch_num', type=int, default=200, help='epoch number for training, default is 200.')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='number of epochs with the initial learning rate, default is 100.')
    parser.add_argument('--n_epochs_decay', type=int, default=100,
                        help='number of epochs with the dynamic learning rate, default is 100.')
    parser.add_argument('--dataset_size', type=int, default=400, choices=(400, 1096),
                        help='for Facade_dataset,the number is 400; for Maps_dataset,the number is 1096.')

    # The location of input and output data
    parser.add_argument('--train_data_dir', type=str, default=None, help='the file path of input data during training.')
    parser.add_argument('--val_data_dir', type=str, default=None, help='the file path of input data during validating.')
    parser.add_argument('--train_fakeimg_dir', type=str, default='./results/fake_img/',
                        help='during training, the file path of stored fake img.')
    parser.add_argument('--loss_show_dir', type=str, default='./results/loss_show',
                        help='during training, the file path of stored loss img.')
    parser.add_argument('--ckpt_dir', type=str, default='./results/ckpt/',
                        help='during training, the file path of stored CKPT.')
    parser.add_argument('--ckpt', type=str, default=None, help='during validating, the file path of the CKPT used.')
    parser.add_argument('--predict_dir', type=str, default='./results/predict/',
                        help='during validating, the file path of Generated image.')

    args = parser.parse_args(namespace=argparse.Namespace(**args_dict))
    return args

def process_args(device_target, run_distribute, device_num, device_id, save_graphs, init_type, init_gain,
                 pad_mode, load_size, batch_size, LAMBDA_Dis, LAMBDA_GAN, LAMBDA_L1, beta1, beta2, lr, lr_policy,
                 epoch_num, n_epochs, n_epochs_decay, dataset_size, train_data_dir, val_data_dir, train_fakeimg_dir,
                 loss_show_dir, ckpt_dir, ckpt, predict_dir):

    args_dict = {
        'device_target': device_target,
        'run_distribute': run_distribute,
        'device_num': device_num,
        'device_id': device_id,
        'save_graphs': save_graphs,
        'init_type': init_type,
        'init_gain': init_gain,
        'pad_mode': pad_mode,
        'load_size': load_size,
        'batch_size': batch_size,
        'LAMBDA_Dis': LAMBDA_Dis,
        'LAMBDA_GAN': LAMBDA_GAN,
        'LAMBDA_L1': LAMBDA_L1,
        'beta1': beta1,
        'beta2': beta2,
        'lr': lr,
        'lr_policy': lr_policy,
        'epoch_num': epoch_num,
        'n_epochs': n_epochs,
        'n_epochs_decay': n_epochs_decay,
        'dataset_size': dataset_size,
        'train_data_dir': train_data_dir,
        'val_data_dir': val_data_dir,
        'train_fakeimg_dir': train_fakeimg_dir,
        'loss_show_dir': loss_show_dir,
        'ckpt_dir': ckpt_dir,
        'ckpt': ckpt,
        'predict_dir': predict_dir
    }

    args = get_args_from_dict(args_dict)
    return f"Arguments: {args}"

def greet(name):
    return f"Hello, {name}!"

def goodbye(name):
    return f"Goodbye, {name}!"
train = gr.Interface(
    fn=process_args,
    inputs=[
        gr.Dropdown(['Ascend', 'GPU'], label='Device Target'),
        gr.Number(value=0, label='Run Distribute'),
        gr.Number(value=1, label='Device Num'),
        gr.Number(value=6, label='Device ID'),
        gr.Checkbox(label='Save Graphs'),
        gr.Textbox(value='normal', label='Init Type'),
        gr.Number(value=0.02, label='Init Gain'),
        gr.Dropdown(['CONSTANT', 'REFLECT', 'SYMMETRIC'], label='Pad Mode'),
        gr.Number(value=286, label='Load Size'),
        gr.Number(value=1, label='Batch Size'),
        gr.Number(value=0.5, label='Lambda Dis'),
        gr.Number(value=1, label='Lambda GAN'),
        gr.Number(value=100, label='Lambda L1'),
        gr.Number(value=0.5, label='Beta1'),
        gr.Number(value=0.999, label='Beta2'),
        gr.Number(value=0.0002, label='Learning Rate'),
        gr.Textbox(value='linear', label='LR Policy'),
        gr.Number(value=200, label='Epoch Number'),
        gr.Number(value=100, label='N Epochs'),
        gr.Number(value=100, label='N Epochs Decay'),
        gr.Number(value=400, label='Dataset Size'),
        gr.Textbox(value=None, label='Train Data Dir'),
        gr.Textbox(value=None, label='Val Data Dir'),
        gr.Textbox(value='./results/fake_img/', label='Train Fake Image Dir'),
        gr.Textbox(value='./results/loss_show', label='Loss Show Dir'),
        gr.Textbox(value='./results/ckpt/', label='Checkpoint Dir'),
        gr.Textbox(value=None, label='Checkpoint'),
        gr.Textbox(value='./results/predict/', label='Predict Dir')
    ],
    outputs="text"
)

predict = gr.Interface(
    fn=process_args,
    inputs=[
        gr.Dropdown(['Ascend', 'GPU'], label='Device Target'),
        gr.Number(value=0, label='Run Distribute'),
        gr.Number(value=1, label='Device Num'),
        gr.Number(value=6, label='Device ID'),
        gr.Checkbox(label='Save Graphs'),
        gr.Textbox(value='normal', label='Init Type'),
        gr.Number(value=0.02, label='Init Gain'),
        gr.Dropdown(['CONSTANT', 'REFLECT', 'SYMMETRIC'], label='Pad Mode'),
        gr.Number(value=286, label='Load Size'),
        gr.Number(value=1, label='Batch Size'),
        gr.Number(value=0.5, label='Lambda Dis'),
        gr.Number(value=1, label='Lambda GAN'),
        gr.Number(value=100, label='Lambda L1'),
        gr.Number(value=0.5, label='Beta1'),
        gr.Number(value=0.999, label='Beta2'),
        gr.Number(value=0.0002, label='Learning Rate'),
        gr.Textbox(value='linear', label='LR Policy'),
        gr.Number(value=200, label='Epoch Number'),
        gr.Number(value=100, label='N Epochs'),
        gr.Number(value=100, label='N Epochs Decay'),
        gr.Number(value=400, label='Dataset Size'),
        gr.Textbox(value=None, label='Train Data Dir'),
        gr.Textbox(value=None, label='Val Data Dir'),
        gr.Textbox(value='./results/fake_img/', label='Train Fake Image Dir'),
        gr.Textbox(value='./results/loss_show', label='Loss Show Dir'),
        gr.Textbox(value='./results/ckpt/', label='Checkpoint Dir'),
        gr.Textbox(value=None, label='Checkpoint'),
        gr.Textbox(value='./results/predict/', label='Predict Dir')
    ],
    outputs="text"
)

# 创建界面
with gr.Blocks() as iface:
    # 添加输入文本框
    text_input = gr.Textbox(label="Name")
    #
    # 添加输出文本框
    text_output = gr.Textbox(label="Output")

    # 添加两个按钮
    greet_button = gr.Button("Train")
    goodbye_button = gr.Button("Predict")

    # 绑定按钮点击事件
    greet_button.click(
        fn=process_args,
        inputs=[
            gr.Dropdown(['Ascend', 'GPU'], label='Device Target', visible=False),
            gr.Number(value=0, label='Run Distribute', visible=False),
            gr.Number(value=1, label='Device Num', visible=False),
            gr.Number(value=6, label='Device ID', visible=False),
            gr.Checkbox(label='Save Graphs', visible=False),
            gr.Textbox(value='normal', label='Init Type', visible=False),
            gr.Number(value=0.02, label='Init Gain', visible=False),
            gr.Dropdown(['CONSTANT', 'REFLECT', 'SYMMETRIC'], label='Pad Mode', visible=False),
            gr.Number(value=286, label='Load Size', visible=False),
            gr.Number(value=1, label='Batch Size', visible=False),
            gr.Number(value=0.5, label='Lambda Dis', visible=False),
            gr.Number(value=1, label='Lambda GAN', visible=False),
            gr.Number(value=100, label='Lambda L1', visible=False),
            gr.Number(value=0.5, label='Beta1', visible=False),
            gr.Number(value=0.999, label='Beta2', visible=False),
            gr.Number(value=0.0002, label='Learning Rate', visible=False),
            gr.Textbox(value='linear', label='LR Policy', visible=False),
            gr.Number(value=200, label='Epoch Number', visible=False),
            gr.Number(value=100, label='N Epochs', visible=False),
            gr.Number(value=100, label='N Epochs Decay', visible=False),
            gr.Number(value=400, label='Dataset Size', visible=False),
            gr.Textbox(value=None, label='Train Data Dir', visible=False),
            gr.Textbox(value=None, label='Val Data Dir', visible=False),
            gr.Textbox(value='./results/fake_img/', label='Train Fake Image Dir', visible=False),
            gr.Textbox(value='./results/loss_show', label='Loss Show Dir', visible=False),
            gr.Textbox(value='./results/ckpt/', label='Checkpoint Dir', visible=False),
            gr.Textbox(value=None, label='Checkpoint', visible=False),
            gr.Textbox(value='./results/predict/', label='Predict Dir', visible=False)
        ],
        outputs=text_output
        )
    goodbye_button.click(fn=goodbye, inputs=text_input, outputs=text_output)

# 启动 Gradio 应用
iface.launch()
