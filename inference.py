"""
by zk
镭射标签分类模型调用接口
    输入：
        模型文件： pt（torchscript）/onnx格式
        图像文件路径： str
    输出：
        类别： real:真标签
              print: 彩印标签
              laminat: 覆膜标签
              shading: 底纹标签
最后修改：20220307
"""

import logging
import os
import time
from os.path import exists

import cv2
import numpy as np
import onnxruntime
import torch


def pad_with_params(
    img: np.ndarray,
    h_pad_top: int,
    h_pad_bottom: int,
    w_pad_left: int,
    w_pad_right: int,
    border_mode: int = cv2.BORDER_CONSTANT,
    value: [int] = (0, 0, 0),
) -> np.ndarray:
    pad_fn = cv2.copyMakeBorder(img,
                                top=h_pad_top,
                                bottom=h_pad_bottom,
                                left=w_pad_left,
                                right=w_pad_right,
                                borderType=border_mode,
                                value=value)

    return pad_fn.astype(np.float32)


def preprocess(im_array, resize_or_padding=0, size=(240, 180),  mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True):
    """preprocess input image.

    Args:
        resize_or_padding:
        size:  (w, h), dst size of image
        im_array (ndarray): Image to be normalized.
        mean (ndarray): The mean to be used for normalize.
        std (ndarray): The std to be used for normalize.
        to_rgb (bool): Whether to convert to rgb.

    Returns:
        ndarray: The normalized image.
    """

    # check dims
    assert im_array.dtype == np.uint8
    if resize_or_padding == 0:  # resize
        img = cv2.resize(im_array.astype(np.float32), size,
                         interpolation=cv2.INTER_LINEAR)  # dsize: w, h = (240, 180)
    else:  # padding with 0
        rows, cols = im_array.shape[:2]
        h_pad_top = int((size[1] - rows) / 2.0)
        h_pad_bottom = size[1] - rows - h_pad_top
        w_pad_left = int((size[0] - cols) / 2.0)
        w_pad_right = size[0] - cols - w_pad_left
        img = pad_with_params(im_array, h_pad_top, h_pad_bottom, w_pad_left, w_pad_right)

    if len(img.shape) < 3:
        img = np.expand_dims(img, -1)

    # imnormalize:  to float32(very important!), normalize, bgr2rgb
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    assert img.dtype != np.uint8
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace

    return img


def inference_LaserLabel(im_array, model_path):
    # config
    class_names = ["laminate", "print", "real", "shading"]

    img = preprocess(im_array)
    img_tensor = np.array(img).transpose(2, 0, 1).reshape(1, 3, 180, 240)

    # inference
    # model
    if model_path.endswith('pt'):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        img_tensor = torch.from_numpy(img_tensor).to(device)
        model = torch.jit.load(model_path)
        model.to(device)
        model.eval()
        start = time.time()
        output = model(img_tensor).cpu()  # 全连接层输出
        print("Inference time = {} ms".format((time.time() - start) * 1000))
        output = np.vstack(output.detach().numpy())

    elif model_path.endswith('onnx'):
        session = onnxruntime.InferenceSession(model_path, None, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        # output = session.run(None, img_tensor)
        # todo: debug input size problem
        start = time.time()
        output = session.run(output_names=[output_name], input_feed={input_name: img_tensor})[0]  # required imsize=224
        print("Inference time = {} ms".format((time.time() - start) * 1000))

    else:
        print('wrong format of model!')
        return 0

    # get result
    pred_score = np.max(output, axis=1)
    pred_label = np.argmax(output, axis=1)
    pred_class = class_names[pred_label[0]]

    return pred_class, pred_score


def inference_frameH500(im_array, model_path):
    # config
    class_names = ["real", "shading"]

    input_size = (256, 192)
    img = preprocess(im_array, resize_or_padding=1, size=input_size, to_rgb=True)
    img_tensor = np.array(img).transpose(2, 0, 1).reshape(1, 3, input_size[1], input_size[0])

    # inference
    # model
    if model_path.endswith('pt'):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        img_tensor = torch.from_numpy(img_tensor).to(device)
        model = torch.jit.load(model_path)
        model.to(device)
        model.eval()
        start = time.time()
        output = model(img_tensor).cpu()  # 全连接层输出
        print("Inference time = {} ms".format((time.time() - start) * 1000))
        output = np.vstack(output.detach().numpy())

    elif model_path.endswith('onnx'):
        session = onnxruntime.InferenceSession(model_path, None, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        start = time.time()
        output = session.run(output_names=[output_name], input_feed={input_name: img_tensor})[0]
        print("Inference time = {} ms".format((time.time() - start) * 1000))

    else:
        print('wrong format of model!')
        return 0

    # get result
    pred_score = np.max(output, axis=1)
    pred_label = np.argmax(output, axis=1)
    pred_class = class_names[pred_label[0]]

    return pred_class, pred_score


if __name__ == "__main__":

    # path
    model_path = r'C:\Users\zk\Desktop\mmclassification-0.18.0\experiment\mobilenet_v2\ColorLine300_epoch300.onnx'

    # # 1. test single image
    # test_path = r'C:\Users\zk\Desktop\NewAlgo_LineLaser\videos\LineLaser\Dataset_frameH500\shading'
    # # img_path = '1.png'
    # img_path = os.path.join(test_path, '0003506032_1_284.png')
    # img_array = cv2.imread(img_path, 1)
    # # predict_cls, max_prob = inference_LaserLabel(img_array, model_path)
    # predict_cls, max_prob = inference_frameH500(img_array, model_path)
    #
    # print('predicted class: {}'.format(predict_cls))
    # print('prob: {}'.format(max_prob))

    # 2. test multiple images
    # path = r'C:\Users\zk\Desktop\NewAlgo_LineLaser\videos\LineLaser\ColorLine500'
    # path = r'C:\Users\zk\Desktop\NewAlgo_LineLaser\videos\validate_DL\ColorLine500'
    path = r'C:\Users\zk\Desktop\NewAlgo_LineLaser\videos\LineLaser202203\ColorLine300'
    # path = '../ColorLine500_linelaser'

    class_list = ['real', 'laminate', 'shading', 'real2']
    # class_list = os.listdir(path)

    save_fail = False

    logfile = os.path.join(path, 'DL_log.txt')
    logging.basicConfig(filename=logfile, filemode='w', level=logging.INFO)
    # gt_label = 'shading'

    for cls in class_list:

        test_path = os.path.join(path, cls)
        if not exists(test_path):
            continue
        if save_fail:
            save_path = os.path.join(os.path.dirname(test_path), cls + '_fails')
            if not os.path.exists(save_path):
                os.makedirs(save_path)

        # if os.path.isdir(test_path) and dir.startswith('4'):
        # if os.path.isdir(test_path) and dir == 'Dataset_LaserLabel':
        if os.path.isdir(test_path):

            results_list = list()

            for file in os.listdir(test_path):
                # input
                if not file.endswith(('png', 'jpg', 'jpeg', 'bmp')):
                    continue
                img_path = os.path.join(test_path, file)
                img_array = cv2.imread(img_path, 1)

                # inference
                # predict_cls, max_prob = inference_LaserLabel(img_array, model_path)
                predict_cls, max_prob = inference_frameH500(img_array, model_path)

                # logging.info('file:{} predicted class: {}'.format(file, predict_cls))
                results_list.append(predict_cls)

                # save fail
                if save_fail and predict_cls != cls:
                    save_name = os.path.join(save_path, file)
                    cv2.putText(img_array, 'pred:{}'.format(predict_cls), (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imwrite(save_name, img_array)

                    cv2.namedWindow('fail')
                    cv2.imshow('fail', img_array)
                    cv2.waitKey(5)

            # compute metrics
            if len(results_list) > 0:
                num_real = results_list.count('real')
                # num_laminate = results_list.count('laminate')
                # num_print = results_list.count('print')
                num_shading = results_list.count('shading')

                num_correct = results_list.count(cls)
                ratio_correct = float(num_correct / len(results_list))
                print('ratio of num_correct: {:.3f}({}/{})'
                      .format(ratio_correct, num_correct, len(results_list))
                      )

                # log
                logging.info('cls: {}'.format(cls))
                logging.info('ratio of num_real: {:.3f}({}/{})'
                             .format(num_real / len(results_list), num_real, len(results_list))
                             )
                # logging.info('ratio of num_laminate: {:.3f}({}/{})'
                #              .format(num_laminate / len(results_list), num_laminate, len(results_list))
                #              )
                # logging.info('ratio of num_print: {:.3f}({}/{})'
                #              .format(num_print / len(results_list), num_print, len(results_list))
                #              )
                logging.info('ratio of num_shading: {:.3f}({}/{})'
                             .format(num_shading / len(results_list), num_shading, len(results_list))
                             )
