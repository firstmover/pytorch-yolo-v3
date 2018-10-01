"""
    LYC:
    this file is copied and modified from detect.py.
    very unclean implementation of detect person and crop. 
"""

from __future__ import division

import argparse
import os
import os.path as osp
import time

import cv2
import torch

from darknet import Darknet
from preprocess import prep_image
from util import write_results, load_classes, as_numpy


def arg_parse():
    """
    Parse arguments to the detect module
    """

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument("--images", dest='images', help="Image / Directory containing images to perform detection upon",
                        default="imgs", type=str)
    parser.add_argument("--det", dest='det', help="Image / Directory to store detections to", default="det", type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help="Config file", default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="weightsfile", default="yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso',
                        help="Input resolution of the network, trade off between accuracy and speed.", default="1024",
                        type=str)
    parser.add_argument("--scales", dest="scales", help="Scales to use for detection", default="1,2,3", type=str)

    return parser.parse_args()


def get_model(args):
    # Set up the neural network
    print("Loading network.....")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")
    return model


def main(args, model):

    images = args.images
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)

    CUDA = torch.cuda.is_available()

    num_classes = 80
    classes = load_classes('data/coco.names')

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    # If there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda()

    # Set the model in evaluation mode
    model.eval()

    read_dir = time.time()
    # Detection phase
    try:
        imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images) if
                  os.path.splitext(img)[1] == '.png' or os.path.splitext(img)[1] == '.jpeg' or os.path.splitext(img)[
                      1] == '.jpg']
    except NotADirectoryError:
        imlist = [osp.join(osp.realpath('.'), images)]
    except FileNotFoundError:
        print("No file or directory with the name {}".format(images))
        exit()

    if not os.path.exists(args.det):
        os.makedirs(args.det)

    load_batch = time.time()

    batches = [prep_image(img, inp_dim) for img in imlist]
    im_batches = [x[0] for x in batches]  # each shape (1, 3, H, W) resized H, W
    orig_ims = [x[1] for x in batches]  # each shape (1, 3, H0, W0) not resized
    im_dim_list = torch.FloatTensor([x[2] for x in batches]).repeat(1, 2)  # (nr_img, 4)

    if CUDA:
        im_dim_list = im_dim_list.cuda()

    if batch_size != 1:
        leftover = 1 if len(im_dim_list) % batch_size else 0
        num_batches = len(imlist) // batch_size + leftover
        im_batches = [torch.cat((im_batches[i * batch_size: min((i + 1) * batch_size, len(im_batches))]))
                      for i in range(num_batches)]

    i = 0

    write = False

    start_det_loop = time.time()

    for batch in im_batches:
        # load the image
        if CUDA:
            batch = batch.cuda()

        with torch.no_grad():
            prediction = model(batch, CUDA)

        prediction = write_results(prediction, confidence, num_classes, nms=True, nms_conf=nms_thesh)

        if type(prediction) == int:
            i += 1
            continue

        prediction[:, 0] += i * batch_size

        if not write:
            output = prediction
            write = 1
        else:
            output = torch.cat((output, prediction))

        i += 1

        if CUDA:
            torch.cuda.synchronize()

    try:
        output
    except NameError:
        print("No detections were made")
        exit()

    im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())

    scaling_factor = torch.min(inp_dim / im_dim_list, 1)[0].view(-1, 1)

    output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
    output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

    output[:, 1:5] /= scaling_factor

    for i in range(output.shape[0]):
        output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim_list[i, 0])
        output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim_list[i, 1])

    output_recast = time.time()

    class_load = time.time()

    draw = time.time()

    def _pad_bbox_to_square(c1, c2, pad_ratio=0.1):
        x1, y1 = c1  # left up
        x2, y2 = c2  # right down
        w, h = x2 - x1, y2 - y1
        if w > h:
            a, x, y = w, x1, y1 - (w - h) / 2.0
        else:
            a, x, y = h, x1 - (h - w) / 2.0, y1
        # expand bbox
        x = int(x - a * pad_ratio / 2)
        y = int(y - a * pad_ratio / 2)
        a = int(a + a * pad_ratio)
        return a, x, y

    def _write(a, x, y, img, filename):
        crop = img[y:y + a, x:x + a]
        crop = cv2.resize(crop, (224, 224))
        cv2.imwrite(filename, crop)

    # crop, resize and save person detection
    img_idx2size = {}
    for o in output:
        if int(o[-1]) == 0:  # person: 0
            img_idx = int(o[0])
            a, x, y = _pad_bbox_to_square(as_numpy(o[1:3].int()).tolist(), as_numpy(o[3:5].int()).tolist())
            img = orig_ims[img_idx]
            if 0 < y and y + a < img.shape[0] and 0 < x and x + a < img.shape[1]:
                if img_idx in img_idx2size.keys() and a < img_idx2size[img_idx]:
                    continue
                save_filename = "{}/{}_cropped.png".format(args.det, img_idx)
                _write(a, x, y, img, save_filename)
                img_idx2size[img_idx] = a

    end = time.time()

    print()
    print("SUMMARY")
    print("----------------------------------------------------------")
    print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
    print()
    print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
    print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
    print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) + " images)", output_recast - start_det_loop))
    print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
    print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
    print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch) / len(imlist)))
    print("----------------------------------------------------------")

    torch.cuda.empty_cache()


if __name__ == '__main__':
    args = arg_parse()
    model = get_model(args)

    raw_img_dir = "/afs/csail.mit.edu/u/l/liuyingcheng/lyc_storage/rf-pose-shape/single_person"
    _, exp_dirs, _ = next(os.walk(raw_img_dir))
    for exp in exp_dirs:
        print("exp:", exp)
        args.images = os.path.join(raw_img_dir, exp)
        args.det = os.path.join(raw_img_dir, exp, "crop_person")
        if os.path.exists(args.det):
            raise ValueError("crop image exists")
        os.mkdir(args.det)

        main(args, model)
