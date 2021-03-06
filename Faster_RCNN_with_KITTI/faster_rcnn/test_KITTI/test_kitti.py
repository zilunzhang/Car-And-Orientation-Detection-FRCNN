import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2
import argparse
import os.path as osp
import glob
from tensorflow.core.protobuf import saver_pb2

this_dir = osp.dirname(__file__)
print(this_dir)

from faster_rcnn.lib.networks.factory import get_network
from faster_rcnn.lib.fast_rcnn.config import cfg
from faster_rcnn.lib.fast_rcnn.test import im_detect
from faster_rcnn.lib.fast_rcnn.nms_wrapper import nms
from faster_rcnn.lib.utils.timer import Timer
import re
CLASSES = ('background', 'Pedestrian', 'Car', 'Cyclist')


# CLASSES = ('__background__','person','bike','motorbike','car','bus')

def vis_detections(im, img_name, class_name, dets, ax, thresh=0.5):
    """Draw detected bounding boxes."""
    image_name = re.findall(r'\d+', img_name)[0]

    inds = np.where(dets[:, -1] >= thresh)[0]

    if class_name == "Car":
        color = "red"
    elif class_name == "Pedestrian":
        color = "blue"
    elif class_name == "Cyclist":
        color = "yellow"
    else:
        color = "red"


    if len(inds) == 0:
        print "no {}".format(class_name)
        return

    str = "{}_{}.csv".format(image_name, class_name)
    write_down = np.zeros((len(inds), 4))
    q = 0

    for index in inds:
        write_down[q, :] = dets[index, :4]
        q+=1
    np.savetxt(str, write_down)



    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor=color, linewidth=3.5)
        )

        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                 fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = cv2.imread(image_name)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    CONF_THRESH = 0.4
    NMS_THRESH = 0.2
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, image_name, cls, dets, ax, thresh=CONF_THRESH)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        default='VGGnet_test')
    parser.add_argument('--model', dest='model', help='Model path',
                        # default="./data/demo/VGGnet_fast_rcnn_iter_100000.ckpt")
                        default = "../output/VGGnet_faster_rcnn")

    #"../output/default/kittivoc_train"
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    if args.model == ' ' or not os.path.exists(args.model):
        print ('current path is ' + os.path.abspath(__file__))
        raise IOError(('Error: Model not found.\n'))

    # init session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # load network
    net = get_network(args.demo_net)
    # load model
    print ('Loading network {:s}... '.format(args.demo_net)),
    # saver = tf.train.Saver()
    # saver.restore(sess, args.model)
    saver = tf.train.Saver(write_version=saver_pb2.SaverDef.V1)
    ckpt = tf.train.get_checkpoint_state(args.model)
    print("ckpt:", ckpt)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    print (' done.')

    # Warmup on a dummy image
    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _ = im_detect(sess, net, im)

    # im_names = glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.png')) + \
    #            glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.jpg'))


    dir_1 = "./*.png"
    dir_2 = "./*.jpg"
    im_names = glob.glob(dir_1) + \
               glob.glob(dir_2)
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for {:s}'.format(im_name)
        demo(sess, net, im_name)

    plt.show()

