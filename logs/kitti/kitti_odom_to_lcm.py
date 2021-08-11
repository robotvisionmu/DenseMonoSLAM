import argparse
import glob
import os

import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
from numpy.linalg import inv
import onnxruntime
from matplotlib import pyplot as plt


from lcm import EventLog 
from lcm import Event
from eflcm.Frame import Frame

import os, sys, getopt, datetime, re, string
from collections import defaultdict
from collections import namedtuple

from PIL import Image
from PIL import ImageOps

import pykitti

import yaml

basename = 'kitti.lcmlog.'
extension = ".lcm"
orb_template_params = "KITTI_RGBD_template_params.yaml"
kitti_virtual_stereo_baseline = 0.54

def corrected_intrinsics(fx, fy, cx, cy, feed_height, feed_width, image_height, image_width):

    K = np.identity(3, dtype=np.float32)
    K[0,0] = fx
    K[1,1] = fy
    K[0,2] = cx
    K[1,2] = cy
    dist_coeffs = np.zeros(4)
    new_k, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (image_width, image_height), 0.0, (feed_width, feed_height),centerPrincipalPoint=True)
   
    fx_new = new_k[0,0]
    fy_new = new_k[1,1]
    cx_new = new_k[0,2] - 0.5
    cy_new = new_k[1,2] - 0.5

    x_ratio = feed_width / image_width
    y_ratio = feed_height / image_height
    
    return fx* x_ratio, fy* y_ratio, cx * x_ratio, cy * y_ratio, roi

    #return fx_new, fy_new, cx_new, cy_new, roi

def post_process_distance(output):
    inv_norm = 1 / output
    vmax = np.percentile(inv_norm, 95)
    normalizer = mpl.colors.Normalize(vmin=inv_norm.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap=cm.get_cmap('magma'))
    colormapped_im = (mapper.to_rgba(inv_norm)[:, :, :3] * 255).astype(np.uint8)
    return colormapped_im


def predict_depth(ort_session, image_frame, data_type):   
    input_frame = ort_session.get_inputs()[0].name    

    frame = np.expand_dims(image_frame, axis=0).astype(data_type) / 255
    output = ort_session.run(None, {input_frame: frame})
    
    # colored_output = post_process_distance(output[0][0])
    # plt.imshow(colored_output)
    # plt.show(block=True)

    return output[0][0]

def display_result(input_image, output_depth):
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(input_image)
    axarr[1].imshow(output_depth)
    plt.show(block=True)    

def load_timestamps(timestamps_file):
    timestamps = []
    with open(timestamps_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            time  = float(line)
            timestamps.append(time)
    return timestamps

def load_gt_poses(gt_poses_file_name):
    gt_poses = []
    with open(gt_poses_file_name) as gt_poses_file:
        lines = gt_poses_file.readlines()
        for line in lines:
            P_i = np.array([float(x) for x in line.split()]).astype(np.float32).reshape(3,4)
            gt_poses.append(P_i)

    return gt_poses


def save_intrinsics(intrinsics_file, adjusted_intrinsics_file, kitti_sequence_cam_no, feed_height, feed_width, image_height, image_width):
    with open(intrinsics_file, 'r') as intrinsics:
        line = intrinsics.readlines()[int(kitti_sequence_cam_no)]
        K = np.array([float(x) for x in line.split()[1:]]).reshape(3,4)
        print("K: ")
        print(K)
        fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
        print("original intrinsics from K: %f %f %f %f" % (fx, fy, cx, cy))
        fx_r, fy_r, cx_r, cy_r, roi = corrected_intrinsics(fx, fy, cx, cy, feed_height, feed_width, image_height, image_width)
        print("adjusted intrinsics: %f %f %f %f" % (fx_r, fy_r, cx_r, cy_r))
        with open(adjusted_intrinsics_file, 'w') as corrected_intrinsics_file:
            corrected_intrinsics_file.write("%f %f %f %f" % (fx_r, fy_r, cx_r, cy_r))
        return fx, fy, cx, cy, fx_r, fy_r, cx_r, cy_r, roi
        

def prepend_yaml_version(filename):
    yaml_version = "%YAML:1.0"
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(yaml_version.rstrip('\r\n') + '\n' + content)

def save_orb_params(orb_params_file, adjusted_intrinsics_file, feed_height, feed_width):
    with open(adjusted_intrinsics_file, 'r') as intrinsics:
        line = intrinsics.readlines()[0]
        K = np.array([float(x) for x in line.split()])
        fx, fy, cx, cy = K[0], K[1], K[2], K[3]
        with open(orb_template_params, 'r') as template_params_file:
            template_params = yaml.load(template_params_file)
            template_params["Camera.fx"] = float(fx)
            template_params["Camera.fy"] = float(fy)
            template_params["Camera.cx"] = float(cx)
            template_params["Camera.cy"] = float(cy)
            template_params["Camera.height"] = feed_height
            template_params["Camera.width"] = feed_width
            template_params["Camera.bf"] = kitti_virtual_stereo_baseline * float(fx)
            for item, val in template_params.items():
                print(item, ":", val)
            with open(orb_params_file, 'w') as orb_params:
                yaml.dump(template_params, orb_params)
            prepend_yaml_version(orb_params_file)

def compute_lcm_sequence(kitti_sequence_base_dir, kitti_sequence_no, kitti_sequence_cam_no, lcm_channel, 
        trackOnly, make_depth_prediction, ort_session, feed_height, feed_width, data_type):
    
    outdir = "%s/lcm_sequences/%s/image_%s/" % (kitti_sequence_base_dir,kitti_sequence_no,kitti_sequence_cam_no)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    outname = "%s/lcm_sequences/%s/image_%s/sequence.lcm" % (kitti_sequence_base_dir,kitti_sequence_no,kitti_sequence_cam_no)    

    raw_images_dir = "%s/raw_images/" %(outdir)
    if not os.path.isdir(raw_images_dir):
        os.makedirs(raw_images_dir)

    kitti_sequence_dir = "%s/sequences/%s/image_%s/" % (kitti_sequence_base_dir, kitti_sequence_no, kitti_sequence_cam_no)

    filenames = [f for f in os.listdir(kitti_sequence_dir)]

    file_map = defaultdict(list)
    for f in filenames:
        parts = re.split(r'[_\.]+', f)
        if parts[1] in ('png', 'depth'):
            file_map[int(parts[0])].append(f)

    print( 'writing to file: {}'.format(outname))

    indexes = sorted(file_map)

    kitti_sequence_gt_poses = load_gt_poses("%s/poses/%s.txt" % (kitti_sequence_base_dir, kitti_sequence_no))

    kitti_seqeuence_images_timestamps = load_timestamps("%s/sequences/%s/times.txt" % (kitti_sequence_base_dir, kitti_sequence_no))
    start_idx = 0
    end_idx = int(len(kitti_seqeuence_images_timestamps) / 1)
    
    with open("%s/lcm_sequences/%s/image_%s/sequence.gt.freiburg" % (kitti_sequence_base_dir, kitti_sequence_no, kitti_sequence_cam_no), 'w') as gt_file:
        for i, ts in enumerate(kitti_seqeuence_images_timestamps[start_idx:end_idx]):
            T_cam = kitti_sequence_gt_poses[i]
            line = "%i,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n" % (i,T_cam[0][0], T_cam[0][1], T_cam[0][2], T_cam[0][3], T_cam[1][0], T_cam[1][1], T_cam[1][2], T_cam[1][3], T_cam[2][0], T_cam[2][1], T_cam[2][2], T_cam[2][3])
            gt_file.write(line)

    files = sorted(file_map[0])
    image = cv2.imread(os.path.join(kitti_sequence_dir + "/", files[0]), cv2.COLOR_BGR2RGB)
    image_height = image.shape[0]
    image_width = image.shape[1]
    intrinsics_file = "%s/sequences/%s/calib.txt" % (kitti_sequence_base_dir, kitti_sequence_no)
    adjusted_intrinsics_file = "%s/lcm_sequences/%s/image_%s/sequence.calib.txt" % (kitti_sequence_base_dir, kitti_sequence_no, kitti_sequence_cam_no)

    fx, fy, cx, cy, fx_r, fy_r, cx_r, cy_r, roi = save_intrinsics(intrinsics_file, adjusted_intrinsics_file, kitti_sequence_cam_no, feed_height, feed_width, image_height, image_width)

    K = np.identity(3)
    K[0,0] = fx
    K[1,1] = fy
    K[0,2] = cx
    K[1,2] = cy
    K_new = np.identity(3)
    K_new[0,0] = fx_r
    K_new[1,1] = fy_r
    K_new[0,2] = cx_r
    K_new[1,2] = cy_r
    R = np.identity(3)
    distCoeffs = np.zeros(4)
    map_x, map_y = cv2.initUndistortRectifyMap(K, None, None, K_new, (feed_width, feed_height), cv2.CV_32FC1)

    orb_params_file = "%s/lcm_sequences/%s/image_%s/sequence.orb.params.yaml" % (kitti_sequence_base_dir, kitti_sequence_no, kitti_sequence_cam_no)

    save_orb_params(orb_params_file, adjusted_intrinsics_file, feed_height, feed_width)

    lcm_log = EventLog(outname, 'w', True)

    for i in indexes[start_idx:end_idx]:
        print ('writing frame: {}'.format(i))
        files = sorted(file_map[i])
        image = cv2.imread(os.path.join(kitti_sequence_dir + "/", files[0]), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (feed_width, feed_height), cv2.INTER_LANCZOS4)
        image_network = image.transpose((2, 0, 1))
        depth = predict_depth(ort_session, image_network, data_type) if make_depth_prediction else np.zeros((feed_width, feed_height), dtype=np.float32)
        
        #display_result(image, post_process_distance(depth))

        cv2.imwrite("%s/rgb_%05d.png" %(raw_images_dir,i), image)
        depth_image =  np.array(depth*1000.0).clip(0.0, 65535.0).astype(np.uint16)
        cv2.imwrite("%s/depth_%05d.png" %(raw_images_dir,i), depth_image)

        f = Frame()
        f.trackOnly = trackOnly
        f.compressed = False
        f.last = i == indexes[-1]
        f.depthSize = depth.size * 2
        f.imageSize = image.size
        f.depth = np.array(depth*1000.0).clip(0.0, 65535.0).flatten().astype(np.uint16).tobytes()
        f.image = np.array(image).flatten().tobytes()
        f.timestamp = i - start_idx
        f.frameNumber = i - start_idx
        f.senderName = outname

        e = Event(i - start_idx, (i - start_idx) * 33000, lcm_channel, f.encode())

        lcm_log.write_event(e.timestamp, e.channel, e.data)

    lcm_log.close()

def main(argv):
    kitti_sequence_base_dir = ''
    kitti_sequence_nos = []
    kitti_sequence_cam_no = ''
    lcm_channel = ''
    depth_prediction_model = ''
    trackOnly = False
    make_depth_prediction = False
	
    try:
        opts, args = getopt.getopt(argv,"hb:s:c:e:m:tp",["base_directory=", "seq_no=","cam=" ,"ef_lcm_channel=," "model=", "trackOnly=", "predict_depth="])
    except getopt.GetoptError:
	    print('kitti_odom_to_lcm.py -b <kitti_base_directory> -c <cam_no> -s <sequence_no> -e <ef_lcm_channel> -m <onnx_depth_prediction_model> -t <track only> -p <predict_depth>')
	    sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('kitti_odom_to_lcm.py -b <kitti_base_directory> -c <cam_no> -s <sequence_no> -e <ef_lcm_channel> -m <onnx_depth_prediction_model> -t <track only> -p <predict_depth>')
            sys.exit()
        if opt in ("-b", "--base_directory"):
            kitti_sequence_base_dir = arg
        if opt in ("-s", "--seq_no"):
            kitti_sequence_nos.append(arg)
        if opt in ("-c", "--cam"):
            kitti_sequence_cam_no = arg
        if opt in ("-e", "--ef_lcm_channel"):
            lcm_channel = arg
        if opt in ("-m", "--model"):
            depth_prediction_model = arg
        if opt in ("-t", "--trackOnly"):
            trackOnly = True
        if opt in ("-p", "--predict_depth"):
            make_depth_prediction = True

    ort_session = onnxruntime.InferenceSession(depth_prediction_model)
    feed_height = ort_session.get_inputs()[0].shape[2]
    feed_width = ort_session.get_inputs()[0].shape[3]
    print("%d, %d" % (feed_width, feed_height))
    data_type = np.float16 if "16" in depth_prediction_model else np.float32
    print(onnxruntime.get_device())

    for kitti_sequence_no in kitti_sequence_nos:
        compute_lcm_sequence(kitti_sequence_base_dir, kitti_sequence_no, kitti_sequence_cam_no, lcm_channel, 
        trackOnly, make_depth_prediction, ort_session, feed_height, feed_width, data_type)

if __name__ == '__main__':
	main(sys.argv[1:])