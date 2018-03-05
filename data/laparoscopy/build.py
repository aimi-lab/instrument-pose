import numpy as np
import json
import math
import imageio
import copy
from PIL import Image
import itertools
import matplotlib as mpl
mpl.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

imageio.plugins.ffmpeg.download()

num_objects_max = 4
num_joints_max = 5


training_video_files = ['Video_1.avi', 'Video_2.avi', 'Video_3.avi', 'Video_4.avi', 'Test_Video_6.avi' ]
training_pose_files = [
    'training_labels/train1_labels.json', 'training_labels/train2_labels.json',
    'training_labels/train3_labels.json', 'training_labels/train4_labels.json',
    'test_labels/test6_changed_labels.json'
    ]

eval_video_files = ['Test_Video_1.avi','Test_Video_2.avi', 'Test_Video_3.avi','Test_Video_4.avi', 'Test_Video_5.avi']

eval_pose_files = [
    'test_labels/test1_changed_labels.json', 'test_labels/test2_changed_labels.json',
    'test_labels/test3_changed_labels.json', 'test_labels/test4_changed_labels.json',
    'test_labels/test5_changed_labels.json']


joint_order = {"LeftClasperPoint": 0, "RightClasperPoint": 1, "ShaftPoint": 2, "TrackedPoint": -1, "EndPoint": 3, "HeadPoint": 4}



object_order = {"tool1": 0, "tool2": 1, "tool3": 2, "tool4": 3}




def save_frame(frame, datadir, frame_number, video_number, rot):
    """Helper function that saves a frame to a file
    Takes a numpy frame and stores it in a file
    Args:
        frame: Numpy frame
        datadir: Directory to store frame
        frame_number: Frame number in video
        video_number: Video Number
        rot: Rotation used
    Returns:
        filename: Filename of frame
    """
    filename = datadir + str(frame_number) + "_"+str(video_number) + "_" + str(rot) + ".png"
    image_out = Image.fromarray(frame)
    image_out.save(filename)
    return filename


def load_positions(file):
    """Loads json annotations file
    Takes a numpy frame and stores it in a file
    Args:
        file: JSON filename to load
    Returns:
        positions: array containing joint position annotations
        objects: array containing object presence
    """
    json_file = open(file)
    data = json.load(json_file)
    positions = np.ones([len(data), num_objects_max*num_joints_max*2]) * -1e6
    objects = np.zeros([len(data), num_objects_max])

    for x in range(0, len(data)):
        annotations = data[x]['annotations']
        for a in range(0, len(annotations)):
            annotation = annotations[a]
            try:
                object = object_order[annotation['id']]
            except:
                object = 0
            objects[x, object] = 1.0
            joint_num = joint_order[annotation['class']]
            if(joint_num >= 0):
                position = np.zeros(2)
                position[0] = annotation['y']
                position[1] = annotation['x']
                positions[x, object*num_joints_max*2+joint_num*2: object*num_joints_max*2+joint_num*2 +2 ] = position
    return positions, objects


def flip_up_down(image, positions, objects):
    """Flips frame up-down
    Args:
        image: image frame
        positions: joint position annotations in frame
        objects: object presence in frame
    Returns:
        image: flipped image frame
        positions: flipped positions
        objects: flipped objects
    """
    image = np.flipud(image)
    for y_pos in range(0, num_joints_max*num_objects_max*2, 2):
        positions[y_pos] = np.absolute(image.shape[0] - positions[y_pos])

    # we also need to flip the left right tips of each instrument...
    for o in range(0, num_objects_max):
        idx = num_joints_max*2*o
        temp = np.copy(positions[idx:idx+2])
        positions[idx:idx+2] = positions[idx+2:idx+4]
        positions[idx+2:idx+4] = temp

    return image, positions, objects


def flip_left_right(image, positions, objects):
    """Flips frame left-right
    Args:
        image: image frame
        positions: joint position annotations in frame
        objects: object presence in frame
    Returns:
        image: flipped image frame
        positions: flipped positions
        objects: flipped objects
    """
    image = np.fliplr(image)
    print(image.shape[1])
    for x_pos in range(1,num_joints_max*num_objects_max*2, 2):
        positions[x_pos] =  np.absolute(image.shape[1] - positions[x_pos])

    # Clamper
    positions_temp = np.copy(positions[0:num_joints_max*2])
    positions[0:num_joints_max*2] = positions[num_joints_max*2:num_joints_max * 2 * 2]
    positions[num_joints_max*2:num_joints_max * 2 * 2] = positions_temp
    objects_temp = np.copy(objects[0])
    objects[0] = objects[1]
    objects[1] = objects_temp

    # Scissor
    positions_temp  = np.copy(positions[num_joints_max * 2 * 2:num_joints_max * 2 * 3])
    positions[num_joints_max * 2 * 2: num_joints_max * 2 * 3] = positions[num_joints_max * 2 * 3: num_joints_max * 2 * 4]
    positions[num_joints_max * 2 * 3: num_joints_max * 2 * 4] = positions_temp
    objects_temp = np.copy(objects[2])
    objects[2] = objects[3]
    objects[3] = objects_temp

    #we also need to flip the left right tips of each instrument...
    for o in range(0, num_objects_max):
        idx = num_joints_max*2*o
        temp = np.copy(positions[idx:idx+2])
        positions[idx:idx+2] = positions[idx+2:idx+4]
        positions[idx+2:idx+4] = temp

    return image, positions, objects


def create_set(name, videodir, videofiles, posefiles, datadir, rotations):
    """Take annotations and videos and create sets usable with the training and evaluation scripts
    Args:
        name: name of set
        videodir: base directory containing videos
        videofiles: list of video files
        posefiles: list of position annotation files
        datadir: directory to store the frames
        rotations: rotations to apply 0,1,2,3 (* 90°)
    Returns:
        nothing
    """
    for v in range(0, len(videofiles)):

            positions_load, objects_load = load_positions(posefiles[v])
            video_file = videodir + videofiles[v]
            vid = imageio.get_reader(video_file,  'ffmpeg')
            image_filenames = np.array([])  # out array of filenames
            positions = np.empty((0,num_joints_max*num_objects_max*2), float)
            objects = np.empty((0,num_objects_max), float)
            for seq in range(0, vid.get_length()):
                if(np.sum(positions_load[seq, :]) !=-1e6*num_joints_max*num_objects_max*2):
                    print(seq)
                    for rot in rotations[v]:
                        image = vid.get_data(seq)
                        pos_image = np.copy(positions_load[seq,:])
                        obj_image = np.copy(objects_load[seq,:])

                        if rot == 1:
                            image, pos_image, obj_image = flip_up_down(image, pos_image, obj_image)
                        elif rot == 2:
                            image, pos_image, obj_image = flip_left_right(image, pos_image, obj_image)
                        elif rot == 3:
                            image, pos_image, obj_image = flip_up_down(image, pos_image, obj_image)

                            image, pos_image, obj_image = flip_left_right(image, pos_image, obj_image)


                        filename = save_frame(image, datadir, seq, v, rot)  # save the frame and get filename

                        # also plot them for sanity checks...
                        # plt.imshow(image)
                        # colors = itertools.cycle(['b', 'r', 'g', 'y', 'c'])
                        # for o in range(0, num_objects_max):
                        #
                        #     for p in range(0, num_joints_max):
                        #         idx = o*num_joints_max*2 + p*2
                        #         color = next(colors)
                        #         if(pos_image[idx] > 0.0 and pos_image[idx] < 1000.0):
                        #             plt.scatter(pos_image[idx+1], pos_image[idx],    color=color, s=10)

                        # plt.savefig("sanity/" + str(seq)+"_"+str(v)+ "_" + str(rot) + ".jpg")
                        # plt.close()
                        image_filenames = np.append(image_filenames, filename)

                        positions = np.vstack((positions, pos_image))
                        objects = np.vstack((objects, obj_image))
                # else:
                #     idx_delete = np.append(idx_delete, seq)
            # positions = np.delete(positions, idx_delete, axis=0)
            # objects = np.delete(objects, idx_delete, axis=0)
            image_filenames = np.expand_dims(image_filenames, axis=1)
            print(image_filenames.shape)
            print(objects.shape)
            print(positions.shape)
            sequence_list = np.ones(image_filenames.shape) * v
            out = np.hstack((np.hstack((np.hstack((image_filenames, sequence_list)), objects)),positions))
            print(out.shape)
            if v == 0:
                out_tot = out
            else:
                out_tot = np.vstack((out_tot,out))

    print(out.shape)
    print(out_tot.shape)

    np.savetxt(name + ".csv", out_tot, delimiter=",", fmt="%s")


def main(argv=None):  # pylint: disable=unused-argument
    create_set('training_laparoscopy', 'training_videos/', training_video_files, training_pose_files, 'data_train/', [[0, 1, 2, 3,], [0, 1, 2, 3,], [0, 1, 2, 3,], [0, 1, 2, 3,], [0, 1, 2, 3,]])
    create_set('eval_laparoscopy', 'eval_videos/', eval_video_files, eval_pose_files, 'data_eval/', [[0], [0], [0], [0], [0,2]])

if __name__ == "__main__":
    main()
