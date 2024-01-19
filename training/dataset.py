import os
import cv2
import numpy as np

from pycocotools.coco import maskUtils

from tensorpack.dataflow.common import BatchData, MapData
from tensorpack.dataflow.common import TestDataSpeed
from tensorpack.dataflow import PrefetchData

from training.augmentors import ImageScalingAugmentation, ImageRotationAugmentation, ImageCropAugmentation, FlipAugmentation, \
    convert_joints_to_array, convert_array_to_joints, ImageAugmentData
from training.dataflow import COCODataGenerator, KeypointLoader, COCODatasetPaths
from training.label_maps import create_heatmap, create_paf

PAF_MASKS = np.repeat(
    np.ones((46, 46, 1), dtype=np.uint8), 38, axis=2)

HEATMAP_MASKS = np.repeat(
    np.ones((46, 46, 1), dtype=np.uint8), 19, axis=2)

DATA_AUGMENTORS = [
        ImageScalingAugmentation(min_scale_factor=0.5,
                                 max_scale_factor=1.1,
                                 desired_distance=0.6,
                                 interpolation_method=cv2.INTER_CUBIC),

        ImageRotationAugmentation(max_rotation_deg=40,
                                  interpolation=cv2.INTER_CUBIC,
                                  border_mode=cv2.BORDER_CONSTANT,
                                  image_border_value=(128, 128, 128), mask_border_value=1),

        ImageCropAugmentation(crop_width=368, crop_height=368, pivot_perturbation_max=40, 
                              image_border_value=(128, 128, 128),
                              mask_border_value=1),

        FlipAugmentation(parts_count=18, flip_probability=0.5),
    ]


def load_image(components):
    """
    Loads an image from the file path specified in meta.image_path. Assigns the image to
    the field augmented_image of the same meta instance.

    :param components: components
    :return: updated components
    """
    data_point = components[0]
    image_buffer = open(data_point.image_path, 'rb').read()

    if not image_buffer:
        raise Exception('Image not read, path=%s' % data_point.image_path)

    arr = np.frombuffer(image_buffer, np.uint8)
    data_point.source_image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    data_point.image_height, data_point.image_width = data_point.source_image.shape[:2]

    return components


def generate_masks(components):
    """
    Generate masks based on the COCO mask polygons.

    :param components: components
    :return: updated components
    """
    data_point = components[0]
    if data_point.segmentation_masks:
        missing_mask = np.ones((data_point.image_height, data_point.image_width), dtype=np.uint8)
        for segment in data_point.segmentation_masks:
            binary_mask = maskUtils.decode(segment)
            binary_mask = np.logical_not(binary_mask)
            missing_mask = np.bitwise_and(missing_mask, binary_mask)

        data_point.augmented_mask = missing_mask

    return components


def perform_augmentation(components):
    """
    Augmentation of images.

    :param components: components
    :return: updated components.
    """
    data_point = components[0]

    augmented_center = data_point.image_center.copy()
    augmented_joints = convert_joints_to_array(data_point.original_joints)

    for augmentor in DATA_AUGMENTORS:
        augmentation_params = augmentor._get_augment_params(data_point)
        augmented_image, augmented_mask = augmentor._augment(data_point, augmentation_params)

        # Augment joints
        augmented_joints = augmentor._augment_coords(augmented_joints, augmentation_params)

        # Handle special cases like flipping
        if isinstance(augmentor, FlipAugmentation):
            augmented_joints = augmentor.recover_left_right_joints(augmented_joints, augmentation_params)

        # Augment center position
        augmented_center = augmentor._augment_coords(augmented_center, augmentation_params)

        data_point.source_image = augmented_image
        data_point.augmented_mask = augmented_mask

    data_point.augmented_joints = convert_array_to_joints(augmented_joints)
    data_point.augmented_center = augmented_center

    return components

def apply_augmented_mask(components):
    """
    Applies the augmented mask (if exists) to the augmented image.

    :param components: components
    :return: updated components
    """
    data_point = components[0]
    if data_point.augmented_mask is not None:
        for channel in range(3):
            data_point.source_image[:, :, channel] *= data_point.augmented_mask
    return components

def create_scaled_mask(mask, num_layers, stride):
    """
    Helper function to create a stack of scaled-down masks.

    :param mask: mask image
    :param num_layers: number of layers
    :param stride: parameter used to scale down the mask image because it 
                         needs to match the size of the network output.
    :return: scaled-down mask
    """
    scale_factor = 1.0 / stride
    smaller_mask = cv2.resize(mask, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    smaller_mask = smaller_mask[:, :, np.newaxis]
    return np.repeat(smaller_mask, num_layers, axis=2)



def construct_sample(components):
    """
    Constructs a sample for a model.

    :param components: components
    :return: list of final components of a sample.
    """
    data_point = components[0]
    augmented_image = data_point.source_image

    if data_point.augmented_mask is None:
        paf_mask = PAF_MASKS
        heatmap_mask = HEATMAP_MASKS
    else:
        paf_mask = create_scaled_mask(data_point.augmented_mask, 38, stride=8)
        heatmap_mask = create_scaled_mask(data_point.augmented_mask, 19, stride=8)

    generated_heatmap = create_heatmap(KeypointLoader.total_joints_with_background, 46, 46,
                                         data_point.augmented_joints, 7.0, stride=8)

    generated_pafmap = create_paf(KeypointLoader.total_connections, 46, 46,
                                    data_point.augmented_joints, 1, stride=8)

    # Release references to save memory
    data_point.augmented_mask = None
    data_point.source_image = None
    data_point.augmented_joints = None
    data_point.augmented_center = None
    return [augmented_image.astype(np.uint8), paf_mask, heatmap_mask, generated_pafmap, generated_heatmap]

def initialize_dataflow(dataset_paths):
    """
    Initializes the tensorpack dataflow and serves a generator for training.

    :param dataset_paths: paths to the COCO files: annotation file and image folder
    :return: dataflow object
    """
    dataflow = COCODataGenerator((368, 368), dataset_paths)
    dataflow.prepare()
    dataflow = MapData(dataflow, load_image)
    dataflow = MapData(dataflow, generate_masks)
    dataflow = MapData(dataflow, perform_augmentation)
    dataflow = MapData(dataflow, apply_augmented_mask)
    dataflow = MapData(dataflow, construct_sample)
    dataflow = PrefetchData(dataflow, nr_proc=4, nr_prefetch=5)

    return dataflow

def build_batch_dataflow(dataflow, batch_size):
    """
    Builds a batch dataflow from the input dataflow of samples.

    :param dataflow: dataflow of samples
    :param batch_size: batch size
    :return: dataflow of batches
    """
    batched_dataflow = BatchData(dataflow, batch_size, use_list=False)
    batched_dataflow = MapData(batched_dataflow, lambda x: (
        [x[0], x[1], x[2]],
        [x[3], x[4], x[3], x[4], x[3], x[4], x[3], x[4], x[3], x[4], x[3], x[4]])
    )
    batched_dataflow.reset_state()
    return batched_dataflow

if __name__ == '__main__':
    """
    Run this script to check speed of generating samples. Tweak the nr_proc
    parameter of PrefetchDataZMQ. Ideally it should reflect the number of cores 
    in your hardware
    """
    batch_size = 10
    curr_dir = os.path.dirname(__file__)
    annot_path = os.path.join(curr_dir, '../dataset/annotations/person_keypoints_val2017.json')
    img_dir = os.path.abspath(os.path.join(curr_dir, '../dataset/val2017/'))
    df = COCODataGenerator((368, 368), COCODatasetPaths(annot_path, img_dir))#, select_ids=[1000])
    df.prepare()
    df = MapData(df, load_image)
    df = MapData(df, generate_masks)
    df = MapData(df, apply_augmented_mask)
    df = MapData(df, create_scaled_mask)
    df = MapData(df, construct_sample)
    df = PrefetchData(df, nr_proc=4, nr_prefetch=5)
    df = BatchData(df, batch_size, use_list=False)
    df = MapData(df, lambda x: (
        [x[0], x[1], x[2]],
        [x[3], x[4], x[3], x[4], x[3], x[4], x[3], x[4], x[3], x[4], x[3], x[4]])
    )

    TestDataSpeed(df, size=100).start()
