import numpy as np
import cv2

from tensorpack.dataflow.imgaug.base import ImageAugmentor
from tensorpack.dataflow.imgaug.transform import ResizeTransform
from tensorpack.dataflow.imgaug.geometry import WarpAffineTransform,TransformAugmentorBase


class ImageAugmentData:
    """
    Holder for data required for augmentation - subset of metadata
    """
    __slots__ = ["source_image", "augmented_mask", "pivot_point", "scaling_factor"]

    def __init__(self, source_image, augmented_mask, pivot_point, scaling_factor):
        self.source_image = source_image
        self.augmented_mask = augmented_mask
        self.pivot_point = pivot_point
        self.scaling_factor = scaling_factor

    def update_source_image(self, updated_image, updated_mask):
        return ImageAugmentData(updated_image, updated_mask, self.pivot_point, self.scaling_factor)


def convert_joints_to_array(joint_data, points_count=18):
    """
    Converts joint structure to Nx2 nparray (format expected by tensorpack augmentors)
    Nx2 = floating point nparray where each row is (x, y)

    :param joint_data:
    :param points_count:
    :return: Nx2 nparray
    """
    array_data = np.zeros((points_count * len(joint_data), 2), dtype=float)

    for combined_idx, joint_list in enumerate(joint_data):
        for individual_idx, joint_point in enumerate(joint_list):
            if joint_point:
                array_data[combined_idx * points_count + individual_idx, 0] = joint_point[0]
                array_data[combined_idx * points_count + individual_idx, 1] = joint_point[1]
            else:
                array_data[combined_idx * points_count + individual_idx, 0] = -1000000
                array_data[combined_idx * points_count + individual_idx, 1] = -1000000

    return array_data


def convert_array_to_joints(array_points, points_count=18):
    """
    Converts Nx2 nparray to the list of joints

    :param array_points:
    :param points_count:
    :return: list of joints [[(x1,y1), (x2,y2), ...], []]
    """
    list_length = array_points.shape[0] // points_count

    joint_groups = []
    for group_index in range(list_length):
        single_joint_group = []
        for point_index in range(points_count):
            idx = group_index * points_count + point_index
            x_coord = array_points[idx, 0]
            y_coord = array_points[idx, 1]

            if x_coord <= 0 or y_coord <= 0 or x_coord > 2000 or y_coord > 2000:
                single_joint_group.append(None)
            else:
                single_joint_group.append((x_coord, y_coord))

        joint_groups.append(single_joint_group)
    return joint_groups


class FlipAugmentation(ImageAugmentor):
    """
    Flips images and coordinates
    """
    def __init__(self, parts_count, flip_probability=0.5):
        super(FlipAugmentation, self).__init__()
        self._init(locals())

    def _get_augment_params(self, augment_data):
        source_image = augment_data.source_image

        _, image_width = source_image.shape[:2]

        should_flip = self._rand_range() < self.flip_probability
        return (should_flip, image_width)

    def _augment(self, augment_data, flip_params):
        source_image = augment_data.source_image
        augmented_mask = augment_data.augmented_mask

        should_flip, _ = flip_params

        if should_flip:
            flipped_image = cv2.flip(source_image, 1)
            if source_image.ndim == 3 and flipped_image.ndim == 2:
                flipped_image = flipped_image[:, :, np.newaxis]

            if augmented_mask is not None:
                flipped_mask = cv2.flip(augmented_mask, 1)
            else:
                flipped_mask = None

            augment_result = (flipped_image, flipped_mask)
        else:
            augment_result = (source_image, augmented_mask)

        return augment_result

    def _augment_coords(self, joint_coords, flip_params):
        should_flip, image_width = flip_params
        if should_flip:
            joint_coords[:, 0] = image_width - joint_coords[:, 0]

        return joint_coords

    def recover_left_right_joints(self, joint_coords, flip_params):
        """
        Recovers a few joints. After flip operation coordinates of some parts like
        left hand would land on the right side of a person so it is
        important to recover such positions.

        :param joint_coords:
        :param flip_params:
        :return:
        """
        should_flip, _ = flip_params
        if should_flip:
            right_joints = [2, 3, 4, 8, 9, 10, 14, 16]
            left_joints = [5, 6, 7, 11, 12, 13, 15, 17]

            for left_idx, right_idx in zip(left_joints, right_joints):
                idxs = range(0, joint_coords.shape[0], self.parts_count)
                for idx in idxs:
                    tmp = joint_coords[left_idx + idx, [0, 1]]
                    joint_coords[left_idx + idx, [0, 1]] = joint_coords[right_idx + idx, [0, 1]]
                    joint_coords[right_idx + idx, [0, 1]] = tmp

        return joint_coords


class ImageCropAugmentation(ImageAugmentor):
    """
    Crops images and coordinates
    """
    def __init__(self, crop_width, crop_height, pivot_perturbation_max=40, image_border_value=0, mask_border_value=0):
        super(ImageCropAugmentation, self).__init__()
        self._init(locals())

    def _get_augment_params(self, augment_data):
        pivot_point = augment_data.image_center

        x_perturb = int(self._rand_range(-0.5, 0.5) * 2 * self.pivot_perturbation_max)
        y_perturb = int(self._rand_range(-0.5, 0.5) * 2 * self.pivot_perturbation_max)

        new_center_x = pivot_point[0, 0] + x_perturb
        new_center_y = pivot_point[0, 1] + y_perturb

        top_left_corner = (int(new_center_x - self.crop_width / 2),
                           int(new_center_y - self.crop_height / 2))

        return top_left_corner

    def _augment(self, augment_data, top_left_corner):
        source_image = augment_data.source_image
        augmented_mask = augment_data.augmented_mask

        x1, y1 = top_left_corner

        blank_image = np.ones((self.crop_height, self.crop_width, 3), dtype=np.uint8) * self.image_border_value

        if x1 < 0:
            dx = -x1
        else:
            dx = 0

        if y1 < 0:
            dy = -y1
        else:
            dy = 0

        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0

        cropped_image = source_image[y1:y1+self.crop_height-dy, x1:x1+self.crop_width-dx, :]
        cropped_height, cropped_width = cropped_image.shape[:2]
        blank_image[dy:dy+cropped_height, dx:dx+cropped_width, :] = cropped_image

        if augmented_mask is not None:
            blank_mask = np.ones((self.crop_height, self.crop_width), dtype=np.uint8) * self.mask_border_value
            cropped_mask = augmented_mask[y1:y1 + self.crop_height - dy, x1:x1 + self.crop_width - dx]
            mask_height, mask_width = cropped_mask.shape[:2]
            blank_mask[dy:dy + mask_height, dx:dx + mask_width] = cropped_mask
        else:
            blank_mask = augmented_mask

        return blank_image, blank_mask

    def _augment_coords(self, joint_coords, top_left_corner):

        joint_coords[:, 0] -= top_left_corner[0]
        joint_coords[:, 1] -= top_left_corner[1]

        return joint_coords

class ImageScalingAugmentation(TransformAugmentorBase):
    def __init__(self, min_scale_factor, max_scale_factor, desired_distance = 1.0, interpolation_method=cv2.INTER_CUBIC):
        super(ImageScalingAugmentation, self).__init__()
        self._init(locals())

    def _get_augment_params(self, augment_data):
        source_image = augment_data.source_image
        scaling_factor = augment_data.scaling_factor

        image_height, image_width = source_image.shape[:2]

        random_scale_multiplier = self._rand_range(self.min_scale_factor, self.max_scale_factor)

        absolute_scale = self.desired_distance / scaling_factor

        adjusted_scale = absolute_scale * random_scale_multiplier

        new_height, new_width = int(adjusted_scale * image_height + 0.5), int(adjusted_scale * image_width + 0.5)

        return ResizeTransform(
            image_height, image_width, new_height, new_width, self.interpolation_method)

    def _augment(self, augment_data, resize_transform):
        resized_image = resize_transform.apply_image(augment_data.source_image)

        if augment_data.augmented_mask is not None:
            resized_mask = resize_transform.apply_image(augment_data.augmented_mask)
        else:
            resized_mask = None

        return resized_image, resized_mask


class ImageRotationAugmentation(TransformAugmentorBase):
    """
    Rotates images and coordinates
    """
    def __init__(self, scale_factor=None, translation_fraction=None, max_rotation_deg=0.0, shear_factor=0.0,
                 interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REPLICATE, image_border_value=0, mask_border_value=0):

        super(ImageRotationAugmentation, self).__init__()
        self._init(locals())

    def _get_augment_params(self, augment_data):
        source_image = augment_data.source_image

        image_height, image_width = source_image.shape[:2]
        image_center_x, image_center_y = (image_width // 2, image_height // 2)
        rotation_degrees = self._rand_range(-self.max_rotation_deg, self.max_rotation_deg)
        rotation_matrix = cv2.getRotationMatrix2D((image_center_x, image_center_y), rotation_degrees, 1.0)

        # calculate new bounding box dimensions
        cos_value = np.abs(rotation_matrix[0, 0])
        sin_value = np.abs(rotation_matrix[0, 1])
        new_width = int((image_height * sin_value) + (image_width * cos_value))
        new_height = int((image_height * cos_value) + (image_width * sin_value))
        rotation_matrix[0, 2] += (new_width / 2) - image_center_x
        rotation_matrix[1, 2] += (new_height / 2) - image_center_y

        return WarpAffineTransform(rotation_matrix, (new_width, new_height),
                            self.interpolation, self.border_mode, self.image_border_value)

    def _augment(self, augment_data, warp_transform):
        rotated_image = warp_transform.apply_image(augment_data.source_image)

        if augment_data.augmented_mask is not None:
            warp_transform.borderValue = self.mask_border_value
            rotated_mask = warp_transform.apply_image(augment_data.augmented_mask)
        else:
            rotated_mask = None

        return rotated_image, rotated_mask
