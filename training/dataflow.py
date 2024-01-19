import os
import numpy as np

from scipy.spatial.distance import cdist
from pycocotools.coco import COCO
from tensorpack.dataflow.base import RNGDataFlow


class KeypointLoader:
    """
    Loader for keypoints from coco keypoints
    """
    @staticmethod
    def _calculate_neck(keypoints, left_shoulder_idx, right_shoulder_idx):
        left_shoulder = keypoints[left_shoulder_idx]
        right_shoulder = keypoints[right_shoulder_idx]
        if left_shoulder and right_shoulder:
            return (left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2
        else:
            return None

    total_joints = 18
    total_joints_with_background = total_joints + 1
    total_connections = 19

    coco_indices = [0, lambda kp: KeypointLoader._calculate_neck(kp, 5, 6), 6, 8,
                    10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]

    coco_joint_names = [
        'Nose','Neck','Right Shoulder','Right Elbow','Right Wrist','Left Shoulder','Left Elbow','Left Wrist',
        'Right Hip','Right Knee','Right Ankle','Left Hip','Left Knee','Left Ankle','Right Eye','Left Eye','Right Ear','Left Ear']

    paired_joints = list(zip(
        [1, 8, 9, 1, 11, 12, 1, 2, 3, 2, 1, 5, 6, 5, 1, 0, 0, 14, 15],
        [8, 9, 10, 11, 12, 13, 2, 3, 4, 16, 5, 6, 7, 17, 0, 14, 15, 16, 17]))

    @staticmethod
    def load_from_coco_keypoints(all_keypoints, image_width, image_height):
        """
        Creates a list of keypoints based on the list of coco keypoints vectors.

        :param all_keypoints: list of coco keypoints vector [[x1,y1,v1,x2,y2,v2,....], []]
        :param image_width: image width
        :param image_height: image height
        :return: list of keypoints [[(x1,y1), (x1,y1), ...], [], []]
        """
        all_joint_lists = []
        for keypoints in all_keypoints:
            keypoints_array = np.array(keypoints)
            xs = keypoints_array[0::3]
            ys = keypoints_array[1::3]
            visibilities = keypoints_array[2::3]

            # Filter and load keypoints to the list
            processed_keypoints = []
            for idx, (x, y, visibility) in enumerate(zip(xs, ys, visibilities)):
                # Only visible and occluded keypoints are used
                if visibility >= 1 and 0 <= x < image_width and 0 <= y < image_height:
                    processed_keypoints.append((x, y))
                else:
                    processed_keypoints.append(None)

            # Build the list of joints with the same coordinates as in the original coco keypoints
            # plus additional body parts interpolated from coco keypoints (e.g., a neck)
            joint_list = []
            for part_idx in range(len(KeypointLoader.coco_indices)):
                coco_keypoint_idx = KeypointLoader.coco_indices[part_idx]

                if callable(coco_keypoint_idx):
                    joint = coco_keypoint_idx(processed_keypoints)
                else:
                    joint = processed_keypoints[coco_keypoint_idx]

                joint_list.append(joint)
            all_joint_lists.append(joint_list)

        return all_joint_lists


class TrainingDataPoint(object):
    """
    Metadata representing a single data point for training.
    """
    __slots__ = (
        'image_path',
        'image_height',
        'image_width',
        'image_center',
        'bounding_box',
        'image_area',
        'keypoints_count',
        'segmentation_masks',
        'scaling_factor',
        'original_joints',
        'source_image',
        'augmented_mask',
        'augmented_center',
        'augmented_joints')

    def __init__(self, image_path, image_height, image_width, image_center, bounding_box,
                 image_area, scaling_factor, keypoints_count):

        self.image_path = image_path
        self.image_height = image_height
        self.image_width = image_width
        self.image_center = image_center
        self.bounding_box = bounding_box
        self.image_area = image_area
        self.scaling_factor = scaling_factor
        self.keypoints_count = keypoints_count

        # updated after iterating over all persons
        self.segmentation_masks = None
        self.original_joints = None

        # updated during augmentation
        self.source_image = None
        self.augmented_mask = None
        self.augmented_center = None
        self.augmented_joints = None


class COCODatasetPaths:
    """
    Holder for COCO dataset paths
    """
    def __init__(self, annot_path, image_directory):
        self.annotations = COCO(annot_path)
        self.image_directory = image_directory


class COCODataGenerator(RNGDataFlow):
    """
    Tensorpack dataflow serving COCO data points.
    """
    def __init__(self, desired_size, dataset_paths, selected_ids=None):
        """
        Initializes dataflow.

        :param desired_size:
        :param dataset_paths: paths to the COCO files: annotation file and image folder
        :param selected_ids: (optional) identifiers of images to serve (for debugging)
        """
        self.dataset_paths = dataset_paths if isinstance(dataset_paths, list) else [dataset_paths]
        self.metadata_list = []
        self.selected_ids = selected_ids
        self.desired_size = desired_size

    def prepare(self):
        """
        Loads COCO metadata. Partially populates metadata objects (image path,
        scale of main person, bounding box, area, joints). Remaining fields
        are populated in next steps - MapData tensorpack transformer.
        """
        for dataset in self.dataset_paths:

            print("Loading dataset {} ...".format(dataset.image_directory))

            image_ids = self.selected_ids if self.selected_ids else list(dataset.annotations.imgs.keys())

            for i, img_id in enumerate(image_ids):
                image_metadata = dataset.annotations.imgs[img_id]

                # load annotations
                image_id = image_metadata['id']
                image_file_name = image_metadata['file_name']
                height, width = image_metadata['height'], image_metadata['width']
                image_path = os.path.join(dataset.image_directory, image_file_name)
                annotation_ids = dataset.annotations.getAnnIds(imgIds=image_id)
                annotations = dataset.annotations.loadAnns(annotation_ids)

                total_keypoints = sum([ann.get('num_keypoints', 0) for ann in annotations])
                if total_keypoints == 0:
                    continue

                persons_metadata = []
                previous_centers = []
                segmentation_masks = []
                joint_keypoints = []

                # sort from largest to smallest person
                person_ids_sorted = np.argsort([-a['area'] for a in annotations], kind='mergesort')

                for person_id in list(person_ids_sorted):
                    person_data = annotations[person_id]

                    if person_data["iscrowd"]:
                        segmentation_masks.append(dataset.annotations.annToRLE(person_data))
                        continue

                    # skip if keypoint count is too low or area is too small
                    if person_data["num_keypoints"] < 5 or person_data["area"] < 32 * 32:
                        segmentation_masks.append(dataset.annotations.annToRLE(person_data))
                        continue

                    person_center = [person_data["bbox"][0] + person_data["bbox"][2] / 2,
                                     person_data["bbox"][1] + person_data["bbox"][3] / 2]

                    # skip if too close to another person
                    if any(cdist(np.expand_dims(pc[:2], axis=0), np.expand_dims(person_center, axis=0))[0] < pc[2]*0.3 for pc in previous_centers):
                        segmentation_masks.append(dataset.annotations.annToRLE(person_data))
                        continue

                    person_metadata = TrainingDataPoint(
                        image_path=image_path,
                        image_height=height,
                        image_width=width,
                        image_center=np.expand_dims(person_center, axis=0),
                        bounding_box=person_data["bbox"],
                        image_area=person_data["area"],
                        scaling_factor=person_data["bbox"][3] / self.desired_size[0],
                        keypoints_count=person_data["num_keypoints"])

                    joint_keypoints.append(person_data["keypoints"])
                    persons_metadata.append(person_metadata)
                    previous_centers.append(np.append(person_center, max(person_data["bbox"][2], person_data["bbox"][3])))

                if persons_metadata:
                    main_person = persons_metadata[0]
                    main_person.segmentation_masks = segmentation_masks
                    main_person.original_joints = KeypointLoader.load_from_coco_keypoints(joint_keypoints, width, height)
                    self.metadata_list.append(main_person)

                if i % 1000 == 0:
                    print("Loading image annotation {}/{}".format(i, len(image_ids)))

    def save(self, path):
        raise NotImplemented

    def load(self, path):
        raise NotImplemented

    def size(self):
        """
        :return: number of items
        """
        return len(self.metadata_list)

    def get_data(self):
        """
        Generator of data points

        :return: instance of TrainingDataPoint
        """
        indices = np.arange(self.size())
        self.rng.shuffle(indices)
        for index in indices:
            yield [self.metadata_list[index]]
