import torch
import logging
from differentiable_robot_model.rigid_body_params import (
    UnconstrainedTensor,
)
import cv2
from dlclive import DLCLive, Processor
import yaml
import shutil
import xml
import xml.etree.ElementTree as ET
from differentiable_robot_model.robot_model import (
    DifferentiableRobotModel,
)

def constantTensor(constant):
    return torch.tensor(float(constant))

def differentiableConstantTensor(constant):
    return constantTensor(constant).requires_grad_(True)


def getTorchDevice(USE_GPU = False):
    device = torch.device("cpu")
    try:
        if USE_GPU and torch.cuda.is_available():
            device = torch.device('cuda')
        elif USE_GPU and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device('cpu')
    except:
        if USE_GPU and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    return device
class NMSELoss(torch.nn.Module):
    def __init__(self, var):
        super(NMSELoss, self).__init__()
        self.var = var

    def forward(self, yp, yt):
        err = (yp - yt) ** 2
        werr = err / self.var
        return werr.mean()

class ConstrainedTensor(torch.nn.Module):
    def __init__(self, dim1, dim2, init_tensor=None, init_std=0.1, min_val=0.0, max_val=1.0):
        super().__init__()
        self._dim1 = dim1
        self._dim2 = dim2
        if init_tensor is None:
            init_tensor = torch.empty(dim1, dim2).normal_(mean=0.0, std=init_std)
        self.param = torch.nn.Parameter(init_tensor)
        self.min_val = min_val
        self.max_val = max_val

    def forward(self):
        param = self.param
        param = torch.clamp(param, min=self.min_val, max=self.max_val)
        return param


def print_tree_connections(links, base_name = "base"):
    tree = {}
    for parent, child in links:
        if parent not in tree:
            tree[parent] = []
        tree[parent].append(child)

    def print_tree(node, prefix=''):
        print(prefix + node)
        if node in tree:
            for child in tree[node]:
                print_tree(child, prefix + '  ')

    print_tree(base_name)


def getLinksLength(learnable_robot_model, linkName):
    linkObject = learnable_robot_model._get_parent_object_of_param(linkName, "trans")
    return linkObject.trans().detach().clone()

def makeLinkLengthLearnable(learnable_robot_model, linkName, value = None):
    linkObject = learnable_robot_model._get_parent_object_of_param(linkName, "trans")
    learnable_robot_model.make_link_param_learnable(
        linkName, 
        "trans", 
        UnconstrainedTensor(dim1 = 1, dim2 = 3, init_tensor = value)
    )
    logging.debug(f"Made {linkName} link's length learnable initialized tensor to {value}")

class CV2VideoReader():
    def __init__(self, video_path, frames_to_pull = None):
        """
        video_path: path to video file
        frames_to_pull: list of frames to pull from video. If None, all frames are pulled. Used for large videos.
        """
        self.video_path = video_path
        self.frames = []
        cap = cv2.VideoCapture(str(self.video_path))
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if frames_to_pull is None:
            frames_to_pull = range(self.frame_count)
        
        for frame_number in frames_to_pull:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                break
            self.frames.append(frame)
        cap.release()
        self.frame_count = len(self.frames)

    @classmethod
    def get_frame_count(cls, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_count
    def get_subset(self, frame_indices):
        #TODO: Convert this to some kind of slicing object pass through?
        return [self.frames[i] for i in frame_indices]
    
class TrainingKeypoints():
    def __init__(self, keypoints, images, keypoint_names_ordering):
        self.keypoints = keypoints
        self.images = images
        self.keypoint_names_ordering = keypoint_names_ordering
    def get_frame_count(self):
        return len(self.images)
    def get_subset(self, indices):
        indices = list(indices)
        return TrainingKeypoints(self.keypoints[indices], [self.images[i] for i in indices], self.keypoint_names_ordering)
    def get_video_dimensions(self):
        return self.images[0].shape[0:2]
    @classmethod
    def init_from_video_using_dlc_model(cls, videoReader: CV2VideoReader, dlc_model_path, frame_indices = None):
        videoReader = videoReader
        if frame_indices is None:
            frame_indices = range(videoReader.frame_count)
        frames = videoReader.get_subset(frame_indices)
        keypoints = []
        dlc_proc = Processor()
        dlc_live = DLCLive(model_path=str(dlc_model_path), processor=dlc_proc)

        for index, frame in enumerate(frames):
            frame_keypoints = []
            if index == 0:
                frame_keypoints = dlc_live.init_inference(frame)
            else:
                frame_keypoints = dlc_live.get_pose(frame)
            keypoints.append(torch.from_numpy(frame_keypoints[:,:2]).unsqueeze(0))
        keypoints_in_order = None
        with open(dlc_model_path/"pose_cfg.yaml", 'r') as file:
            # Load the YAML data into a Python dictionary
            keypoints_in_order = yaml.safe_load(file)['all_joints_names']
        return cls(torch.vstack(keypoints), frames, keypoints_in_order)
    def reordered_training_set(self, new_keypoint_ordering):
        """
            Returns a new Training Keypoint object, with the keypoints in a new ordering.
            new_keypoint_ordering is the desired order of the new keypoints, in the form of their names (NOT INDICES)
        """
        indexes_of_new_ordering = [self.keypoint_names_ordering.index(keypoint) for keypoint in new_keypoint_ordering]
        new_keypoints = self.keypoints[:,indexes_of_new_ordering,:]
        return TrainingKeypoints(new_keypoints, self.images, new_keypoint_ordering)
    

class RobotData():
    def __init__(self, robot_model, robot_urdf_path):
        self.robot_model = robot_model
        self.robot_urdf_path = robot_urdf_path
        self.initial_joint_angles = {joint: getLinksLength(robot_model,joint) for joint in robot_model.get_link_names()}
        self.make_limb_length_learnable()
    def make_limb_length_learnable(self, limb_lengths = None):
        if limb_lengths == None:
            limb_lengths = self.initial_joint_angles
        for joint in self.get_link_names():
            makeLinkLengthLearnable(self.robot_model,joint, self.initial_joint_angles[joint].detach().clone())
    @classmethod
    def init_from_urdf_file(cls, robot_urdf_path, device = None):
        robot_model = DifferentiableRobotModel(
            robot_urdf_path, "A1", device=device
        )
        return cls(robot_model, robot_urdf_path)
    def get_link_names(self):
        return self.robot_model.get_link_names()
    def write_to_urdf_file(self, exported_robot_path):
        shutil.copyfile(self.robot_urdf_path, exported_robot_path)
        tree = xml.etree.ElementTree.parse(exported_robot_path)
        root = tree.getroot()
        for joint in root.findall("joint"):
            name = joint.find("./child").get('link')
            desiredXYZ = getLinksLength(self.robot_model,name)
            currentXYZ = joint.find("origin").get("xyz")
            formattedDesiredXYZ = " ".join(map(str, desiredXYZ.reshape(3).tolist()))
            logging.debug(f"{name}")
            logging.debug(f"\tChanging XYZ from '{currentXYZ}' to '{formattedDesiredXYZ}'")
            joint.find("origin").set("xyz", formattedDesiredXYZ)
        tree.write(exported_robot_path)   
    def get_skeleton(self):
        tree = ET.parse(self.robot_urdf_path)
        root = tree.getroot()
        return[(joint.find("parent").get("link"),joint.find("child").get("link")) for joint in root.findall("./joint")]
    def print_skeleton(self):
        print_tree_connections(self.get_skeleton())     

#Utility functions
def homogenize_vectors(tensor, padding_size = 1):
  padded_tensor = torch.nn.functional.pad(
      tensor, pad=(0, padding_size), mode='constant', value=1)
  return padded_tensor

def dehomogenize_vector(tensor, padding_size=1):
  cropped_tensor = tensor[..., :-padding_size]
  return cropped_tensor
trainingFrames = []

def calculateCameraProjection(intrinsic, extrinsic, jointPositions):
    transform = intrinsic@extrinsic
    pixelPositions = transform@(homogenize_vectors(jointPositions).transpose(1,2))
    pixelPositions = pixelPositions.transpose(1,2)
    pixelPositions = dehomogenize_vector(pixelPositions)
    return pixelPositions

def drawPredictionOnImage(*,robotEEPositions, training_dataset: TrainingKeypoints, image_index: int, intrinsic, extrinsic, robot_data, output_to_console = False):
    img = (training_dataset.images[image_index]).copy()
    positionByJoint = torch.cat([v[image_index].unsqueeze(0) if len(v.shape) >= 2 else v.unsqueeze(0) for v in robotEEPositions.values()])
    #TODO: Note that position by joint used to also be repeated so maybe there's another unsqueeze needed?
    allJointPosiitons = calculateCameraProjection(intrinsic, extrinsic[image_index], positionByJoint.unsqueeze(0))[0]
    jointToCameraPosition = {key:allJointPosiitons[index] for index,key in enumerate(robotEEPositions)}
    size = 16
    #Draw Robot Links
    for line in robot_data.get_skeleton():
        start_name,end_name = line[0], line[1]
        start, end = jointToCameraPosition[start_name], jointToCameraPosition[end_name]
        start = tuple(start.clone().detach().numpy())
        end = tuple(end.clone().detach().numpy())
        start = tuple((int(i) for i in start))
        end = tuple((int(i) for i in end))
        X_MAX, Y_MAX = training_dataset.get_video_dimensions()
        inFrame = lambda pos: 0 <= pos[0] <= X_MAX and 0 <= pos[1] <= Y_MAX
        if output_to_console:
            out = ""
            if not inFrame(start):
                out += (f"Start point {start_name} is out of frame\n")
            if not inFrame(end):
                out += (f"End point {end_name} is out of frame\n")
            if out != "":
                print(out, "---")
        img = cv2.line(img,start, end, (0,0,255),size//4)
    #Draw Robot Joints
    for joint in jointToCameraPosition:
        jointPositionTensor = jointToCameraPosition[joint]
        color = (255,255,0)
        dotSize = size//2
        if joint == "base":
            color = (255,124,124)
            dotSize = int(dotSize*1.5)
        jointPosition = jointPositionTensor.clone().detach().numpy()
        x,y = tuple((int(i) for i in jointPosition))
        img = cv2.circle(img,(x, y),dotSize,color,-1)
    #Draw Every Optimized Keypoint  
    for x,y in training_dataset.keypoints[image_index]:
        color = (0,255,0)
        img = cv2.rectangle(img, (int(x), int(y)),(int(x)+size, int(y)+size), color, -1)
    return img
        
#Copied from https://pytorch3d.readthedocs.io/en/v0.6.0/_modules/pytorch3d/transforms/rotation_conversions.html#quaternion_to_matrix
class pytorch3d():
    class transforms:
        @staticmethod
        def quaternion_to_matrix(quaternions):
            """
            Convert rotations given as quaternions to rotation matrices.

            Args:
                quaternions: quaternions with real part first,
                    as tensor of shape (..., 4).

            Returns:
                Rotation matrices as tensor of shape (..., 3, 3).
            """
            r, i, j, k = torch.unbind(quaternions, -1)
            two_s = 2.0 / (quaternions * quaternions).sum(-1)

            o = torch.stack(
                (
                    1 - two_s * (j * j + k * k),
                    two_s * (i * j - k * r),
                    two_s * (i * k + j * r),
                    two_s * (i * j + k * r),
                    1 - two_s * (i * i + k * k),
                    two_s * (j * k - i * r),
                    two_s * (i * k - j * r),
                    two_s * (j * k + i * r),
                    1 - two_s * (i * i + j * j),
                ),
                -1,
            )
            return o.reshape(quaternions.shape[:-1] + (3, 3))

        