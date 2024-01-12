import torch
import optimization_library.utilities as utilities
def interframeJointLoss(jointAngles):
    """
        The L2 norm computed for every joint. Measures frame by frame the changes in joint angles, to minimze it
    """
    epsilon = 1.e-8
    return torch.sum(torch.sqrt(torch.sum(torch.pow(jointAngles[1:] - jointAngles[:-1],2), -1) + epsilon))

def totalLimbLength(robot_data ,relu = False, reluThreshold = 0.1):
    """
        Returns the length of all of the links of the robot
    """
    learnable_robot_model = robot_data.robot_model
    robotLinks = torch.vstack([learnable_robot_model._get_parent_object_of_param(joint, "trans").trans().detach().clone() for joint in learnable_robot_model.get_link_names()])
    distances = (torch.sqrt(torch.sum(torch.pow(robotLinks,2),-1)))
    if relu:
        distances = torch.nn.functional.relu(distances - reluThreshold)
    return torch.sum(distances)

def out_of_frame_loss(intrinsic,extrinsic, total_video,joint_positions_3d, scalar = 1000):
    """
        Penalizes the robot for going out of frame
        :args joint_positions_3d: Dictionary of Joint Name to 3D Position Tensor, output from differentiable robot model
        :args scalar: Scalar to multiply the loss by
    """
    robot_ee_positions = torch.hstack([positions_tensor.unsqueeze(1) for positions_tensor in joint_positions_3d.values()])
    robot_pixel_positions = utilities.calculateCameraProjection(intrinsic, extrinsic, robot_ee_positions)
    robot_pixel_positions = robot_pixel_positions.reshape(-1,2)
    return scalar*(sum(robot_pixel_positions[:,0] >= total_video.width)+sum(robot_pixel_positions[:,0] <= 0)+sum(robot_pixel_positions[:,1] >= total_video.height)+sum(robot_pixel_positions[:,1] <= 0))
    