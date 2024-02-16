# Setup

We need the following packages:

* deeplabcut[gui, tf]
* deeplabcut-live
* roboticstoolbox-python
* differentiable-robot-model (not on pip)
    * https://github.com/facebookresearch/differentiable-robot-model  
    * Install directions for this are provided in the repo. But basically have to run setup.py, be sure to use the conda envirionment that is being used for everything else

Additionally, install CoppeliaSim for the ttt file. This is a simulation where the urdf can be imported and then the json joint_angles used to move the robot.

# Processing Pipeline (Python notebooks)

### morph_robot_model_blender.ipynb
First step. Takes a URDF file and matches it up to a set of keypoints from Blender to make the robot more in the shape of the animal in question based on the skeleton
(optional)

### parameterized_optimization_with_head.ipynb
Trains a URDF file with the video from HorseInferenceFiles and using the DLC model in HorseInferenceFiles in order to get the horse with its joints properly positioned. Returns the robot as a urdf file. Depends on the optimization_library code-base.

### prepare_robot_for_sim.ipynb
Takes the trained robot from the previous step and makes it ready to be imported into CoppeliaSim. This means the robot is going to be scaled down and it will have its links have cylinders for visualization purposes.

## Coppeliasim
Import the robot urdf at tools > importers > urdf. Dont change any settings.
In the tree of the robot mark every link ending in _respondable with dynamic to make it movable.

## visualize_robot.ipynb
Visualizes a robot

# Robot Files

### a1.urdf
Initial urdf comes from (https://github.com/unitreerobotics/unitree_ros/blob/master/robots/a1_description/urdf/a1.urdf)

## cleaned_a1.urdf
Removed a lot of material tags amongst other things by hand to lcean the robot file for training.

## horse_like_robot_model.urdf
cleaned_a1 with added joints/links for the head and tail. So that the robot looks more like a horse.

## horse_based_robot.urdf
For comparison purposes. This is based on horse_like_robot_model.urdf, where the joint positioned are adjusted to match blender points, placed on a horse skeleton. See ./HorseInferenceFiles/HorseSkeletonPointExtraction.blend for the blender file.

## generated/generatedURDF.urdf

This is the urdf of the robot after it has been matched to the image

Generated urdf from the parameterized_optimization_with_head.ipynb notebook. 

## generated/sim_ready_robot.urdf

This is the urdf of the robot after it has been adjusted to be ready to be imported into coppeliasim.

Generated from prepare_robot_for_sim.ipynb

# Misc. Training Artifacts

# generated/videos

final_video is the total video with the robot horse and the training vidoe
training_video is the first frame as the training occurs for debugging purposes.

# generated/joint_angles.json

Joint angles by frame for the final video.

# Deeplabcut Models

## HorseInferenceFiles/AnotherDetectionproject-Aman-2023-10-11

Deeplabcut project used to get the DLC_Model_With_Head. 

For use modify: HorseInferenceFiles/AnotherDetectionproject-Aman-2023-10-11/config.yaml specifically: project_path and video_sets

# HorseInferenceFiles/DLC_HorseProject1_efficientnet-b0_iteration-0_shuffle-1

Old exported deeplabcut model that doesnt include the horse's head/tail joints.

## HorseInferenceFiles/DLC_Model_With_Head

New exported deeplabcut model that includes the horse's head/tail joints

## HorseInferenceFiles/trainset/trimmedHorseVideo.mp4

Training video, video of the horse moving trimmer for the relevant portion.

## HorseInferenceFiles/Horse Labelled.png

Horse with each of the points labelled.

## HorseInferenceFiles/HorseSkeletonPointExtraction.blend

Horse Skeleton in belnder along with points for the joints to line up the robot to for validation purposes. 
Easy way to extract the point coordinates is provided in the morph_robot_model_blender.ipynb notebook.

# Robot Simulator

## robot_with_position_control.ttt

A coppeliasim scene that has a robot along with a script that control it. The joint angles are from the json file that is exported.
In order to get a future robot to work as this one does, you need to make all the respondable links as dynamic bodies, this is done by a checkbox.

## static_body_horse_model.ttt
A coppeliasim scene that has a robot imported from a urdf.
