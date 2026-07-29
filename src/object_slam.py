# python imports
import threading
import rclpy.logging
import tf2_ros
import rclpy
import cv_bridge
from nav_msgs.msg import Odometry
from message_filters import  Subscriber
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseWithCovarianceStamped
from message_filters import ApproximateTimeSynchronizer

from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from message_filters import ApproximateTimeSynchronizer
import numpy as np
import pickle
from ultralytics import YOLO
import supervision as sv
from scipy.interpolate import interp1d

from ament_index_python.packages import get_package_share_directory
import os
import time


# MMMR imports
import sys

from scene_graph import search_for_loop_closure_scene_graph, transform_points_2D
from scene_graph import merge_robot_with_prior_list, proccess_image, draw_boxes, load_prior, proccess_detections


# sys.path.append('/home/jake/Desktop/MMMR/')
# from data_utils import load_data
# from robot_class import Robot

# bruce imports
# from bruce_slam.utils.io import *
from conversions import *
from visualization import *
from topics import *
from slam import SLAM, Keyframe
import pcl
import sensor_msgs_py.point_cloud2 as pointcloud2_to_numpy

import matplotlib.pyplot as plt
import gtsam
from tf2_ros import TransformListener, Buffer
from scipy.spatial.transform import Rotation
from rclpy.time import Time
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


# Argonaut imports
from sonar_oculus.msg import OculusPing
from starfish_ros.msg import DeadReckoning
from starfish_ros.msg import Gps


class ObjectSLAM(Node):
    """This class takes the functionality from slam.py and implements it in the ros
    environment. 
    """
    
    def __init__(self):
        # super(Node,self).__init__()
        super().__init__('object_slam_node')
        # Node.__init__(self, 'object_slam_node')

        # the threading lock
        # self.lock = threading.RLock()

        # self.node = rclpy.create_node('random_node_slam')

    def get_node_ready(self, ns="~")->None:
        """Configures the SLAM node

        Args:
            ns (str, optional): The namespace of the node. Defaults to "~".
        """

        # keyframe paramters, how often to add them
        self.declare_parameter('keyframe_translation', 3.0)
        self.declare_parameter('keyframe_rotation', np.radians(30.0))
        self.keyframe_translation = self.get_parameter('keyframe_translation').get_parameter_value().double_value
        self.keyframe_rotation = self.get_parameter('keyframe_rotation').get_parameter_value().double_value
        self.point_resolution = 0.5

        
        # if we want to scrape for LiDAR data
        self.use_lidar = False

        # make delay between an incoming point cloud and dead reckoning
        self.feature_odom_sync_max_delay = 0.5

        # define the subsrcibing topics
        self.feature_sub = Subscriber(self, PointCloud2, SONAR_FEATURE_TOPIC)
        self.odom_sub = Subscriber(self, DeadReckoning, LOCALIZATION_ODOM_TOPIC)
        self.sonar_img_sub = Subscriber(self, OculusPing, SONAR_TOPIC)
        # self.odom_rapid_sub = self.create_subscription(DeadReckoning, LOCALIZATION_ODOM_TOPIC, self.odom_callback, 200)
        self.prior_pose_sub = self.create_subscription(Float32MultiArray, "/poses", self.prior_poses_callback, 10)  
        self.prior_boxes_sub = self.create_subscription(Float32MultiArray, "/boxes", self.prior_boxes_callback, 10)  

        # time sync
        self.time_sync = ApproximateTimeSynchronizer([self.feature_sub, self.odom_sub, self.sonar_img_sub], 
                                                        20, 
                                                        self.feature_odom_sync_max_delay, 
                                                        allow_headerless = False)

        # register the callback in the sync policy
        self.time_sync.registerCallback(self.SLAM_callback)

            
        #pose publisher
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped,SLAM_POSE_TOPIC,10)

        #dead reckoning topic
        self.odom_pub = self.create_publisher(Odometry,SLAM_ODOM_TOPIC,10)

        #SLAM trajectory topic
        self.traj_pub = self.create_publisher(PointCloud2,SLAM_TRAJ_TOPIC,10)

        #prior poses topic
        self.prior_pose_pub = self.create_publisher(PointCloud2,SLAM_TRAJ_TOPIC + "/prior",10)
        
        #constraints between poses
        self.constraint_pub = self.create_publisher(Marker,SLAM_CONSTRAINT_TOPIC,10)

        #point cloud publisher topic
        self.cloud_pub = self.create_publisher(PointCloud2,SLAM_CLOUD_TOPIC,10)

        #yolo image pub
        self.image_pub = self.create_publisher(Image,"yolo_image",10)
        
        #tf broadcaster to show pose
        # self.tf = tf2_ros.TransformBroadcaster(self.node)

        # tf listener for dead reckoning information
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        #cv bridge object
        self.CVbridge = cv_bridge.CvBridge()

        self.frame_count = 0

        self.bridge = CvBridge()

        # are we doing object SLAM? 
        self.object_slam = True

        # do all required config for object SLAM
        if self.object_slam:

            pkg_share = get_package_share_directory("boat_slam")

            # load the yolo model
            self.model = YOLO(os.path.join(pkg_share, "config", "best.pt"))
            self.yolo_conf = 0.55

            # TODO, move this to this repo
            self.prior_sensor_map_x = np.load(os.path.join(pkg_share, "config", "hopper_map_x.npy"))
            self.prior_sensor_map_y = np.load(os.path.join(pkg_share, "config", "hopper_map_y.npy"))

            # Define class names (update with actual class names from your model)
            self.class_names = {0: "Edge", 1: "Object", 2: "Ship"} #ENSURE THESE ARE ACCURATE

            # Initialize annotators
            self.color_list = ['#000000', '#ff0000', '#00ff00']
            self.oriented_box_annotator = sv.OrientedBoxAnnotator(color = sv.ColorPalette.from_hex(self.color_list))

            self.label_annotator = sv.LabelAnnotator(
                                                color = sv.ColorPalette.from_hex(self.color_list),
                                                text_scale=.4, 
                                                text_padding=0,
                                                text_position=sv.Position.TOP_CENTER
                                                )
            
            # we need a publisher for the yolo image
            self.publisher_image = self.create_publisher(Image,"sonar_yolo_image",10)

            self.object_loop_closures = []

            self.rov_2_feature_vectors = []
            self.rov_2_angles = []
            self.rov_2_positions = []
            self.rov_2_annotated_imgs = []
            self.rov_2_detections = []
            self.rov_2_poses = []

            self.dead_reckoning_poses = None

            
            '''prior_data = np.load(os.path.join(pkg_share, "config", "hopper_prior_boxes.npy"))
            prior_data = load_prior(prior_data)

            prior_poses = np.load(os.path.join(pkg_share, "config", "hopper_prior_poses.npy"))

            self.rov_2_poses = []
            for row in prior_poses:
                self.rov_2_poses.append(gtsam.Pose2(row[0], row[1], row[2]))
            

            for row in prior_data:
                (detections, 
                intersections, 
                feature_vector, 
                angles, 
                positions) = proccess_detections(row,
                                                np.zeros((490, 256)),
                                                30.0)
                
                # log the results for yolo
                self.rov_2_feature_vectors.append(feature_vector)
                self.rov_2_angles.append(angles)
                self.rov_2_positions.append(positions)
                
                # self.rov_2_annotated_imgs.append(annotated_img)
                self.rov_2_detections.append(detections)'''

        # some slam data structures
        self.old_pose = None
        self.step = 0
        self.pose_list = []
        self.dr_poses = []
        self.loop_closures = []
        self.points = []
        self.prior_poses = []
        self.live_sensor_map_x = None
        self.live_sensor_map_y = None

        # some sonar params
        self.res = None
        self.height = None
        self.rows = None
        self.width = None
        self.cols = None

        print("SLAM node is initialized")

    def prior_poses_callback(self, msg: Float32MultiArray) -> None:
        """Handle an incoming prior pose message

        Args:
            msg (Float32MultiArray): an array of 3D poses
        """

        if len(self.rov_2_poses) != 0:
            return

        self.rov_2_poses = []
        arr = np.array(msg.data).reshape(-1,3)
        for row in arr:
                self.rov_2_poses.append(gtsam.Pose2(row[0], row[1], row[2]))

        print("poses recived")

    def prior_boxes_callback(self, msg: Float32MultiArray) -> None:

        if len(self.rov_2_feature_vectors) != 0:
            return

        self.rov_2_feature_vectors = []
        self.rov_2_angles = []
        self.rov_2_positions = []
        self.rov_2_annotated_imgs = []
        self.rov_2_detections = []

        prior_data = load_prior(msg.data)

        for row in prior_data:
            (detections, 
            intersections, 
            feature_vector, 
            angles, 
            positions) = proccess_detections(row,
                                            np.zeros((490, 256)),
                                            30.0)
            
            # log the results for yolo
            self.rov_2_feature_vectors.append(feature_vector)
            self.rov_2_angles.append(angles)
            self.rov_2_positions.append(positions)
            
            # self.rov_2_annotated_imgs.append(annotated_img)
            self.rov_2_detections.append(detections)

        print("boxes recdioved")

        
    def add_frame(self, feature_msg:PointCloud2, current_pose:gtsam.Pose2, sonar_msg:OculusPing) -> None:

        # parse the sonar image
        sonar_img = self.CVbridge.compressed_imgmsg_to_cv2(sonar_msg.ping)
        sonar_img = cv2.flip(sonar_img, 1)

        # parse and update the points
        points_arr = np.array(list(pointcloud2_to_numpy.read_points_numpy(feature_msg, field_names=("x", "y", "z"), skip_nans=True)))
        points_arr =  np.c_[points_arr[:,0] , points_arr[:,1]]
        self.points.append(points_arr)

        # update old the old pose
        self.old_pose = current_pose
        self.dr_poses.append(current_pose)
        dr_poses_temp = self.set_zero_reference_frame(self.dr_poses)

        # update the sensor mapping
        self.generate_map_xy(sonar_msg)

        return dr_poses_temp, sonar_img



    def SLAM_callback(self, feature_msg:PointCloud2, odom_msg:DeadReckoning, sonar_msg:OculusPing)->None:

        # check if we have ever had a pose before
        current_pose = gtsam.Pose2(odom_msg.x,odom_msg.y,odom_msg.yaw)
        if self.old_pose is None:
            dr_poses_temp, sonar_img = self.add_frame(feature_msg, current_pose, sonar_msg)

        # get the difference
        position_difference = np.sqrt((self.old_pose.x() - current_pose.x())**2 + (self.old_pose.y() - current_pose.y())**2)
        yaw_difference = abs(self.old_pose.theta() - current_pose.theta())

        # if the diff is big, we have a keyframe
        if position_difference > 3 or yaw_difference >  0.5235987756:
            print(self.step)

            self.step += 1
            dr_poses_temp, sonar_img = self.add_frame(feature_msg, current_pose, sonar_msg)

            # make sure we actually have a prior
            if len(self.rov_2_poses) > 0 and len(self.rov_2_detections) > 0:

                # define the calibration between DR and sonar
                calib = gtsam.Pose2(0,0,np.radians(90.))

                # call the search method using scene graphs
                start_time = time.time()
                loop, annotated_img_source = search_for_loop_closure_scene_graph(self.step,
                                                                                    sonar_img, 
                                                                                    self.rov_2_detections,
                                                                                    self.rov_2_feature_vectors, 
                                                                                    self.rov_2_angles, 
                                                                                    self.rov_2_positions, 
                                                                                    self.live_sensor_map_x,
                                                                                    self.live_sensor_map_y,
                                                                                    self.prior_sensor_map_x,
                                                                                    self.prior_sensor_map_y,
                                                                                    self.model,
                                                                                    self.yolo_conf,
                                                                                    self.class_names, 
                                                                                    self.oriented_box_annotator, 
                                                                                    self.label_annotator)
                total_time = time.time() - start_time
                with open("fucnction_time.txt", "a") as file:
                    file.write(f"{total_time}\n")

                # log only if we have a valid loop
                if loop is not None:
                    self.loop_closures.append(loop)

            # if there are any loop closures, merge the robots using a pose graph
            if len(self.loop_closures) > 0: 
                result = merge_robot_with_prior_list(self.loop_closures, self.rov_2_poses, dr_poses_temp, calib) # merge graphs
                results = [] 
                results_prior = []
                for i in range(len(self.dr_poses)): # convert ROV_3 poses into a list
                    pose = result.atPose2(gtsam.symbol("x",i))
                    results.append(pose)
                    
                for i in range(len(self.rov_2_poses)): # convert ROV_2 poses into a list
                    pose = result.atPose2(gtsam.symbol("p",i))
                    results_prior.append(pose) 

                # save the results and zero their reference frames
                self.prior_poses = self.set_zero_reference_frame(results_prior, results[0])
                self.pose_list = self.set_zero_reference_frame(results)

            else: 
                self.pose_list = dr_poses_temp


            self.dead_reckoning_poses = list(dr_poses_temp)

            self.publish_trajectory()
            self.publish_constraint()
            self.publish_point_cloud()
            if len(self.rov_2_poses) > 0 and len(self.rov_2_detections) > 0:
                self.publish_image(annotated_img_source)


        
        if self.dead_reckoning_poses is not None:
            dr_poses_temp = list(self.dr_poses)
            dr_poses_temp.append(current_pose)
            dr_poses_temp = self.set_zero_reference_frame(dr_poses_temp)

            temp_between = dr_poses_temp[-2].between(dr_poses_temp[-1])
            temp_pose = self.pose_list[-1].compose(temp_between)


            #temp_pose = self.pose_list[-1].compose(self.dead_reckoning_poses[-1].between(current_pose))
            temp_pose = pose223(temp_pose)
            self.publish_pose(temp_pose,sonar_msg.header.stamp)


                #results_prior = set_zero_reference_frame(results_prior)
                #results = set_zero_reference_frame(results) # zero the reference frame for generating results

    def publish_image(self, image) -> None:
        """Publish the yolo image

        Args:
            image (np.array): the marked up yolo sonar image
        """

        msg_out = self.bridge.cv2_to_imgmsg(image)
        self.image_pub.publish(msg_out)

    def publish_trajectory(self)->None:
        """Publish 3D trajectory as point cloud in [x, y, z, roll, pitch, yaw, index] format.
        """

        
        poses = []
        for pose in self.pose_list:
            pose3 = gtsam.Pose3(gtsam.Rot3.Yaw(pose.theta()), gtsam.Point3(pose.x(), pose.y(), 0))
            poses.append(g2n(pose3))

        #convert to a ros color line
        traj_msg = ros_colorline_trajectory(poses)
        # traj_msg.header.stamp = self.current_keyframe.time
        traj_msg.header.frame_id = "map"
        self.traj_pub.publish(traj_msg)


        if len(self.prior_poses) > 0:
            poses = []
            for pose in self.prior_poses:
                pose3 = gtsam.Pose3(gtsam.Rot3.Yaw(pose.theta()), gtsam.Point3(pose.x(), pose.y(), 0))
                poses.append(g2n(pose3))

            #convert to a ros color line
            traj_msg = ros_colorline_trajectory(poses)
            # traj_msg.header.stamp = self.current_keyframe.time
            traj_msg.header.frame_id = "map"
            self.prior_pose_pub.publish(traj_msg)
        

    def publish_constraint(self)->None:
        """Publish constraints between poses in the factor graph,
        either sequential or non-sequential.
        """

        # fix the poses, they are pose2 right now and need to be pose3
        poses = []
        for pose in self.pose_list:
            pose3 = gtsam.Pose3(gtsam.Rot3.Yaw(pose.theta()), gtsam.Point3(pose.x(), pose.y(), 0))
            poses.append(pose3)
        
        #define a list of all the constraints
        links = []

        #iterate over all the keframes
        for x, kf in enumerate(poses[1:], 1):

            #append each SSM factor in green
            p1 = poses[x - 1].x(), poses[x - 1].y(), poses[x - 1].z()
            p2 = poses[x].x(), poses[x].y(), poses[x].z()
            links.append((p1, p2, "green"))

            #loop over all loop closures in this keyframe and append them in red
            '''for k, _ in self.pose_list[x].constraints:
                p0 = self.pose_list[k].pose3.x(), self.pose_list[k].pose3.y(), self.pose_list[k].dr_pose3.z()
                links.append((p0, p2, "red"))'''
            
        # loop over all the loop closures
        if len(self.loop_closures) > 0:
            for i, j, _ in self.loop_closures: 
                p1 = poses[i].x(), poses[i].y(), poses[i].z()
                p2 = self.prior_poses[j].x(), self.prior_poses[j].y(), 0.0
                links.append((p1, p2, "red"))


        #if nothing, do nothing
        if links:

            #conver this list to a series of multi-colored lines and publish
            link_msg = ros_constraints(links)
            # link_msg.header.stamp = self.current_keyframe.time
            link_msg.header.frame_id = "map"
            self.constraint_pub.publish(link_msg)

    
    def publish_point_cloud(self)->None:
        """Publish downsampled 3D point cloud with z = 0.
        The last column represents keyframe index at which the point is observed.
        """

        #define an empty array
        all_points = [np.zeros((0, 2), np.float32)]

        #list of keyframe ids
        all_keys = []

        #loop over all the keyframes, register 
        #the point cloud to the orign based on the SLAM estinmate
        for key in range(self.step):

            #get the resgistered point cloud
            transf_points = transform_points_2D(self.points[key], self.pose_list[key])

            #append
            all_points.append(transf_points)
            all_keys.append(key * np.ones((len(transf_points), 1)))

        all_points = np.concatenate(all_points)
        all_keys = np.concatenate(all_keys)

        #use PCL to downsample this point cloud
        '''sampled_points, sampled_keys = pcl.downsample(
            all_points, all_keys, self.point_resolution
        )'''

        #parse the downsampled cloud into the ros xyzi format
        sampled_xyzi = np.c_[all_points, np.zeros_like(all_keys), all_keys]
        
        #if there are no points return and do nothing
        if len(sampled_xyzi) == 0:
            return

        #convert the point cloud to a ros message and publish
        cloud_msg = n2r(sampled_xyzi, "PointCloudXYZI")
        # cloud_msg.header.stamp = self.current_keyframe.time
        cloud_msg.header.frame_id = "map"
        self.cloud_pub.publish(cloud_msg)

    def set_zero_reference_frame(self, poses:list, ref_frame:gtsam.Pose2 = None) -> list:

        if ref_frame is None:
            ref_frame = poses[0]

        new_poses = []
        for pose_i in poses:
            new_poses.append(ref_frame.between(pose_i))

        return new_poses
    
    def generate_map_xy(self, ping: OculusPing) -> None:
        """Generate a mesh grid map for the sonar image, this enables converison to cartisian from the 
        source polar images

        Args:
            ping (OculusPing): incoming sonar message
        """


        to_rad = lambda bearing: bearing * np.pi / 18000

        # get the parameters from the ping message
        _res = ping.range_resolution
        _height = ping.num_ranges * _res
        _rows = ping.num_ranges
        _width = np.sin(
            to_rad(ping.bearings[-1] - ping.bearings[0]) / 2) * _height * 2
        _cols = int(np.ceil(_width / _res))

        # check if the parameters have changed
        if self.res == _res and self.height == _height and self.rows == _rows and self.width == _width and self.cols == _cols:
            return

        # if they have changed do some work    
        self.res, self.height, self.rows, self.width, self.cols = _res, _height, _rows, _width, _cols

        # generate the mapping
        
        bearings = to_rad(np.asarray(ping.bearings, dtype=np.float32))
        f_bearings = interp1d(
            bearings,
            range(len(bearings)),
            kind='linear',
            bounds_error=False,
            fill_value=-1,
            assume_sorted=True)
        
        # build the meshgrid
        XX, YY = np.meshgrid(range(self.cols), range(self.rows))
        x = self.res * (self.rows - YY)
        y = self.res * (-self.cols / 2.0 + XX + 0.5)
        b = np.arctan2(y, x)
        r = np.sqrt(np.square(x) + np.square(y))
        self.live_sensor_map_y = np.asarray(r / self.res, dtype=np.float32)
        self.live_sensor_map_x = np.asarray(f_bearings(b), dtype=np.float32)

    def publish_pose(self, pose_temp, time_stamp)->None:
        """Append dead reckoning from Localization to SLAM estimate to achieve realtime TF.
        """

        #define a pose with covariance message 
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = time_stamp
        pose_msg.header.frame_id = "map"
        pose_msg.pose.pose = g2r(pose_temp)

        #cov = 1e-4 * np.identity(6, np.float32)
        # FIXME Use cov in current_frame
        #0cov[np.ix_((0, 1, 5), (0, 1, 5))] = self.current_keyframe.transf_cov
        #pose_msg.pose.covariance = cov.ravel().tolist()
        self.pose_pub.publish(pose_msg)

        '''o2m = self.current_frame.pose3.compose(self.current_frame.dr_pose3.inverse())
        o2m = g2r(o2m)
        p = o2m.position
        q = o2m.orientation

        transform = TransformStamped()
        transform.header.stamp = self.current_frame.time
        transform.header.frame_id = 'map'
        transform.child_frame_id = 'slam_link'
        transform.transform.translation.x = pose_msg.pose.pose.position.x
        transform.transform.translation.y = pose_msg.pose.pose.position.y
        transform.transform.translation.z = 0.0
        transform.transform.rotation.x = pose_msg.pose.pose.orientation.x
        transform.transform.rotation.y = pose_msg.pose.pose.orientation.y
        transform.transform.rotation.z = pose_msg.pose.pose.orientation.z
        transform.transform.rotation.w = pose_msg.pose.pose.orientation.w
        self.tf.sendTransform(transform)

        odom_msg = Odometry()
        odom_msg.header = pose_msg.header
        odom_msg.pose.pose = pose_msg.pose.pose
        if self.rov_id == "":
            odom_msg.child_frame_id = "base_link"
        else:
            odom_msg.child_frame_id = self.rov_id + "_base_link"
        # odom_msg.twist.twist = self.current_frame.twist
        self.odom_pub.publish(odom_msg)'''




