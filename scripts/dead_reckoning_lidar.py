#!/usr/bin/env python3

import math
import rclpy
import gtsam
import tf2_ros
import numpy as np
from rclpy.node import Node
from rclpy.time import Time
from dvl_msgs.msg import DVL
from sensor_msgs.msg import Imu
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation

from starfish_ros.msg import DeadReckoning
from starfish_ros.msg import NortekMessage
from starfish_ros.msg import GyroMessage

from topics import *
from conversions import *


def quaternion_from_euler(ai, aj, ak):
    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    q = np.empty((4, ))
    q[0] = cj*sc - sj*cs
    q[1] = cj*ss + sj*cc
    q[2] = cj*cs - sj*sc
    q[3] = cj*cc + sj*ss

    return q

class DeadReckoningNode(Node):

    def __init__(self):
        super().__init__('dead_reckoning')
        self.node = rclpy.create_node('dynamic_transform_publisher')

        # Declare parameters with defaults
        self.declare_parameter('dvl_mode', 'nortek')
        self.declare_parameter('use_gyro', False)

        # Retrieve parameter values
        self.dvl_mode = self.get_parameter('dvl_mode').get_parameter_value().string_value
        self.use_gyro = self.get_parameter('use_gyro').get_parameter_value().bool_value

        # output the current set of params
        self.get_logger().info(f"DVL mode: {self.dvl_mode}")
        self.get_logger().info(f"Use gyro: {self.use_gyro}")

        # vectornav sub
        self.subscription_vn = self.create_subscription(Imu,'/vectornav/imu',self.vectornav_callback,200)

        # dvl sub
        if self.dvl_mode == "waterlinked":
            self.subscription_dvl = self.create_subscription(DVL,'/dvl/data',self.water_linked_dvl_callback,10)
        elif self.dvl_mode == "nortek":
            self.subscription_dvl = self.create_subscription(NortekMessage,'/nortek_dvl',self.nortek_dvl_callback,10)    

        # gyroscope sub
        if self.use_gyro:
            self.subscription_dvl = self.create_subscription(GyroMessage,'/gyro',self.gyro_callback,250)
        
        # pubs
        self.broadcaster = tf2_ros.TransformBroadcaster(self.node)
        self.publisher = self.node.create_publisher(PoseStamped, 'dvl_pose', 10)
        self.odom_publisher = self.node.create_publisher(DeadReckoning, LOCALIZATION_ODOM_TOPIC, 10)
        self.odom_lidar_publisher = self.node.create_publisher(DeadReckoning, LOCALIZATION_ODOM_TOPIC, 10)

        # tracker varibles
        self.prev_time = None # the old time
        self.pose = gtsam.Pose2(0,0,0) # location starting from zero
        self.vn_yaw_start = None # starting yaw location
        self.velocity = None # most recent velocity
        self.gyro_heading = 0.0 # the starting heading for the gyroscope

        self.transform = TransformStamped()
        self.pose_msg = PoseStamped()

        # Create a static transform
        static_tf_dvl_velodyne = TransformStamped()
        static_tf_dvl_velodyne.header.frame_id = 'sonar'
        static_tf_dvl_velodyne.child_frame_id = 'velodyne'

        # Set translation
        static_tf_dvl_velodyne.transform.translation.x = -0.18 # 0.200 #-0.545 # 0.0925
        static_tf_dvl_velodyne.transform.translation.y = -0.79 #-0.113
        static_tf_dvl_velodyne.transform.translation.z = 0.733

        # Set rotation (Quaternion)
        x,y,z,w = quaternion_from_euler(0,0,np.radians(-105))
        static_tf_dvl_velodyne.transform.rotation.x = x
        static_tf_dvl_velodyne.transform.rotation.y = y
        static_tf_dvl_velodyne.transform.rotation.z = z
        static_tf_dvl_velodyne.transform.rotation.w = w
        self.static_tf_dvl_velodyne = static_tf_dvl_velodyne

        # Create a static transform
        static_tf_dvl_sonar = TransformStamped()
        static_tf_dvl_sonar.header.frame_id = 'dvl'
        static_tf_dvl_sonar.child_frame_id = 'sonar'

        # Set translation
        static_tf_dvl_sonar.transform.translation.x = 0.893
        static_tf_dvl_sonar.transform.translation.y = 0.0
        static_tf_dvl_sonar.transform.translation.z = 0.0

        # Set rotation (Quaternion)
        x,y,z,w = quaternion_from_euler(0,0,np.radians(90))
        static_tf_dvl_sonar.transform.rotation.x = x
        static_tf_dvl_sonar.transform.rotation.y = y
        static_tf_dvl_sonar.transform.rotation.z = z
        static_tf_dvl_sonar.transform.rotation.w = w
        self.static_tf_dvl_sonar = static_tf_dvl_sonar

    def publish_static_transform(self, time_stamp:Header.stamp) -> None:
        """Publish the static transform between the DVL and LIDAR link. 

        Args:
            time_stamp (Header.stamp): the timestamp for the transform
        """

        self.static_tf_dvl_velodyne.header.stamp = time_stamp
        self.static_tf_dvl_sonar.header.stamp = time_stamp
        self.broadcaster.sendTransform(self.static_tf_dvl_velodyne)
        self.broadcaster.sendTransform(self.static_tf_dvl_sonar)

    def vectornav_callback(self, msg:Imu) -> None:
        """Handle an incoming vectornav imu message. Here we use IMU data and the cached dvl velocity
        to update the position estimate for the vehicle. 

        Args:
            msg (Imu): the incoming imu message
        """

        # initilize the heading to zero
        if self.vn_yaw_start is None:
            self.vn_yaw_start = gtsam.Rot3(msg.orientation.x,msg.orientation.y,msg.orientation.z,msg.orientation.w).roll()

        # we need to be able to find delta time
        if self.prev_time is None:
            self.prev_time = Time.from_msg(msg.header.stamp).nanoseconds
            return
        
        # we can't propagate forward without a velocity 
        if self.velocity is None:
            return
        
        scipy_rot = Rotation.from_quat([msg.orientation.x,msg.orientation.y,msg.orientation.z,msg.orientation.w])
        roll,pitch,yaw = scipy_rot.as_euler('xyz', degrees=False)
        yaw = yaw - self.vn_yaw_start

        if self.use_gyro:
            yaw = self.gyro_heading

        '''# package the IMU yaw
        temp = gtsam.Pose3(gtsam.Rot3.Ypr(yaw, 0, 0), gtsam.Point3(0, 0, 0)) # IMU

        # add the transform from the IMU to the DVL
        temp = temp.compose(gtsam.Pose3(gtsam.Rot3.Ypr(0, 0, 0), gtsam.Point3(25.095, -739.56, -340.831)))

        print(np.degrees(temp.rotation().yaw()), np.degrees(yaw))'''

        # set the roll pitch yaw order and determine if we are using the gyroscope
        # nortek and no gyro
        if self.dvl_mode == "nortek": #and self.use_gyro == False:
            x,y,z,w = Rotation.from_euler("xyz",[0,0,yaw],degrees=False).as_quat()
        # waterlinked and no gyro
        elif self.dvl_mode == "waterlinked": # and self.use_gyro == False:
            x,y,z,w = Rotation.from_euler("xyz",[roll,pitch,yaw],degrees=False).as_quat()

        # figure out how much time has passed
        dt = (Time.from_msg(msg.header.stamp).nanoseconds - self.prev_time) * 1e-9

        # figure out how far we moved in the body frame using the DVL message
        translation = self.velocity * dt

        # find how far we moved in the global frame
        local_point = gtsam.Point2(translation[0], translation[1])
        point = self.pose.transformFrom(local_point)
        self.pose = gtsam.Pose2(point[0],point[1],yaw)
    
        # log old time
        self.prev_time = Time.from_msg(msg.header.stamp).nanoseconds

        # define a transform message with the same timestamp as the imu
        self.transform.header.stamp = msg.header.stamp
        self.transform.header.frame_id = 'map'
        self.transform.child_frame_id = 'dvl'
        self.transform.transform.translation.x = self.pose.x()
        self.transform.transform.translation.y = self.pose.y()  
        self.transform.transform.translation.z = 0.0
        self.transform.transform.rotation.x = x
        self.transform.transform.rotation.y = y
        self.transform.transform.rotation.z = z
        self.transform.transform.rotation.w = w

        '''pose_temp = gtsam.Pose3(gtsam.Rot3(x,y,z,w),[self.pose.x(),self.pose.y(),0])
        pose_temp = pose_temp.compose(gtsam.Pose3(gtsam.Rot3.Ypr(np.radians(-15.),0,0),[-0.545,-0.113,0.733]))

        sonar_pose_dr = self.pose.compose(gtsam.Pose2(0.893,0,np.radians(90.)))'''
        sonar_pose_dr = self.pose.compose(gtsam.Pose2(0.893,0,np.radians(90.)))

        header = Header()
        header.stamp = msg.header.stamp
        dr_msg = DeadReckoning()
        dr_msg.header = header
        dr_msg.x = sonar_pose_dr.x()
        dr_msg.y = sonar_pose_dr.y()
        dr_msg.yaw = sonar_pose_dr.theta()
        dr_msg.pitch = pitch
        dr_msg.roll = roll
        self.odom_publisher.publish(dr_msg)


        # Publish the PoseStamped message and transform
        # self.publisher.publish(self.pose_msg)
        self.broadcaster.sendTransform(self.transform)
        self.publish_static_transform(msg.header.stamp)

    def water_linked_dvl_callback(self, msg:DVL) -> None:
        """Handle an incoming waterlinked dvl message. All we do is handle the signs and cache the data. 

        Args:
            msg (DVL): the incoming dvl message
        """

        self.velocity = np.array([-msg.velocity.x,msg.velocity.y])

    def nortek_dvl_callback(self, msg:NortekMessage) -> None:
        """Handle an incoming nortek dvl message. All we do is handle the signs and cache the data. 

        Args:
            msg (NortekMessage): the incoming dvl message
        """

        # perform a zero order hold, if the dvl has no contact, then just use the old velocity
        if abs(msg.measured_velocity_x) < 4.0 and abs(msg.measured_velocity_x) < 4.0:
            self.velocity = np.array([msg.measured_velocity_x,-msg.measured_velocity_y])

    def gyro_callback(self, msg:GyroMessage) -> None:
        """Handle an incoming FOG message. 

        Args:
            msg (GyroMessage): the fiber optic gyroscope message
        """

        self.gyro_heading += msg.dz    


def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = DeadReckoningNode()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


