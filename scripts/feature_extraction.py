#!/usr/bin/env python3

import cv2
import rclpy
import numpy as np

from rclpy.node import Node
from cfar_wrapper import CFAR
from cv_bridge import CvBridge
from scipy.interpolate import interp1d
from sonar_oculus.msg import OculusPing
from sensor_msgs.msg import Image, PointCloud2, PointField

from topics import *
import pcl

class SonarSubNode(Node):
    """A node that subscriber to sonar messages. 
    """

    def __init__(self):
        super().__init__('feature_extraction')

        # node and subscriber
        # self.node = rclpy.create_node('sonar_image_pub')
        self.subscription = self.create_subscription(OculusPing,'/sonar_oculus_node/M750d/ping',self.listener_callback,10)
        self.subscription  # prevent unused variable warning

        # Declare parameters with defaults
        self.declare_parameter('threshold', 65)
        self.declare_parameter('alg', 'SOCA')
        self.declare_parameter('guard_cells', 40)
        self.declare_parameter('training_cells', 10)
        self.declare_parameter('pfa', 0.1)
        self.declare_parameter('tau', 10)

        # Load parameter values
        self.threshold = self.get_parameter('threshold').get_parameter_value().integer_value
        self.alg = self.get_parameter('alg').get_parameter_value().string_value
        self.guard_cells = self.get_parameter('guard_cells').get_parameter_value().integer_value
        self.training_cells = self.get_parameter('training_cells').get_parameter_value().integer_value
        self.pfa = self.get_parameter('pfa').get_parameter_value().double_value
        self.tau = self.get_parameter('tau').get_parameter_value().integer_value

        # do a print statement of params    
        self.get_logger().info(
            f"Loaded feature extraction params: alg={self.alg}, threshold={self.threshold}, "
            f"guard_cells={self.guard_cells}, training_cells={self.training_cells}, "
            f"pfa={self.pfa}, tau={self.tau}"
        )

        # define the detector
        self.detector = CFAR(self.guard_cells, self.training_cells, self.pfa, self.tau) # CFAR(40,10,0.1,10)
        # self.threshold = 65
        # self.alg = "SOCA"

        # publisher
        self.publisher_image = self.create_publisher(Image,"sonar_feature_image",10)
        self.publisher_point_cloud = self.create_publisher(PointCloud2,SONAR_FEATURE_TOPIC,10)

        # cv bridge object
        self.bridge = CvBridge()

        # for remapping from polar to cartisian
        self.res = None
        self.height = None
        self.rows = None
        self.width = None
        self.cols = None
        self.map_x = None
        self.map_y = None
        self.f_bearings = None
        self.to_rad = lambda bearing: bearing * np.pi / 18000
        self.REVERSE_Z = 1
        self.maxRange = None

        # image logging
        self.log_raw_images = False
        self.count = 0

    def generate_map_xy(self, ping: OculusPing) -> None:
        """Generate a mesh grid map for the sonar image, this enables converison to cartisian from the 
        source polar images

        Args:
            ping (OculusPing): incoming sonar message
        """

        # get the parameters from the ping message
        _res = ping.range_resolution
        _height = ping.num_ranges * _res
        _rows = ping.num_ranges
        _width = np.sin(
            self.to_rad(ping.bearings[-1] - ping.bearings[0]) / 2) * _height * 2
        _cols = int(np.ceil(_width / _res))

        # check if the parameters have changed
        if self.res == _res and self.height == _height and self.rows == _rows and self.width == _width and self.cols == _cols:
            return

        # if they have changed do some work    
        self.res, self.height, self.rows, self.width, self.cols = _res, _height, _rows, _width, _cols

        # generate the mapping
        bearings = self.to_rad(np.asarray(ping.bearings, dtype=np.float32))
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
        b = np.arctan2(y, x) * self.REVERSE_Z
        r = np.sqrt(np.square(x) + np.square(y))
        self.map_y = np.asarray(r / self.res, dtype=np.float32)
        self.map_x = np.asarray(f_bearings(b), dtype=np.float32)

    def numpy_to_pointcloud(self, points: np.array) -> None:
        """Convert a numpy array to a point cloud

        Args:
            points (np.array): the incoming array
        """

        pointcloud_msg = PointCloud2()
        pointcloud_msg.header.frame_id = 'slam_link'  # Set frame ID
        pointcloud_msg.height = 1  # 1 row of points
        pointcloud_msg.width = len(points)  # Number of points
        pointcloud_msg.fields.append(PointField(
            name="x", offset=0, datatype=PointField.FLOAT32, count=1))
        pointcloud_msg.fields.append(PointField(
            name="y", offset=4, datatype=PointField.FLOAT32, count=1))
        pointcloud_msg.fields.append(PointField(
            name="z", offset=8, datatype=PointField.FLOAT32, count=1))
        pointcloud_msg.point_step = 12  # 3 floats (x, y, z)
        pointcloud_msg.row_step = pointcloud_msg.point_step * len(points)
        pointcloud_msg.is_dense = True  # All points are finite
        pointcloud_msg.is_bigendian = False  # Endianness
        pointcloud_msg.data = np.array(points, dtype=np.float32).tobytes()

        return pointcloud_msg

    def listener_callback(self, msg:OculusPing) -> None:
        """Handle an incoming sonar message.

        Args:
            msg (OculusPing): the incoming sonar message
        """         

        # generate a mapping from polar to cart coords
        self.generate_map_xy(msg)

        # decompress the sonar image
        img = np.asarray(self.bridge.compressed_imgmsg_to_cv2(msg.ping))
        img = np.flip(img,axis=1)

        # call CFAR on the sonar image
        peaks = self.detector.detect(img, self.alg)
        peaks &= img > self.threshold

        # remapp the CFAR image into cartisiain coords
        peaks = cv2.remap(peaks, self.map_x, self.map_y, cv2.INTER_LINEAR)
        feature_locations = np.c_[np.nonzero(peaks)] # grab the non zero points in the image

        # remap the input image into cart coords and apply a color map
        # img = cv2.remap(img, self.map_x, self.map_y, cv2.INTER_LINEAR)
        # img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        # log the remap
        #np.save("/home/jake/Desktop/MMMR/config/bridge_map_x.npy", self.map_x)
        #np.save("/home/jake/Desktop/MMMR/config/bridge__map_y.npy", self.map_y)

        # do some raw data logging if desired
        if self.log_raw_images:
            if self.count % 8 == 0:
                cv2.imwrite("/home/jake/Desktop/bridge_sonar_images/penn_" + f"{self.count:06d}.png", img)
            self.count += 1

        # apply a color map to make the sonar image pop in the sun
        # img = cv2.applyColorMap(img,2)

        # draw the feature points
        #for location in feature_locations:
        #    cv2.circle(img,(location[1],location[0]),1,(0,0,255),-1)

        # convert to a point cloud
        x = feature_locations[:,1] - self.cols / 2.
        x = (-1 * ((x / float(self.cols / 2.)) * (self.width / 2.))) #+ self.width
        y = (-1*(feature_locations[:,0] / float(self.rows)) * self.height) + self.height
        points = np.column_stack((y,x,np.zeros_like(x)))

        if len(points) != 0:
            points, _ = pcl.downsample(points, points[:,0], 0.5)
            points = pcl.remove_outlier(points, 1.0, 5)

        if len(points) == 0:
            points = np.array([[np.nan, np.nan, np.nan]])

        # convert to a point cloud message
        point_cloud_message = self.numpy_to_pointcloud(points)
        point_cloud_message.header.stamp = msg.header.stamp
        self.publisher_point_cloud.publish(point_cloud_message)

        # publish a message
        msg_out = self.bridge.cv2_to_imgmsg(img)
        msg_out.header.stamp = msg.header.stamp # make sure to copy the timestamp
        self.publisher_image.publish(msg_out)


def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = SonarSubNode()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()



 
