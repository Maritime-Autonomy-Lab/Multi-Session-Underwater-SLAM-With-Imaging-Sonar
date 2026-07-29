#!/usr/bin/env python3

import rclpy
from slam_ros import SLAMNode

def main(args=None) -> None:
    """Main function for SLAM node

    Args:
        args (_type_, optional): Standard input argument object. Defaults to None.
    """

    rclpy.init(args=args)
    minimal_subscriber = SLAMNode()
    minimal_subscriber.init_node()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

