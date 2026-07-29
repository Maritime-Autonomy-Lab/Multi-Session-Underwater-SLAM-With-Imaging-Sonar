#!/usr/bin/env python3

import rclpy
from object_slam import ObjectSLAM

def main(args=None) -> None:
    """Main function for SLAM node

    Args:
        args (_type_, optional): Standard input argument object. Defaults to None.
    """

    rclpy.init(args=args)
    minimal_subscriber = ObjectSLAM()
    minimal_subscriber.get_node_ready()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

