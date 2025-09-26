"""
Auro ROS2 Publisher Node
Part of the AurobotNav PRIMECORE navigation system

This ROS2 node publishes φ-enhanced paths to the /auro_path topic
for integration with trance entropy-based navigation systems.
Compatible with both ROS2 environments and fallback mode for testing.
"""

try:
    import rclpy
    from rclpy.node import Node
    from nav_msgs.msg import Path
    from geometry_msgs.msg import PoseStamped
    ROS2_AVAILABLE = True
except ImportError:
    print("ROS2 not available - node functionality will be limited")
    ROS2_AVAILABLE = False

class AuroPublisher(Node if ROS2_AVAILABLE else object):
    def __init__(self):
        if ROS2_AVAILABLE:
            super().__init__('auro_publisher')
            self.pub = self.create_publisher(Path, '/auro_path', 10)
            self.timer = self.create_timer(1.0, self.publish_mock_path)
        else:
            print("ROS2 not available - creating mock publisher")
            self.pub = None

    def publish_mock_path(self):
        if not ROS2_AVAILABLE:
            print("Mock φ-path: [(0,0), (0,1), (1,1), (1,2), (2,2), (3,2), (3,3), (4,3), (4,4)]")
            return
        
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        # Mock φ-path from rift_weaver
        mock_path = [(0,0), (0,1), (1,1), (1,2), (2,2), (3,2), (3,3), (4,3), (4,4)]  # Example φ-enhanced path
        for pos in mock_path:
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = 'map'
            pose.pose.position.x = float(pos[0])
            pose.pose.position.y = float(pos[1])
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0  # Identity quaternion
            msg.poses.append(pose)
        self.pub.publish(msg)
        self.get_logger().info('Published mock φ-path with {} waypoints'.format(len(mock_path)))

def main(args=None):
    if not ROS2_AVAILABLE:
        print("ROS2 not available - simulating node behavior")
        node = AuroPublisher()
        node.publish_mock_path()
        return
    
    rclpy.init(args=args)
    node = AuroPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()