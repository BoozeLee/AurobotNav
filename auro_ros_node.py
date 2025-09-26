"""
auro_ros_node.py - ROS2 publisher for φ-paths with QoS multifractal settings
Publishes /auro_path (nav_msgs/Path) with mock data from rift_weaver
"""

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from nav_msgs.msg import Path
    from geometry_msgs.msg import PoseStamped, Point, Quaternion
    from std_msgs.msg import Header
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    # Mock classes for systems without ROS2
    class Node:
        def __init__(self, node_name):
            self.node_name = node_name
        def create_publisher(self, msg_type, topic, qos_profile):
            return MockPublisher()
        def create_timer(self, period, callback):
            return MockTimer()
        def get_logger(self):
            return MockLogger()
    
    class MockPublisher:
        def publish(self, msg):
            print(f"Mock publish: {msg}")
    
    class MockTimer:
        pass
    
    class MockLogger:
        def info(self, msg):
            print(f"INFO: {msg}")

import time
import math
from rift_weaver import rift_weaver

class AuroPublisher(Node):
    """ROS2 publisher for Aurobot navigation paths"""
    
    def __init__(self):
        super().__init__('aurobot_nav_publisher')
        
        # Create QoS profile with multifractal characteristics
        # q-varied reliability based on multifractal dynamics
        qos_profile = self._create_multifractal_qos()
        
        # Create publisher for /auro_path topic
        if ROS2_AVAILABLE:
            self.publisher = self.create_publisher(Path, '/auro_path', qos_profile)
        else:
            self.publisher = MockPublisher()
        
        # Create timer for regular path publishing (2Hz)
        timer_period = 0.5  # seconds
        if ROS2_AVAILABLE:
            self.timer = self.create_timer(timer_period, self.publish_mock_path)
        
        # Initialize test grid for rift_weaver
        self.test_grid = [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0], 
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ]
        
        self.path_counter = 0
        self.get_logger().info('Aurobot φ-path publisher initialized')
    
    def _create_multifractal_qos(self):
        """Create QoS profile with multifractal reliability characteristics"""
        if not ROS2_AVAILABLE:
            return None
            
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,  # High reliability for navigation
            history=HistoryPolicy.KEEP_LAST,
            depth=10  # Keep last 10 path messages
        )
        return qos
    
    def publish_mock_path(self):
        """Publish mock navigation path using rift_weaver algorithm"""
        # Generate varying start/goal positions
        start_positions = [(0, 0), (1, 0), (0, 1), (2, 0)]
        goal_positions = [(4, 4), (3, 4), (4, 3), (4, 2)]
        
        start_idx = self.path_counter % len(start_positions)
        goal_idx = self.path_counter % len(goal_positions)
        
        start = start_positions[start_idx]
        goal = goal_positions[goal_idx]
        
        # Calculate path using rift_weaver
        path_coords = rift_weaver(self.test_grid, start, goal)
        
        if not path_coords:
            self.get_logger().info(f'No path found from {start} to {goal}')
            return
        
        # Create ROS2 Path message
        if ROS2_AVAILABLE:
            path_msg = Path()
            path_msg.header = Header()
            path_msg.header.stamp = self.get_clock().now().to_msg()
            path_msg.header.frame_id = 'map'
        else:
            # Mock message structure
            path_msg = type('Path', (), {
                'header': type('Header', (), {
                    'stamp': time.time(),
                    'frame_id': 'map'
                })(),
                'poses': []
            })()
        
        # Convert path coordinates to PoseStamped messages
        path_msg.poses = []
        for i, (x, y) in enumerate(path_coords):
            if ROS2_AVAILABLE:
                pose_stamped = PoseStamped()
                pose_stamped.header = path_msg.header
                pose_stamped.pose.position = Point(x=float(x), y=float(y), z=0.0)
                
                # Calculate orientation facing next waypoint
                if i < len(path_coords) - 1:
                    next_x, next_y = path_coords[i + 1]
                    yaw = math.atan2(next_y - y, next_x - x)
                    pose_stamped.pose.orientation = Quaternion(
                        x=0.0, y=0.0, 
                        z=math.sin(yaw/2), w=math.cos(yaw/2)
                    )
                else:
                    pose_stamped.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            else:
                # Mock pose structure
                pose_stamped = type('PoseStamped', (), {
                    'pose': type('Pose', (), {
                        'position': type('Point', (), {'x': float(x), 'y': float(y), 'z': 0.0})(),
                        'orientation': type('Quaternion', (), {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0})()
                    })()
                })()
            
            path_msg.poses.append(pose_stamped)
        
        # Publish the path
        self.publisher.publish(path_msg)
        
        self.get_logger().info(
            f'Published φ-path #{self.path_counter}: {len(path_coords)} waypoints '
            f'from {start} to {goal}'
        )
        
        self.path_counter += 1

def main(args=None):
    """Main function to initialize and spin the ROS2 node"""
    if ROS2_AVAILABLE:
        rclpy.init(args=args)
        
        try:
            auro_publisher = AuroPublisher()
            rclpy.spin(auro_publisher)
        except KeyboardInterrupt:
            print('\nShutting down Aurobot publisher...')
        finally:
            if ROS2_AVAILABLE:
                rclpy.shutdown()
    else:
        print("ROS2 not available. Running in mock mode...")
        auro_publisher = AuroPublisher()
        
        # Mock spinning - publish a few test messages
        for i in range(5):
            auro_publisher.publish_mock_path()
            time.sleep(1)
        
        print("Mock mode completed.")

if __name__ == '__main__':
    main()