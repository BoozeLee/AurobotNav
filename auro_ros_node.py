"""
auro_ros_node.py - ROS2 publisher /auro_path (nav_msgs/Path) w/ mock from rift_weaver
QoS q-varied for Δh=0.5. Full spin.
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import time
import math
from rift_weaver import rift_weaver, PHI

class AuroPublisher(Node):
    """ROS2 publisher node for Aurobot navigation paths"""
    
    def __init__(self):
        super().__init__('auro_ros_node')
        
        # QoS profile with q-varied settings for Δh=0.5
        qos_profile = QoSProfile(depth=10)
        qos_profile.reliability = ReliabilityPolicy.RELIABLE
        qos_profile.durability = DurabilityPolicy.TRANSIENT_LOCAL
        
        # Create publisher for /auro_path topic
        self.path_publisher = self.create_publisher(
            Path,
            '/auro_path',
            qos_profile
        )
        
        # Create timer for periodic publishing (2 Hz)
        self.timer = self.create_timer(0.5, self.publish_mock_path)
        
        # Navigation state
        self.sequence_counter = 0
        self.current_goal = (4, 4)  # Default goal for 5x5 grid
        
        self.get_logger().info('Auro ROS Node initialized - publishing to /auro_path')
        
    def create_mock_grid(self):
        """Create mock 5x5 fractal grid for navigation"""
        # Create grid with some obstacles for interesting paths
        grid_patterns = [
            # Pattern 1: Basic obstacles
            [
                [0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0]
            ],
            # Pattern 2: Different obstacles
            [
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 0, 1, 0, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0]
            ],
            # Pattern 3: Minimal obstacles
            [
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0]
            ]
        ]
        
        # Cycle through patterns
        pattern_index = self.sequence_counter % len(grid_patterns)
        return grid_patterns[pattern_index]
    
    def publish_mock_path(self):
        """Publish mock navigation path using rift_weaver"""
        try:
            # Create mock grid
            grid = self.create_mock_grid()
            
            # Generate path using rift_weaver
            start = (0, 0)
            path_coords = rift_weaver(grid, start, self.current_goal)
            
            if not path_coords:
                self.get_logger().warn('No path found by rift_weaver')
                return
                
            # Create ROS Path message
            path_msg = Path()
            path_msg.header.stamp = self.get_clock().now().to_msg()
            path_msg.header.frame_id = 'auro_nav_frame'
            
            # Convert path coordinates to PoseStamped messages
            for i, (x, y) in enumerate(path_coords):
                pose_stamped = PoseStamped()
                pose_stamped.header.stamp = path_msg.header.stamp
                pose_stamped.header.frame_id = path_msg.header.frame_id
                
                # Convert grid coordinates to real-world coordinates (meters)
                # Scale factor for 5x5 grid -> 5m x 5m area
                scale = 1.0
                pose_stamped.pose.position.x = float(x * scale)
                pose_stamped.pose.position.y = float(y * scale)
                pose_stamped.pose.position.z = 0.0
                
                # Calculate orientation towards next waypoint
                if i < len(path_coords) - 1:
                    next_x, next_y = path_coords[i + 1]
                    dx = next_x - x
                    dy = next_y - y
                    yaw = math.atan2(dy, dx)
                else:
                    yaw = 0.0  # Final orientation
                
                # Convert yaw to quaternion
                pose_stamped.pose.orientation.x = 0.0
                pose_stamped.pose.orientation.y = 0.0
                pose_stamped.pose.orientation.z = math.sin(yaw / 2.0)
                pose_stamped.pose.orientation.w = math.cos(yaw / 2.0)
                
                path_msg.poses.append(pose_stamped)
            
            # Publish the path
            self.path_publisher.publish(path_msg)
            
            # Log path information with φ-optimization details
            phi_factor = PHI * len(path_coords)  # φ-based path efficiency
            self.get_logger().info(
                f'Published path: {len(path_coords)} waypoints, '
                f'φ-efficiency: {phi_factor:.3f}, '
                f'sequence: {self.sequence_counter}'
            )
            
            # Update sequence and occasionally change goal
            self.sequence_counter += 1
            if self.sequence_counter % 10 == 0:
                # Cycle through different goals
                goals = [(4, 4), (0, 4), (4, 0), (2, 2)]
                goal_index = (self.sequence_counter // 10) % len(goals)
                self.current_goal = goals[goal_index]
                self.get_logger().info(f'Changed goal to {self.current_goal}')
                
        except Exception as e:
            self.get_logger().error(f'Error publishing path: {str(e)}')

def main(args=None):
    """Main function to initialize and run the ROS node"""
    rclpy.init(args=args)
    
    try:
        auro_publisher = AuroPublisher()
        
        print("=== Auro ROS Node Started ===")
        print("Publishing paths to /auro_path topic")
        print("Press Ctrl+C to stop...")
        
        # Spin the node
        rclpy.spin(auro_publisher)
        
    except KeyboardInterrupt:
        print("\nShutting down Auro ROS Node...")
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        if 'auro_publisher' in locals():
            auro_publisher.destroy_node()
        rclpy.shutdown()
        print("Auro ROS Node shutdown complete")

if __name__ == '__main__':
    main()