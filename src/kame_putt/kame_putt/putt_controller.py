import math

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from turtlesim.msg import Pose
from turtlesim.srv import SetPen, TeleportAbsolute
from .ai_caddy import (
    choose_exploration_speed,
    choose_initial_speed,
    record_trial,
)
from .surface_model import launch_speed, update_speed


class PuttController(Node):
    def __init__(self):
        super().__init__('putt_controller')

        self.declare_parameter('target_x', 8.0)
        self.declare_parameter('target_y', 5.5)
        self.declare_parameter('practice_shots', 1)
        self.declare_parameter('mode', 'infer')

        self.target_x = float(self.get_parameter('target_x').value)
        self.target_y = float(self.get_parameter('target_y').value)

        self.pose = None
        self.state = 'aim'
        self.speed = 0.0
        self.initial_speed = 0.0
        self.target_distance = 0.0
        self.policy = 'explore'
        self.start_x = 5.5
        self.start_y = 5.5
        self.start_theta = 0.0
        self.shot_direction_x = 1.0
        self.shot_direction_y = 0.0
        self.practice_shots = 0
        self.max_practice_shots = int(self.get_parameter('practice_shots').value)
        self.mode = str(self.get_parameter('mode').value)
        self.previous_time = None

        self.publisher = self.create_publisher(
            Twist, '/turtle1/cmd_vel', 10
        )
        self.subscription = self.create_subscription(
            Pose, '/turtle1/pose', self.pose_callback, 10
        )
        self.teleport_client = self.create_client(
            TeleportAbsolute, '/turtle1/teleport_absolute'
        )
        self.pen_client = self.create_client(
            SetPen, '/turtle1/set_pen'
        )
        self.timer = self.create_timer(0.02, self.update)

        self.get_logger().info(
            f'Target: x={self.target_x:.2f}, y={self.target_y:.2f}'
        )

    def pose_callback(self, msg):
        self.pose = msg

    def update(self):
        if self.pose is None or self.state in ('done', 'resetting'):
            return

        command = Twist()

        if self.state == 'aim':
            # このショットを始める位置を記録する
            self.shot_start_x = self.pose.x
            self.shot_start_y = self.pose.y
            dx = self.target_x - self.pose.x
            dy = self.target_y - self.pose.y
            target_angle = math.atan2(dy, dx)
            angle_error = math.atan2(
                math.sin(target_angle - self.pose.theta),
                math.cos(target_angle - self.pose.theta)
            )

            if abs(angle_error) > 0.03:
                command.angular.z = max(
                    -1.5, min(1.5, 2.0 * angle_error)
                )
                self.publisher.publish(command)
                return

            distance = math.hypot(dx, dy)
            self.target_distance = distance
            if distance > 1e-6:
                self.shot_direction_x = dx / distance
                self.shot_direction_y = dy / distance
            if self.mode == 'train':
                self.initial_speed, self.policy = choose_exploration_speed()
            else:
                self.initial_speed, self.policy = choose_initial_speed(distance)
            self.speed = launch_speed(self.initial_speed)
            self.get_logger().info(
                f'AI selected speed={self.initial_speed:.2f} ({self.policy})'
            )
            self.previous_time = self.get_clock().now()
            self.state = 'slide'
            self.get_logger().info(
                f'Shot started: distance={distance:.2f}, '
                f'initial speed={self.speed:.2f}'
            )
            return

        if self.state == 'slide':
            now = self.get_clock().now()
            dt = (now - self.previous_time).nanoseconds / 1e9
            self.previous_time = now

            self.speed = update_speed(self.speed, dt)
            command.linear.x = self.speed
            self.publisher.publish(command)

            if self.speed == 0.0:
                self.publisher.publish(Twist())
                self.state = 'resetting'
                final_distance = math.hypot(
                    self.target_x - self.pose.x,
                    self.target_y - self.pose.y
                )
                traveled_x = self.pose.x - self.shot_start_x
                traveled_y = self.pose.y - self.shot_start_y
                travel_distance = max(
                    0.0,
                    traveled_x * self.shot_direction_x
                    + traveled_y * self.shot_direction_y
                )
                self.get_logger().info(
                    f'Stopped. Distance from target: '
                    f'{final_distance:.2f}; '
                    f'traveled={travel_distance:.2f}'
                )
                record_trial(
                    self.target_distance,
                    self.initial_speed,
                    travel_distance,
                    final_distance,
                )
                self.get_logger().info('Trial saved for AI.')

                self.practice_shots += 1
                if self.practice_shots < self.max_practice_shots:
                    self.reset_for_next_trial()
                else:
                    self.state = 'done'
                    self.get_logger().info(
                        f'Practice complete: {self.practice_shots} shots.'
                    )

    def reset_for_next_trial(self):
        self.state = 'resetting'
        self.pose = None
        self.get_logger().info('Resetting for the next practice shot.')

        request = SetPen.Request()
        request.r = 255
        request.g = 255
        request.b = 255
        request.width = 3
        request.off = 1
        future = self.pen_client.call_async(request)
        future.add_done_callback(self.on_pen_disabled)

    def on_pen_disabled(self, future):
        try:
            future.result()
        except Exception as exc:
            self.get_logger().error(f'Could not lift pen: {exc}')
            self.state = 'done'
            return

        request = TeleportAbsolute.Request()
        request.x = self.start_x
        request.y = self.start_y
        request.theta = self.start_theta
        future = self.teleport_client.call_async(request)
        future.add_done_callback(self.on_teleport_done)

    def on_teleport_done(self, future):
        try:
            future.result()
        except Exception as exc:
            self.get_logger().error(f'Could not reset turtle: {exc}')
            self.state = 'done'
            return

        request = SetPen.Request()
        request.r = 255
        request.g = 255
        request.b = 255
        request.width = 3
        request.off = 0
        future = self.pen_client.call_async(request)
        future.add_done_callback(self.on_pen_enabled)

    def on_pen_enabled(self, future):
        try:
            future.result()
        except Exception as exc:
            self.get_logger().error(f'Could not lower pen: {exc}')
            self.state = 'done'
            return

        self.get_logger().info('Practice reset; waiting for turtle pose.')
        self.state = 'aim'


def main(args=None):
    rclpy.init(args=args)
    node = PuttController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.publisher.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
