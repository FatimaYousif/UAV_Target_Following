# #!/usr/bin/env python

# import rosbag
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np

# # Path to the ROS bag file
# rosbag_path = '/home/fatima/bytetrack_sim3.bag'  # Replace with your bag file path
# topic_name = '/uav1/rosbag_data'

# # Lists to store data
# timestamps = []
# uav_x = []
# uav_y = []
# uav_z = []
# real_target_x = []
# real_target_y = []
# estimated_target_x = []
# estimated_target_y = []
# target_detected = []

# # Read data from rosbag
# with rosbag.Bag(rosbag_path, 'r') as bag:
#     for topic, msg, t in bag.read_messages(topics=[topic_name]):
#         timestamps.append(t.to_sec())
#         uav_x.append(msg.uav_x)
#         uav_y.append(msg.uav_y)
#         uav_z.append(msg.altitude)
#         real_target_x.append(msg.real_target_x)
#         real_target_y.append(msg.real_target_y)
#         estimated_target_x.append(msg.estimated_target_x)
#         estimated_target_y.append(msg.estimated_target_y)
#         target_detected.append(msg.target_detected)

# # Find the first index where target is detected
# first_detection_idx = next((i for i, detected in enumerate(target_detected) if detected), 0)

# # Slice all data from first detection onwards
# timestamps = timestamps[first_detection_idx:]
# uav_x = uav_x[first_detection_idx:]
# uav_y = uav_y[first_detection_idx:]
# uav_z = uav_z[first_detection_idx:]
# real_target_x = real_target_x[first_detection_idx:]
# real_target_y = real_target_y[first_detection_idx:]
# estimated_target_x = estimated_target_x[first_detection_idx:]
# estimated_target_y = estimated_target_y[first_detection_idx:]

# # Plot 2D Trajectories
# plt.figure(figsize=(10, 6))
# plt.plot(uav_x, uav_y, label='UAV Trajectory', linewidth=2)
# # plt.plot(real_target_x, real_target_y, 
# #          label='Ground Truth Target', linestyle='--', linewidth=2)
# plt.plot(estimated_target_x, estimated_target_y, 
#          label='Estimated Target', linestyle=':', linewidth=2)
# plt.xlabel('X Position [m]')
# plt.ylabel('Y Position [m]')
# plt.title('2D Trajectories After First Detection')
# plt.legend()
# plt.grid(True)
# plt.axis('equal')

# # Plot 3D Trajectories
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(uav_x, uav_y, uav_z, label='UAV Trajectory', linewidth=2)
# # ax.plot(real_target_x, real_target_y, 
# #         np.zeros(len(real_target_x)),  # Target on ground (z=0)
# #         label='Ground Truth Target', linestyle='--', linewidth=2)
# ax.plot(estimated_target_x, estimated_target_y, 
#         np.zeros(len(estimated_target_x)),  # Target on ground (z=0)
#         label='Estimated Target', linestyle=':', linewidth=2)
# ax.set_xlabel('X Position [m]')
# ax.set_ylabel('Y Position [m]')
# ax.set_zlabel('Altitude [m]')
# ax.set_title('3D Trajectories After First Detection')
# ax.legend()

# plt.tight_layout()
# plt.show()

# # Calculate statistics
# total_uav_distance = np.sum(np.sqrt(np.diff(uav_x)**2 + np.diff(uav_y)**2 + np.diff(uav_z)**2))
# total_target_distance = np.sum(np.sqrt(np.diff(real_target_x)**2 + np.diff(real_target_y)**2))

# print(f"\nTrajectory Statistics (After First Detection):")
# print(f"- First detection at time: {timestamps[0]:.2f} seconds")
# print(f"- Total UAV distance traveled: {total_uav_distance:.2f} meters")
# print(f"- Total target distance traveled: {total_target_distance:.2f} meters")
# print(f"- Tracking duration: {timestamps[-1] - timestamps[0]:.2f} seconds")



#!/usr/bin/env python

import rosbag
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Path to the ROS bag file
rosbag_path = '/home/fatima/estdist_botsort.bag'  # Replace with your bag file path
topic_name = '/uav1/rosbag_data'

# Lists to store data
timestamps = []
uav_x = []
uav_y = []
uav_z = []
real_target_x = []
real_target_y = []
estimated_target_x = []
estimated_target_y = []
target_detected = []

# Read data from rosbag
with rosbag.Bag(rosbag_path, 'r') as bag:
    for topic, msg, t in bag.read_messages(topics=[topic_name]):
        timestamps.append(t.to_sec())
        uav_x.append(msg.uav_x)
        uav_y.append(msg.uav_y)
        uav_z.append(msg.altitude)
        real_target_x.append(msg.real_target_x)
        real_target_y.append(msg.real_target_y)
        estimated_target_x.append(msg.estimated_target_x)
        estimated_target_y.append(msg.estimated_target_y)
        target_detected.append(msg.target_detected)

# Find the first index where target is detected
first_detection_idx = next((i for i, detected in enumerate(target_detected) if detected), 0)

# Slice all data from first detection onwards
timestamps = timestamps[first_detection_idx:]
uav_x = uav_x[first_detection_idx:]
uav_y = uav_y[first_detection_idx:]
uav_z = uav_z[first_detection_idx:]
real_target_x = real_target_x[first_detection_idx:]
real_target_y = real_target_y[first_detection_idx:]
estimated_target_x = estimated_target_x[first_detection_idx:]
estimated_target_y = estimated_target_y[first_detection_idx:]

# Get start and end points
uav_start = (uav_x[0], uav_y[0], uav_z[0])
uav_end = (uav_x[-1], uav_y[-1], uav_z[-1])
estimated_start = (estimated_target_x[0], estimated_target_y[0], 0)
estimated_end = (estimated_target_x[-1], estimated_target_y[-1], 0)

# Plot 2D Trajectories
plt.figure(figsize=(10, 6))
plt.plot(uav_x, uav_y, label='UAV Trajectory', linewidth=2)
plt.plot(estimated_target_x, estimated_target_y, 
         label='Estimated Target', linestyle=':', linewidth=2)

# Mark start and end points for UAV
plt.scatter(uav_x[0], uav_y[0], c='green', marker='o', s=100, label='UAV Start')
plt.scatter(uav_x[-1], uav_y[-1], c='red', marker='x', s=100, label='UAV End')

# Mark start and end points for Estimated Target
plt.scatter(estimated_target_x[0], estimated_target_y[0], c='blue', marker='o', s=100, label='Estimated Start')
plt.scatter(estimated_target_x[-1], estimated_target_y[-1], c='orange', marker='x', s=100, label='Estimated End')

plt.xlabel('X Position [m]')
plt.ylabel('Y Position [m]')
plt.title('2D Trajectories')
plt.legend()
plt.grid(True)
plt.axis('equal')

# Plot 3D Trajectories
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(uav_x, uav_y, uav_z, label='UAV Trajectory', linewidth=2)
ax.plot(estimated_target_x, estimated_target_y, 
        np.zeros(len(estimated_target_x)),  # Target on ground (z=0)
        label='Estimated Target', linestyle=':', linewidth=2)

# Mark start and end points for UAV
ax.scatter(uav_x[0], uav_y[0], uav_z[0], c='green', marker='o', s=100, label='UAV Start')
ax.scatter(uav_x[-1], uav_y[-1], uav_z[-1], c='red', marker='x', s=100, label='UAV End')

# Mark start and end points for Estimated Target
ax.scatter(estimated_target_x[0], estimated_target_y[0], 0, c='blue', marker='o', s=100, label='Estimated Start')
ax.scatter(estimated_target_x[-1], estimated_target_y[-1], 0, c='orange', marker='x', s=100, label='Estimated End')

ax.set_xlabel('X Position [m]')
ax.set_ylabel('Y Position [m]')
ax.set_zlabel('Altitude [m]')
ax.set_title('3D Trajectories After First Detection')
ax.legend()

plt.tight_layout()
plt.show()

# Calculate statistics
total_uav_distance = np.sum(np.sqrt(np.diff(uav_x)**2 + np.diff(uav_y)**2 + np.diff(uav_z)**2))
total_target_distance = np.sum(np.sqrt(np.diff(real_target_x)**2 + np.diff(real_target_y)**2))

print(f"\nTrajectory Statistics (After First Detection):")
print(f"- First detection at time: {timestamps[0]:.2f} seconds")
print(f"- Total UAV distance traveled: {total_uav_distance:.2f} meters")
print(f"- Total target distance traveled: {total_target_distance:.2f} meters")
print(f"- Tracking duration: {timestamps[-1] - timestamps[0]:.2f} seconds")