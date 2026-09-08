# ROS 2

OceanSim publishes sensor data and accepts velocity commands through ROS when enabled. Use **Enable ROS** in the included scenario to turn publishing on or off.
Follow the example scenario to see how to utlilize ROS in your scenario.
Topics are namespaced by the OmniHandler name (e.g. `SensorExample`) for the example scenario provided. 

**Important**: Ensure you have cloned [Oceansim_msgs](https://github.com/umfieldrobotics/oceansim_ros_msgs) into your ROS workspace. Build and source this package in the terminal from which you are starting Isaac Sim prior to launching. This is required as certain sensors depend on custom defined message types (this currently applies to the DVL only).

## Published topics

| Topic | Message | Sensor |
|-------|---------|--------|
| `{name}/rgb` | `sensor_msgs/Image` | Underwater camera |
| `{name}/depth` | `sensor_msgs/Image` (`32FC1`) | Underwater camera |
| `{name}/pointcloud` | `sensor_msgs/PointCloud2` | Underwater camera |
| `{name}/sonar_image` | `sensor_msgs/Image` | Imaging sonar |
| `{name}/mbes/points` | `sensor_msgs/PointCloud2` | Multibeam bathymetric returns |
| `{name}/sidescan/waterfall` | `sensor_msgs/Image` (`mono8`) | Side-scan waterfall |

For `BathymetricSideScanSonarSensor`, both products come from one point-wise
raw-return grid and carry the same timestamp and sensor frame. The bathymetry
message is a selected view of the same ranges, XYZ points, hit flags, and
intensities used to form the side-scan image. They may still be published on
the two conventional topics above; the split is a message representation, not
two simulated sensors or two ray launches.
| `{name}/imu` | `sensor_msgs/Imu` | IMU |
| `{name}/dvl` | `msgs/Dvl` | DVL |
| `{name}/baro` | `sensor_msgs/FluidPressure` | Barometer |
| `/tf` | `tf2_msgs/TFMessage` | Camera / sonar frames |

Also published when camera or sonar is enabled: `{sensor_name}_camera_info` (`sensor_msgs/CameraInfo`). Camera depth and point cloud may also appear as `{sensor_name}_depth` and `{sensor_name}_pointcloud`.

## Subscribed topics

| Topic | Message | Notes |
|-------|---------|-------|
| `/cmd_vel` | `geometry_msgs/Twist` | Active when control mode is not Manual |

Use this topic to send control commands to your vehicle.

See [Multibeam Echosounder and Side-scan Sonar](multibeam_sidescan.md) for
sensor configuration, coordinate conventions, per-ping Python data, and ROS 2
setup.
