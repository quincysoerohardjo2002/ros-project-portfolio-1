# ros-project-portfolio-1

Duckietown portfolio project — self-driving behaviours for **joepduckiebot**.

Built on the official [template-ros](https://github.com/duckietown/template-ros) (DTProject v3).

## Packages

| Package | Description | Status |
|---|---|---|
| `lane_following` | Detects lane markings and steers the Duckiebot to stay in-lane | 🔲 placeholder |
| `traffic_light_detection` | Detects red/green traffic lights from the camera feed | 🔲 placeholder |
| `duckie_detection` | Detects rubber duckies on the road and estimates distance | 🔲 placeholder |

## Project structure

```
packages/
├── lane_following/          # Lane following node
├── traffic_light_detection/ # Traffic light detection node
└── duckie_detection/        # Duckie detection node
launchers/
└── default.sh               # Launches all three nodes
```

## Build & run

```bash
# Build the Docker image
dts devel build -f

# Run on the virtual Duckiebot
dts devel run -H joepduckiebot.local
```

## Dependencies

- **Python 3**: `numpy` (OpenCV is provided by the Duckietown base image)
- **Duckietown**: `dt-duckietown-msgs`
- **ROS**: `rospy`, `std_msgs`, `sensor_msgs`, `duckietown_msgs`