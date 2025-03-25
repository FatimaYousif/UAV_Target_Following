The other MRS-CTU files/folders from their 'old_system' branch (of all the repositories) is given in this link:
https://drive.google.com/drive/u/8/my-drive


### TO LAUNCH
```bash
export UAV_NAME="uav1"
export RUN_TYPE="simulation"
export UAV_TYPE="f550"
export WORLD_NAME="simulation"
export SENSORS="garmin_down"
export ODOMETRY_TYPE="gps"
```
Alternatively, refer to the script available [here](https://github.com/lunagava/summer-school-2022/blob/e8540f564d3c8921710f0020ace035d2c57dfb0e/simulation/mount/singularity_zshrc.sh#L23).

```bash
roscore
```
```bash
roslaunch mrs_simulation simulation.launch world_name:=experiment2 gui:=True
```
```bash
gz camera -c gzclient_camera -f ${UAV_NAME}
```

```bash
rosservice call /mrs_drone_spawner/spawn "1 $UAV_TYPE --pos 0 0 0 0 --enable-rangefinder --use_realistic_realsense --enable-realsense-front-pitched"
```
```bash
roslaunch mrs_uav_general core.launch config_constraint:=./default/constraint_manager/constraints_custom.yaml
```
```bash
rosservice call /$UAV_NAME/control_manager/switch_controller Se3Controller
```
```bash
rosservice call /$UAV_NAME/gain_manager/set_gains supersoft
```
```bash
roslaunch mrs_uav_general automatic_start.launch
```

```bash
roslaunch ultralytics_ros tracker.launch debug:=true yolo_model:=yolov8n.pt
```

```bash
rosservice call /$UAV_NAME/mavros/cmd/arming 1
```
```bash
sleep 2
```
```bash
rosservice call /$UAV_NAME/mavros/set_mode 0 offboard
```

## Others:
```bash
tmux list-sessions
```
```bash
tmux kill-session -t simulation
```
```bash
ps aux | grep ros
```


## Deprecated or Not Recommended Commands
~~roslaunch mrs_uav_status status.launch~~

~~cd /catkin_ws/src/uav_control_test/src/data/Random_EXPS~~
~~sudo chmod +x main.py~~
~~./main.py~~

~~cd ~/catkin_ws/src/Simulation/one_drone_gps_realsense~~
~~./main.py~~

