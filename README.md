# aloam_note

学习ALOAM，添加注释代码
为了同步

具体查看CSDN：

https://blog.csdn.net/zardforever123/article/details/125570551

## 文件结构

```
├── CMakeLists.txt - CMake配置文件
├── docker - 
    ├── Dockerfile - 
    ├── Makefile - 
    └── run.sh - 
├── include - 工具类及一些常用的定义
    └── aloam_velodyne - 该项目名称
        ├── common.h - 主要是数学角度转换，cos,sin等
        └── tic_toc.h - 计算处理时间，计时用的
├── launch - 启动文件
    ├── aloam_velodyne_HDL_32.launch - 32线
    ├── aloam_velodyne_HDL_64.launch - 64线
    ├── aloam_velodyne_VLP_16.launch - 16线
    ├── aloam_velodyne_MY.launch - （原程序没有）适配自己的激光雷达
    ├── kitti_helper.launch - 转换kitti数据集为rosbag
    └── kitti_my.launch - （原程序没有）适配自己的文件路径
├── LICENSE - 授权文件（开源许可证）
├── package.xml - 源代码作者信息和依赖包（库）
├── picture - 原README.md中的图片
    ├── kitti_gif.gif
    └── kitti.png
├── README.md - 本文件
├── rviz_cfg - rviz配置文件
    └── aloam_velodyne.rviz
└── src - 源文件
    ├── kittiHelper.cpp - kitti数据集转换为rosbag的节点
    ├── laserMapping.cpp - 建图节点，根据地图对激光里程计的位姿进行进一步的优化，并将新扫描添加到地图中去
    ├── laserOdometry.cpp - 里程计节点，针对相邻两帧的激光扫描数据进行ICP匹配，估计出两帧之间的位姿变换
    ├── lidarFactor.hpp - 非线性优化因子，包含了里程计节点以及建图节点ICP匹配时非线性优化代价函数中的因子定义
    └── scanRegistration.cpp - 点云注册节点，对激光雷达的数据进行筛选与分类，得到相应的特征点云
```



## 保存点云数据

下载[nsh_indooroutdoor.bag](https://drive.google.com/file/d/1s05tBQOLNEDDurlg48KiUWxCp-YqYyGH/view)测试数据，打开终端输入下面代码开始运行：

```bash
roslaunch aloam_velodyne aloam_velodyne_VLP_16.launch
```

再打开一个终端记录地图

```bash
rosbag record -o bag_out /laser_cloud_map
```

在`nsh_indooroutdoor.bag`所在文件夹下打开终端，开始播放数据集：

```bash
rosbag play nsh_indoor_outdoor.bag 
```

跑完之后ctrl+c关闭记录地图的终端会生成一个bag文件，将其转化为pcd格式

```bash
rosrun pcl_ros bag_to_pcd bag_out_xxxx.bag /laser_cloud_map pcd
```

在生成的文件夹pcd目录下，使用pcl_viewer工具打开点云，last.pcd为最后一个pcd文件名：

```bash
pcl_viewer last.pcd
```

kitti数据集的路径配置要求


