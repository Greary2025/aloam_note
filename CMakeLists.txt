# cmake最低版本
cmake_minimum_required(VERSION 2.8.3)
# 项目名称aloam_velodyne
# 使用roslaunch查找时，也是这个项目名称
project(aloam_velodyne)
# 设置程序是运行版本
set(CMAKE_BUILD_TYPE "Release")

# ---------------适用C++11标准(ubuntu 18.04)
# set(CMAKE_CXX_FLAGS "-std=c++11")
# ---------------ubuntu20.04 的PCL库支持
set(CMAKE_CXX_FLAGS "-std=c++14")
# 
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
# 查找调用库的路径（catkin编译需要的库）
find_package(catkin REQUIRED COMPONENTS
# 位姿信息库
  geometry_msgs
  # 导航信息
  nav_msgs
  # 传感器信息
  sensor_msgs
  # cpp文件
  roscpp
  # py文件
  rospy
  # rosbag库
  rosbag
  # 标准信息
  std_msgs
  # 图像转换
  image_transport
  # ROS点云和PCL点云的转换（错）
  # ROS图像和opencv图像之间的转换
  cv_bridge
  # 旋转矩阵
  tf
)

# eigen使用默认路径，不需要寻找
# find_package(Eigen3 REQUIRED)
# PCL库
find_package(PCL REQUIRED)
# OpenCV库
find_package(OpenCV REQUIRED)
# ---------------高版本的cv库，需配置路径
# set(CMAKE_PREFIX_PATH "/usr/include/opencv4") 
# find_package(OpenCV 4.0 QUIET)
# Ceres库：非线性优化库
find_package(Ceres REQUIRED)
# ---------------高版本的cere库，需配置路径
# 需要在CMakelists中指定运行的版本。
# 这里需要找到你需要指定版本的CeresConfig.cmake文件所在的路径。
# 也就是你安装时指定的路径下lib/cmake/Ceres中。
# 在find_packeage前设置Ceres_DIR即可。
# set(Ceres_DIR /usr/local/Ceres-2.1.0/lib/cmake/Ceres)
# find_package(Ceres REQUIRED)

# 包含C++库的路径
include_directories(
  include
  # catkin编译文件夹
	${catkin_INCLUDE_DIRS}
  # PCL文件库 
	${PCL_INCLUDE_DIRS}
  # Ceres库
  ${CERES_INCLUDE_DIRS}
  # opencv库
  ${OpenCV_INCLUDE_DIRS})
# 编译的包
catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs
  DEPENDS EIGEN3 PCL 
  INCLUDE_DIRS include
)

# 生成可执行文件ascanRegistration
# 该程序统计点云数据
add_executable(ascanRegistration src/scanRegistration.cpp)
# 链接到PCL库
target_link_libraries(ascanRegistration ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(alaserOdometry src/laserOdometry.cpp)
target_link_libraries(alaserOdometry ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES})

add_executable(alaserMapping src/laserMapping.cpp)
target_link_libraries(alaserMapping ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES})

add_executable(kittiHelper src/kittiHelper.cpp)
target_link_libraries(kittiHelper ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS})




