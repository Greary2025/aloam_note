// 作者信息：Tong Qin（邮箱：qintonguav@gmail.com）、Shaozu Cao（邮箱：saozu.cao@connect.ust.hk）
// 功能说明：KITTI数据集辅助工具，用于将KITTI原始数据（激光雷达、图像、真值位姿）转换为ROS消息或rosbag文件

// 包含必要的头文件
#include <iostream>                  // C++标准输入输出库
#include <fstream>                   // 文件流操作库（用于读取二进制/文本文件）
#include <iterator>                  // 迭代器相关功能
#include <string>                    // 字符串处理库
#include <vector>                    // 动态数组容器
#include <opencv2/opencv.hpp>        // OpenCV计算机视觉库（处理图像）
#include <image_transport/image_transport.h>  // ROS图像传输库（发布图像消息）
#include <opencv2/highgui/highgui.hpp>  // OpenCV高层GUI库（图像显示等）
#include <nav_msgs/Odometry.h>       // ROS导航消息（里程计）
#include <nav_msgs/Path.h>           // ROS导航消息（路径轨迹）
#include <ros/ros.h>                 // ROS核心库
#include <rosbag/bag.h>              // ROS bag文件操作库（读写rosbag）
#include <geometry_msgs/PoseStamped.h>  // ROS几何消息（带时间戳的位姿）
#include <cv_bridge/cv_bridge.h>     // OpenCV与ROS图像转换桥接库
#include <sensor_msgs/image_encodings.h>  // ROS图像编码格式定义
#include <eigen3/Eigen/Dense>        // Eigen矩阵运算库（处理位姿变换）
#include <pcl/point_cloud.h>         // PCL点云库（点云数据结构）
#include <pcl/point_types.h>         // PCL点类型定义（如PointXYZI）
#include <pcl_conversions/pcl_conversions.h>  // PCL与ROS点云转换库
#include <sensor_msgs/PointCloud2.h> // ROS点云消息类型

// 读取激光雷达二进制数据，返回浮点数组
// 参数：lidar_data_path - 激光雷达数据文件路径（.bin格式）
// 返回值：包含激光雷达数据的浮点数组（每4个元素为一个点：x,y,z,intensity）
std::vector<float> read_lidar_data(const std::string lidar_data_path)
{
    // 以二进制模式打开激光雷达数据文件
    std::ifstream lidar_data_file(lidar_data_path, std::ifstream::in | std::ifstream::binary);
    lidar_data_file.seekg(0, std::ios::end);          // 移动文件指针到末尾
    const size_t num_elements = lidar_data_file.tellg() / sizeof(float);  // 计算文件总浮点数数量
    lidar_data_file.seekg(0, std::ios::beg);          // 移动文件指针回开头

    // 分配内存并读取数据到缓冲区
    std::vector<float> lidar_data_buffer(num_elements);
    lidar_data_file.read(reinterpret_cast<char *>(&lidar_data_buffer[0]), num_elements * sizeof(float));
    return lidar_data_buffer;  // 返回激光雷达数据缓冲区
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "kitti_helper");  // 初始化ROS节点，节点名"kitti_helper"
    ros::NodeHandle n("~");                 // 创建私有节点句柄（用于获取参数）

    // 从参数服务器获取配置参数
    std::string dataset_folder, sequence_number, output_bag_file;
    n.getParam("dataset_folder", dataset_folder);      // KITTI数据集根目录
    n.getParam("sequence_number", sequence_number);    // 序列编号（如"00"）
    std::cout << "Reading sequence " << sequence_number << " from " << dataset_folder << '\n';
    bool to_bag;
    n.getParam("to_bag", to_bag);                      // 是否输出为rosbag文件
    if (to_bag)
        n.getParam("output_bag_file", output_bag_file); // rosbag输出路径
    int publish_delay;
    n.getParam("publish_delay", publish_delay);        // 发布延迟（控制发布频率）
    publish_delay = publish_delay <= 0 ? 1 : publish_delay;  // 延迟至少为1（避免0值）

    // 创建ROS发布者：激光点云、左右图像、真值里程计、真值路径
    ros::Publisher pub_laser_cloud = n.advertise<sensor_msgs::PointCloud2>("/velodyne_points", 2);  // 激光点云话题
    image_transport::ImageTransport it(n);
    image_transport::Publisher pub_image_left = it.advertise("/image_left", 2);  // 左图像话题
    image_transport::Publisher pub_image_right = it.advertise("/image_right", 2); // 右图像话题
    ros::Publisher pubOdomGT = n.advertise<nav_msgs::Odometry>("/odometry_gt", 5);  // 真值里程计话题
    nav_msgs::Odometry odomGT;  // 真值里程计消息对象
    // -------------ubuntu 18.04（坐标系ID带前斜杠）
    // odomGT.header.frame_id = "/camera_init";
    // -------------ubuntu 20.04（坐标系ID不带前斜杠）
    odomGT.header.frame_id = "camera_init";  // 父坐标系（初始相机坐标系）
    odomGT.child_frame_id = "/ground_truth"; // 子坐标系（真值坐标系）
    ros::Publisher pubPathGT = n.advertise<nav_msgs::Path>("/path_gt", 5);  // 真值路径话题
    nav_msgs::Path pathGT;  // 真值路径消息对象
    // -------------ubuntu 18.04
    // pathGT.header.frame_id = "/camera_init";
    // -------------ubuntu 20.04
    pathGT.header.frame_id = "camera_init";  // 路径的参考坐标系

    // 打开时间戳文件（记录每帧数据的时间）
    std::string timestamp_path = "sequences/" + sequence_number + "/times.txt";
    std::ifstream timestamp_file(dataset_folder + timestamp_path, std::ifstream::in);
    // 打开真值位姿文件（记录每帧的真值位姿）
    std::string ground_truth_path = "results/" + sequence_number + ".txt";
    std::ifstream ground_truth_file(dataset_folder + ground_truth_path, std::ifstream::in);

    // 初始化rosbag（若需要输出）
    rosbag::Bag bag_out;
    if (to_bag)
        bag_out.open(output_bag_file, rosbag::bagmode::Write);  // 以写模式打开bag文件

    // 坐标系转换矩阵（KITTI原始坐标系 -> 程序使用的坐标系）
    Eigen::Matrix3d R_transform;
    R_transform << 0, 0, 1, -1, 0, 0, 0, -1, 0;  // 旋转矩阵（坐标轴变换）
    Eigen::Quaterniond q_transform(R_transform);  // 转换为四元数

    std::string line;          // 临时存储文件读取的行
    std::size_t line_num = 0;  // 帧计数（从0开始）

    ros::Rate r(10.0 / publish_delay);  // 控制发布频率（10Hz / 延迟倍数）

    // 逐帧处理数据（直到时间戳文件读取完毕或ROS终止）
    while (std::getline(timestamp_file, line) && ros::ok())
    {
        float timestamp = stof(line);  // 解析当前帧时间戳（秒）

        // 构造左右图像路径（KITTI序列的image_0为左图，image_1为右图）
        std::stringstream left_image_path, right_image_path;
        left_image_path << dataset_folder << "sequences/" + sequence_number + "/image_0/" 
                        << std::setfill('0') << std::setw(6) << line_num << ".png";  // 左图路径（6位补零）
        cv::Mat left_image = cv::imread(left_image_path.str(), cv::IMREAD_GRAYSCALE);  // 读取左图（灰度模式）
        right_image_path << dataset_folder << "sequences/" + sequence_number + "/image_1/" 
                         << std::setfill('0') << std::setw(6) << line_num << ".png";  // 右图路径
        cv::Mat right_image = cv::imread(left_image_path.str(), cv::IMREAD_GRAYSCALE);  // 读取右图（注意：原代码可能笔误，应为right_image_path.str()）

        // 读取当前帧的真值位姿（3x4变换矩阵：R|t）
        std::getline(ground_truth_file, line);
        std::stringstream pose_stream(line);
        std::string s;
        Eigen::Matrix<double, 3, 4> gt_pose;  // 3x4矩阵（旋转+平移）
        for (std::size_t i = 0; i < 3; ++i)  // 解析矩阵元素（空格分隔）
        {
            for (std::size_t j = 0; j < 4; ++j)
            {
                std::getline(pose_stream, s, ' ');
                gt_pose(i, j) = stof(s);
            }
        }

        // 转换真值位姿到程序坐标系（四元数+平移向量）
        Eigen::Quaterniond q_w_i(gt_pose.topLeftCorner<3, 3>());  // 提取原始旋转矩阵并转为四元数
        Eigen::Quaterniond q = q_transform * q_w_i;  // 应用坐标系旋转变换
        q.normalize();  // 归一化四元数（保证单位长度）
        Eigen::Vector3d t = q_transform * gt_pose.topRightCorner<3, 1>();  // 应用坐标系平移变换

        // 填充真值里程计消息并发布
        odomGT.header.stamp = ros::Time().fromSec(timestamp);  // 同步时间戳
        odomGT.pose.pose.orientation.x = q.x();  // 四元数x分量
        odomGT.pose.pose.orientation.y = q.y();  // 四元数y分量
        odomGT.pose.pose.orientation.z = q.z();  // 四元数z分量
        odomGT.pose.pose.orientation.w = q.w();  // 四元数w分量（实部）
        odomGT.pose.pose.position.x = t(0);      // 平移向量x分量
        odomGT.pose.pose.position.y = t(1);      // 平移向量y分量
        odomGT.pose.pose.position.z = t(2);      // 平移向量z分量
        pubOdomGT.publish(odomGT);  // 发布真值里程计消息

        // 更新并发布真值路径消息
        geometry_msgs::PoseStamped poseGT;
        poseGT.header = odomGT.header;  // 继承时间戳和坐标系
        poseGT.pose = odomGT.pose.pose;  // 继承位姿数据
        pathGT.header.stamp = odomGT.header.stamp;  // 路径时间戳同步
        pathGT.poses.push_back(poseGT);  // 将当前位姿添加到路径列表
        pubPathGT.publish(pathGT);       // 发布真值路径消息

        // 读取激光雷达点云数据（.bin格式）
        std::stringstream lidar_data_path;
        lidar_data_path << dataset_folder << "velodyne/sequences/" + sequence_number + "/velodyne/" 
                        << std::setfill('0') << std::setw(6) << line_num << ".bin";  // 激光雷达数据路径（6位补零）
        std::vector<float> lidar_data = read_lidar_data(lidar_data_path.str());  // 调用函数读取数据
        std::cout << "totally " << lidar_data.size() / 4.0 << " points in this lidar frame \n";  // 输出点云数量（每4个float为一个点）

        std::vector<Eigen::Vector3d> lidar_points;
        std::vector<float> lidar_intensities;
        // 转换为PCL点云格式（PointXYZI：x,y,z,强度）
        pcl::PointCloud<pcl::PointXYZI> laser_cloud;
        for (std::size_t i = 0; i < lidar_data.size(); i += 4)
        {
            lidar_points.emplace_back(lidar_data[i], lidar_data[i + 1], lidar_data[i + 2]);
            lidar_intensities.push_back(lidar_data[i + 3]);

            pcl::PointXYZI point;
            point.x = lidar_data[i];      // x坐标
            point.y = lidar_data[i + 1];  // y坐标
            point.z = lidar_data[i + 2];  // z坐标
            point.intensity = lidar_data[i + 3];  // 回波强度
            laser_cloud.push_back(point);  // 添加到点云
        }

        // 转换为ROS点云消息并发布
        sensor_msgs::PointCloud2 laser_cloud_msg;
        pcl::toROSMsg(laser_cloud, laser_cloud_msg);  // PCL点云转ROS消息
        laser_cloud_msg.header.stamp = ros::Time().fromSec(timestamp);  // 同步时间戳
        // -------------ubuntu 18.04（坐标系ID带前斜杠）
        // laser_cloud_msg.header.frame_id = "/camera_init";
        // -------------ubuntu 20.04（坐标系ID不带前斜杠）
        laser_cloud_msg.header.frame_id = "camera_init";  // 设置参考坐标系
        pub_laser_cloud.publish(laser_cloud_msg);  // 发布激光点云消息

        // 转换左右图像为ROS消息并发布
        sensor_msgs::ImagePtr image_left_msg = cv_bridge::CvImage(laser_cloud_msg.header, "mono8", left_image).toImageMsg();  // 左图转ROS消息（单通道灰度）
        sensor_msgs::ImagePtr image_right_msg = cv_bridge::CvImage(laser_cloud_msg.header, "mono8", right_image).toImageMsg();  // 右图转ROS消息
        pub_image_left.publish(image_left_msg);  // 发布左图像消息
        pub_image_right.publish(image_right_msg);  // 发布右图像消息

        // 若需要输出rosbag，将消息写入bag文件
        if (to_bag)
        {
            bag_out.write("/image_left", ros::Time::now(), image_left_msg);       // 左图像
            bag_out.write("/image_right", ros::Time::now(), image_right_msg);     // 右图像
            bag_out.write("/velodyne_points", ros::Time::now(), laser_cloud_msg);  // 激光点云
            bag_out.write("/path_gt", ros::Time::now(), pathGT);                  // 真值路径
            bag_out.write("/odometry_gt", ros::Time::now(), odomGT);              // 真值里程计
        }

        line_num++;  // 帧计数递增
        r.sleep();   // 按频率休眠（控制发布节奏）
    }

    bag_out.close();  // 关闭rosbag文件（若已打开）
    std::cout << "Done \n";  // 处理完成提示

    return 0;
}