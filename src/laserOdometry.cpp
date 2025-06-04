// 本代码是以下论文算法的改进实现：
//   J. Zhang和S. Singh的LOAM算法（实时激光雷达里程计与建图）
//   发表于2014年机器人科学研究会议（RSS），加州伯克利

// 修改者: 覃桐               qintonguav@gmail.com
//         曹绍祖            saozu.cao@connect.ust.hk

// 版权声明（2013年，卡内基梅隆大学Ji Zhang；2016年西南研究院贡献）
// 保留所有权利。
//  Redistribution和使用需遵守以下条件：
// 1. 源代码再发布需保留上述版权声明、本条件列表和免责声明。
// 2. 二进制形式再发布需在文档和/或发布材料中复制上述版权声明、本条件列表和免责声明。
// 3. 未经版权方书面许可，不得使用版权方或贡献者名称推广衍生产品。
// 本软件按"原样"提供，无任何明示或暗示的担保（包括适销性和特定用途适用性）。
// 版权方和贡献者不对任何直接、间接、偶发、特殊、惩罚性或后果性损害负责。

// 包含数学库（用于三角函数等计算）
#include <cmath>
// 导航消息类型（里程计、路径等）
#include <nav_msgs/Odometry.h>
// 路径消息类型（存储位姿序列）
#include <nav_msgs/Path.h>
// 位姿消息类型（带时间戳的位姿）
#include <geometry_msgs/PoseStamped.h>
// PCL点云库核心头文件
#include <pcl/point_cloud.h>
// PCL点类型定义（如PointXYZI）
#include <pcl/point_types.h>
// 体素滤波器（用于点云降采样）
#include <pcl/filters/voxel_grid.h>
// PCL KD树（用于最近邻搜索）
#include <pcl/kdtree/kdtree_flann.h>
// PCL与ROS消息转换工具
#include <pcl_conversions/pcl_conversions.h>
// ROS核心头文件
#include <ros/ros.h>
// IMU传感器消息类型
#include <sensor_msgs/Imu.h>
// 点云传感器消息类型
#include <sensor_msgs/PointCloud2.h>
// TF坐标转换数据类型（四元数、变换等）
#include <tf/transform_datatypes.h>
// TF坐标变换广播器
#include <tf/transform_broadcaster.h>
// Eigen矩阵运算库（密集矩阵操作）
#include <eigen3/Eigen/Dense>
// C++多线程，互斥锁提供了排他性的、非递归的所有权语义
// https://zhuanlan.zhihu.com/p/598993031
// lock()，调用线程将锁住该互斥量。线程调用该函数会发生下面 3 种情况：
// (1). 如果该互斥量当前没 有被锁住，则调用线程将该互斥量锁住，直到调用 unlock之前，该线程一直拥有该锁。
// (2). 如果当 前互斥量被其他线程锁住，则当前的调用线程被阻塞住。
// (3). 如果当前互斥量被当前调用线程锁 住，则会产生死锁(deadlock)。
// unlock()， 解锁，释放对互斥量的所有权。
// C++11提供如下4种语义的互斥量（mutex） ：

// std::mutex，独占的互斥量，不能递归使用。
// std::time_mutex，带超时的独占互斥量，不能递归使用。
// std::recursive_mutex，递归互斥量，不带超时功能。
// std::recursive_timed_mutex，带超时的递归互斥量。
#include <mutex>
// C++队列容器（用于缓存消息）
#include <queue>
// LOAM算法公共头文件（定义点类型等）
#include "aloam_velodyne/common.h"
// 时间统计工具（用于性能分析）
#include "aloam_velodyne/tic_toc.h"
// 激光雷达残差因子定义（用于Ceres优化）
#include "lidarFactor.hpp"

// Lidar Odometry线程估计的frame在world坐标系的位姿P，
// Transformation from current frame to world frame（间隔：skipFrameNum）

// 是否启用点云畸变校正（0=禁用，1=启用）
#define DISTORTION 0

// 角点/平面点匹配计数（用于优化统计）
int corner_correspondence = 0, plane_correspondence = 0;

// 激光雷达扫描周期（10Hz对应0.1秒）
constexpr double SCAN_PERIOD = 0.1;
// 最近邻搜索距离平方阈值（用于特征匹配筛选）
constexpr double DISTANCE_SQ_THRESHOLD = 25;
// 相邻扫描线范围（用于特征点搜索范围控制）
constexpr double NEARBY_SCAN = 2.5;

// 建图跳过帧数（控制建图频率，如skip=2则5Hz）
int skipFrameNum = 5;
// 系统初始化标志（控制初始状态流程）
bool systemInited = false;

// 各特征点云消息时间戳（用于同步校验）
double timeCornerPointsSharp = 0;
double timeCornerPointsLessSharp = 0;
double timeSurfPointsFlat = 0;
double timeSurfPointsLessFlat = 0;
double timeLaserCloudFullRes = 0;

// 前一帧角点特征KD树（用于最近邻搜索）
pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeCornerLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());
// 前一帧平面特征KD树（用于最近邻搜索）
pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeSurfLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());

// 当前帧尖锐角点（用于优化的强特征）
pcl::PointCloud<PointType>::Ptr cornerPointsSharp(new pcl::PointCloud<PointType>());
// 当前帧次尖锐角点（用于地图构建）
pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp(new pcl::PointCloud<PointType>());
// 当前帧平坦平面点（用于优化的强特征）
pcl::PointCloud<PointType>::Ptr surfPointsFlat(new pcl::PointCloud<PointType>());
// 当前帧次平坦平面点（用于地图构建）
pcl::PointCloud<PointType>::Ptr surfPointsLessFlat(new pcl::PointCloud<PointType>());

// 前一帧角点特征点云（用于匹配）
pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(new pcl::PointCloud<PointType>());
// 前一帧平面特征点云（用于匹配）
pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(new pcl::PointCloud<PointType>());
// 全分辨率点云（用于输出）
pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());

// 前一帧角点/平面点数量（用于匹配范围控制）
int laserCloudCornerLastNum = 0;
int laserCloudSurfLastNum = 0;

// 当前帧到世界坐标系的位姿（四元数+平移向量）
Eigen::Quaterniond q_w_curr(1, 0, 0, 0);
Eigen::Vector3d t_w_curr(0, 0, 0);

// Ceres优化变量（当前帧到前一帧的位姿）
double para_q[4] = {0, 0, 0, 1};  // 四元数（x,y,z,w）
double para_t[3] = {0, 0, 0};     // 平移向量（x,y,z）

// Eigen映射（将数组转换为四元数/向量对象）
Eigen::Map<Eigen::Quaterniond> q_last_curr(para_q);  // 前一帧到当前帧的旋转
Eigen::Map<Eigen::Vector3d> t_last_curr(para_t);      // 前一帧到当前帧的平移

// 各特征点云消息队列（用于多线程同步）
std::queue<sensor_msgs::PointCloud2ConstPtr> cornerSharpBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> cornerLessSharpBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfFlatBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> surfLessFlatBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullPointsBuf;
// 队列互斥锁（保证多线程访问安全）
std::mutex mBuf;

// 将点转换到扫描起始时刻（处理运动畸变）
// 参数：pi-输入点（带时间戳），po-输出点（校正后）
void TransformToStart(PointType const *const pi, PointType *const po)
{
    // 计算插值比例（扫描周期内的相对时间）
    double s;
    if (DISTORTION)  // 启用畸变校正时
        s = (pi->intensity - int(pi->intensity)) / SCAN_PERIOD;  // 利用intensity存储时间偏移
    else              // 禁用时使用全周期
        s = 1.0;

    // 插值计算起始时刻的位姿（前一帧到当前帧的位姿按时间s插值）
    Eigen::Quaterniond q_point_last = Eigen::Quaterniond::Identity().slerp(s, q_last_curr);  // 四元数球面线性插值
    Eigen::Vector3d t_point_last = s * t_last_curr;  // 平移向量线性插值

    // 将当前点转换到扫描起始时刻的坐标系
    Eigen::Vector3d point(pi->x, pi->y, pi->z);
    Eigen::Vector3d un_point = q_point_last * point + t_point_last;  // 旋转+平移变换

    // 输出校正后的点坐标（保留强度信息）
    po->x = un_point.x();
    po->y = un_point.y();
    po->z = un_point.z();
    po->intensity = pi->intensity;
}

// 将点转换到扫描结束时刻（用于输出全分辨率点云）
// 参数：pi-输入点（带时间戳），po-输出点（校正后）
void TransformToEnd(PointType const *const pi, PointType *const po)
{
    // 先校正到扫描起始时刻
    pcl::PointXYZI un_point_tmp;
    TransformToStart(pi, &un_point_tmp);

    // 将起始时刻点转换到当前帧结束时刻坐标系
    Eigen::Vector3d un_point(un_point_tmp.x, un_point_tmp.y, un_point_tmp.z);
    Eigen::Vector3d point_end = q_last_curr.inverse() * (un_point - t_last_curr);  // 逆变换

    // 输出结束时刻点坐标（去除时间戳信息）
    po->x = point_end.x();
    po->y = point_end.y();
    po->z = point_end.z();
    po->intensity = int(pi->intensity);  // 仅保留扫描线编号
}

// 尖锐角点云消息回调（存入队列）
void laserCloudSharpHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsSharp2)
{
    mBuf.lock();          // 加锁保证队列操作原子性
    cornerSharpBuf.push(cornerPointsSharp2);  // 消息入队
    mBuf.unlock();        // 解锁
}

// 次尖锐角点云消息回调（存入队列）
void laserCloudLessSharpHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsLessSharp2)
{
    mBuf.lock();
    cornerLessSharpBuf.push(cornerPointsLessSharp2);
    mBuf.unlock();
}

// 平坦平面点云消息回调（存入队列）
void laserCloudFlatHandler(const sensor_msgs::PointCloud2ConstPtr &surfPointsFlat2)
{
    mBuf.lock();
    surfFlatBuf.push(surfPointsFlat2);
    mBuf.unlock();
}

// 次平坦平面点云消息回调（存入队列）
void laserCloudLessFlatHandler(const sensor_msgs::PointCloud2ConstPtr &surfPointsLessFlat2)
{
    mBuf.lock();
    surfLessFlatBuf.push(surfPointsLessFlat2);
    mBuf.unlock();
}

// 全分辨率点云消息回调（存入队列）
void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2)
{
    mBuf.lock();
    fullPointsBuf.push(laserCloudFullRes2);
    mBuf.unlock();
}

int main(int argc, char **argv)
{
    // 初始化ROS节点（节点名：laserOdometry）
    ros::init(argc, argv, "laserOdometry");
    ros::NodeHandle nh;

    // 从参数服务器获取建图跳过帧数（默认2）
    nh.param<int>("mapping_skip_frame", skipFrameNum, 2);

    // 输出建图频率（10Hz / skipFrameNum）
    printf("Mapping %d Hz \n", 10 / skipFrameNum);

    // 订阅各特征点云话题（尖锐角点、次尖锐角点、平坦平面点、次平坦平面点、全分辨率点云）
    ros::Subscriber subCornerPointsSharp = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 100, laserCloudSharpHandler);
    ros::Subscriber subCornerPointsLessSharp = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100, laserCloudLessSharpHandler);
    ros::Subscriber subSurfPointsFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100, laserCloudFlatHandler);
    ros::Subscriber subSurfPointsLessFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100, laserCloudLessFlatHandler);
    ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100, laserCloudFullResHandler);

    // 发布前一帧角点/平面点云、全分辨率点云、里程计、路径话题
    ros::Publisher pubLaserCloudCornerLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 100);
    ros::Publisher pubLaserCloudSurfLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100);
    ros::Publisher pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 100);
    ros::Publisher pubLaserOdometry = nh.advertise<nav_msgs::Odometry>("/laser_odom_to_init", 100);
    ros::Publisher pubLaserPath = nh.advertise<nav_msgs::Path>("/laser_odom_path", 100);

    // 初始化路径消息（用于存储历史位姿）
    nav_msgs::Path laserPath;

    // 帧计数（控制建图频率）
    int frameCount = 0;
    // ROS循环频率（100Hz）
    ros::Rate rate(100);

    // 主循环（ROS运行时持续执行）
    while (ros::ok())
    {
        // 处理ROS消息回调（触发各特征点云的队列存储）
        ros::spinOnce();

        // 检查各特征点云队列是否有数据（同步所有输入消息）
        if (!cornerSharpBuf.empty() && !cornerLessSharpBuf.empty() &&
            !surfFlatBuf.empty() && !surfLessFlatBuf.empty() &&
            !fullPointsBuf.empty())
        {
            // 获取各消息时间戳（校验同步性）
            timeCornerPointsSharp = cornerSharpBuf.front()->header.stamp.toSec();
            timeCornerPointsLessSharp = cornerLessSharpBuf.front()->header.stamp.toSec();
            timeSurfPointsFlat = surfFlatBuf.front()->header.stamp.toSec();
            timeSurfPointsLessFlat = surfLessFlatBuf.front()->header.stamp.toSec();
            timeLaserCloudFullRes = fullPointsBuf.front()->header.stamp.toSec();

            // 校验所有消息时间戳是否一致（防止数据不同步）
            if (timeCornerPointsSharp != timeLaserCloudFullRes ||
                timeCornerPointsLessSharp != timeLaserCloudFullRes ||
                timeSurfPointsFlat != timeLaserCloudFullRes ||
                timeSurfPointsLessFlat != timeLaserCloudFullRes)
            {
                printf("unsync messeage!");
                ROS_BREAK();  // 不同步时终止程序
            }

            // 加锁并取出各点云数据（转换为PCL格式）
            mBuf.lock();
            cornerPointsSharp->clear();
            pcl::fromROSMsg(*cornerSharpBuf.front(), *cornerPointsSharp);  // ROS消息转PCL点云
            cornerSharpBuf.pop();  // 出队

            cornerPointsLessSharp->clear();
            pcl::fromROSMsg(*cornerLessSharpBuf.front(), *cornerPointsLessSharp);
            cornerLessSharpBuf.pop();

            surfPointsFlat->clear();
            pcl::fromROSMsg(*surfFlatBuf.front(), *surfPointsFlat);
            surfFlatBuf.pop();

            surfPointsLessFlat->clear();
            pcl::fromROSMsg(*surfLessFlatBuf.front(), *surfPointsLessFlat);
            surfLessFlatBuf.pop();

            laserCloudFullRes->clear();
            pcl::fromROSMsg(*fullPointsBuf.front(), *laserCloudFullRes);
            fullPointsBuf.pop();
            mBuf.unlock();  // 解锁释放互斥锁

            TicToc t_whole;  // 统计整体处理时间

            // 系统初始化阶段（第一帧）
            if (!systemInited)
            {
                systemInited = true;  // 标记初始化完成
                std::cout << "Initialization finished \n";
            }
            else  // 正常运行阶段（已有前一帧数据）
            {
                int cornerPointsSharpNum = cornerPointsSharp->points.size();  // 当前帧尖锐角点数量（强特征点）
                int surfPointsFlatNum = surfPointsFlat->points.size();        // 当前帧平坦平面点数量（强特征点）

                TicToc t_opt;  // 统计优化过程耗时
                for (size_t opti_counter = 0; opti_counter < 2; ++opti_counter)  // 优化迭代两次（保证收敛性）
                {
                    corner_correspondence = 0;  // 角点匹配计数归零
                    plane_correspondence = 0;   // 平面点匹配计数归零
                    // ceres::LossFunction *loss_function = NULL;
                    // 定义损失函数（Huber损失降低异常值影响）
                    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
                    // 定义四元数参数化（保证旋转的单位四元数约束）
                    ceres::LocalParameterization *q_parameterization =
                        new ceres::EigenQuaternionParameterization();
                    // 初始化Ceres问题选项
                    ceres::Problem::Options problem_options;

                    ceres::Problem problem(problem_options);  // 创建Ceres优化问题实例
                    // 添加四元数参数块（使用四元数参数化）
                    problem.AddParameterBlock(para_q, 4, q_parameterization);
                    // 添加平移向量参数块（无约束）
                    problem.AddParameterBlock(para_t, 3);

                    pcl::PointXYZI pointSel;  // 存储待匹配的当前帧特征点（校正后）
                    std::vector<int> pointSearchInd;  // 存储搜索到的近邻点索引
                    std::vector<float> pointSearchSqDis;  // 存储近邻点距离平方

                    TicToc t_data;  // 统计数据关联耗时
                    // 处理角点特征匹配（构建角点约束）
                    for (int i = 0; i < cornerPointsSharpNum; ++i)  // 遍历当前帧所有尖锐角点
                    {
                        // 将当前角点转换到扫描起始时刻（校正运动畸变）
                        TransformToStart(&(cornerPointsSharp->points[i]), &pointSel);
                        // 在角点特征的KD树中搜索最近邻点（前一帧角点）
                        kdtreeCornerLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);

                        int closestPointInd = -1, minPointInd2 = -1;  // 最近点索引和次近点索引
                        if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD)  // 距离满足阈值（有效匹配）
                        {
                            closestPointInd = pointSearchInd[0];  // 获取最近点索引
                            // 获取最近点的扫描线编号（intensity字段存储）
                            int closestPointScanID = int(laserCloudCornerLast->points[closestPointInd].intensity);

                            double minPointSqDis2 = DISTANCE_SQ_THRESHOLD;  // 次近点距离平方阈值
                            // 向扫描线递增方向搜索次近点（同一区域相邻扫描线）
                            for (int j = closestPointInd + 1; j < (int)laserCloudCornerLast->points.size(); ++j)
                            {
                                // 跳过同一扫描线的点（需找相邻扫描线）
                                if (int(laserCloudCornerLast->points[j].intensity) <= closestPointScanID)
                                    continue;
                                // 超出相邻扫描线范围（停止搜索）
                                if (int(laserCloudCornerLast->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                                    break;
                                // 计算当前点与搜索点的距离平方
                                double pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                                                        (laserCloudCornerLast->points[j].x - pointSel.x) +
                                                    (laserCloudCornerLast->points[j].y - pointSel.y) *
                                                        (laserCloudCornerLast->points[j].y - pointSel.y) +
                                                    (laserCloudCornerLast->points[j].z - pointSel.z) *
                                                        (laserCloudCornerLast->points[j].z - pointSel.z);
                                // 更新次近点（距离更小则记录）
                                if (pointSqDis < minPointSqDis2)
                                {
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                            }

                            // 向扫描线递减方向搜索次近点（同一区域相邻扫描线）
                            for (int j = closestPointInd - 1; j >= 0; --j)
                            {
                                // 跳过同一扫描线的点（需找相邻扫描线）
                                if (int(laserCloudCornerLast->points[j].intensity) >= closestPointScanID)
                                    continue;
                                // 超出相邻扫描线范围（停止搜索）
                                if (int(laserCloudCornerLast->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                                    break;
                                // 计算当前点与搜索点的距离平方
                                double pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                                                        (laserCloudCornerLast->points[j].x - pointSel.x) +
                                                    (laserCloudCornerLast->points[j].y - pointSel.y) *
                                                        (laserCloudCornerLast->points[j].y - pointSel.y) +
                                                    (laserCloudCornerLast->points[j].z - pointSel.z) *
                                                        (laserCloudCornerLast->points[j].z - pointSel.z);
                                // 更新次近点（距离更小则记录）
                                if (pointSqDis < minPointSqDis2)
                                {
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                            }
                        }
                        // 若找到有效次近点（构建角点到直线的约束）
                        if (minPointInd2 >= 0) // 最近点和次近点均有效
                        {
                            // 当前帧角点坐标（转换到扫描起始时刻）
                            Eigen::Vector3d curr_point(cornerPointsSharp->points[i].x,
                                                       cornerPointsSharp->points[i].y,
                                                       cornerPointsSharp->points[i].z);
                            // 前一帧角点特征中的最近点坐标（用于构建直线）
                            Eigen::Vector3d last_point_a(laserCloudCornerLast->points[closestPointInd].x,
                                                         laserCloudCornerLast->points[closestPointInd].y,
                                                         laserCloudCornerLast->points[closestPointInd].z);
                            // 前一帧角点特征中的次近点坐标（与最近点共同构成直线）
                            Eigen::Vector3d last_point_b(laserCloudCornerLast->points[minPointInd2].x,
                                                         laserCloudCornerLast->points[minPointInd2].y,
                                                         laserCloudCornerLast->points[minPointInd2].z);

                            // 计算当前点的时间插值比例（用于运动畸变校正）
                            double s;
                            if (DISTORTION)  // 启用畸变校正时
                                s = (cornerPointsSharp->points[i].intensity - int(cornerPointsSharp->points[i].intensity)) / SCAN_PERIOD;  // 从intensity字段提取时间偏移（小数部分）
                            else             // 禁用时使用全周期（无畸变校正）
                                s = 1.0;

                            // 创建角点残差因子（点到直线的约束）
                            ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, last_point_a, last_point_b, s);
                            // 向优化问题添加残差块（使用Huber损失函数抑制异常值）
                            problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                            corner_correspondence++;  // 有效角点约束计数加一
                        }
                    }

                    // 处理平面特征匹配（构建平面约束）
                    for (int i = 0; i < surfPointsFlatNum; ++i)  // 遍历当前帧所有平坦平面点（强特征点）
                    {
                        // 将当前平面点转换到扫描起始时刻（校正运动畸变）
                        TransformToStart(&(surfPointsFlat->points[i]), &pointSel);
                        // 在平面特征的KD树中搜索最近邻点（前一帧平面点）
                        kdtreeSurfLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);

                        int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;  // 最近点、次近点、第三近点索引
                        if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD)  // 距离满足阈值（有效匹配）
                        {
                            closestPointInd = pointSearchInd[0];  // 获取最近点索引

                            // 获取最近点的扫描线编号（intensity字段存储）
                            int closestPointScanID = int(laserCloudSurfLast->points[closestPointInd].intensity);
                            double minPointSqDis2 = DISTANCE_SQ_THRESHOLD,  // 次近点距离平方阈值
                                   minPointSqDis3 = DISTANCE_SQ_THRESHOLD;  // 第三近点距离平方阈值

                            // 向扫描线递增方向搜索次近点和第三近点（同一区域相邻扫描线）
                            for (int j = closestPointInd + 1; j < (int)laserCloudSurfLast->points.size(); ++j)
                            {
                                // 超出相邻扫描线范围（停止搜索）
                                if (int(laserCloudSurfLast->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                                    break;

                                // 计算当前点与搜索点的距离平方
                                double pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                                                        (laserCloudSurfLast->points[j].x - pointSel.x) +
                                                    (laserCloudSurfLast->points[j].y - pointSel.y) *
                                                        (laserCloudSurfLast->points[j].y - pointSel.y) +
                                                    (laserCloudSurfLast->points[j].z - pointSel.z) *
                                                        (laserCloudSurfLast->points[j].z - pointSel.z);

                                // 若为同一/更低扫描线且距离更小（更新次近点）
                                if (int(laserCloudSurfLast->points[j].intensity) <= closestPointScanID && pointSqDis < minPointSqDis2)
                                {
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                                // 若为更高扫描线且距离更小（更新第三近点）
                                else if (int(laserCloudSurfLast->points[j].intensity) > closestPointScanID && pointSqDis < minPointSqDis3)
                                {
                                    minPointSqDis3 = pointSqDis;
                                    minPointInd3 = j;
                                }
                            }

                            // 向扫描线递减方向搜索次近点和第三近点（同一区域相邻扫描线）
                            for (int j = closestPointInd - 1; j >= 0; --j)
                            {
                                // 超出相邻扫描线范围（停止搜索）
                                if (int(laserCloudSurfLast->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                                    break;

                                // 计算当前点与搜索点的距离平方
                                double pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                                                        (laserCloudSurfLast->points[j].x - pointSel.x) +
                                                    (laserCloudSurfLast->points[j].y - pointSel.y) *
                                                        (laserCloudSurfLast->points[j].y - pointSel.y) +
                                                    (laserCloudSurfLast->points[j].z - pointSel.z) *
                                                        (laserCloudSurfLast->points[j].z - pointSel.z);

                                // 若为同一/更高扫描线且距离更小（更新次近点）
                                if (int(laserCloudSurfLast->points[j].intensity) >= closestPointScanID && pointSqDis < minPointSqDis2)
                                {
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                                // 若为更低扫描线且距离更小（更新第三近点）
                                else if (int(laserCloudSurfLast->points[j].intensity) < closestPointScanID && pointSqDis < minPointSqDis3)
                                {
                                    minPointSqDis3 = pointSqDis;
                                    minPointInd3 = j;
                                }
                            }

                            // 若找到有效次近点和第三近点（构建平面到点的约束）
                            if (minPointInd2 >= 0 && minPointInd3 >= 0)
                            {
                                // 当前帧平面点的三维坐标（原始点云数据）
                                Eigen::Vector3d curr_point(surfPointsFlat->points[i].x,
                                                           surfPointsFlat->points[i].y,
                                                           surfPointsFlat->points[i].z);
                                // 前一帧平面特征中的最近点（构建平面的基准点）
                                Eigen::Vector3d last_point_a(laserCloudSurfLast->points[closestPointInd].x,
                                                             laserCloudSurfLast->points[closestPointInd].y,
                                                             laserCloudSurfLast->points[closestPointInd].z);
                                // 前一帧平面特征中的次近点（与最近点共同构建平面）
                                Eigen::Vector3d last_point_b(laserCloudSurfLast->points[minPointInd2].x,
                                                             laserCloudSurfLast->points[minPointInd2].y,
                                                             laserCloudSurfLast->points[minPointInd2].z);
                                // 前一帧平面特征中的第三近点（与前两点共同确定平面）
                                Eigen::Vector3d last_point_c(laserCloudSurfLast->points[minPointInd3].x,
                                                             laserCloudSurfLast->points[minPointInd3].y,
                                                             laserCloudSurfLast->points[minPointInd3].z);

                                // 计算当前点的时间插值比例（用于运动畸变校正）
                                double s;
                                if (DISTORTION)  // 启用畸变校正时（从intensity字段提取时间偏移）
                                    s = (surfPointsFlat->points[i].intensity - int(surfPointsFlat->points[i].intensity)) / SCAN_PERIOD;
                                else             // 禁用时使用全周期（无畸变校正）
                                    s = 1.0;

                                // 创建平面残差因子（点到平面的约束）
                                ceres::CostFunction *cost_function = LidarPlaneFactor::Create(curr_point, last_point_a, last_point_b, last_point_c, s);
                                // 向优化问题添加残差块（使用Huber损失函数抑制异常值）
                                problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                                plane_correspondence++;  // 有效平面约束计数加一
                            }
                        }
                    }

                    // printf("coner_correspondance %d, plane_correspondence %d \n", corner_correspondence, plane_correspondence);
                    // 输出数据关联（特征匹配）耗时（单位：毫秒）
                    printf("data association time %f ms \n", t_data.toc());

                    // 检查有效匹配总数（角点+平面点）是否不足10个（阈值经验设定）
                    if ((corner_correspondence + plane_correspondence) < 10)
                    {
                        // 匹配点过少时输出警告（可能影响优化收敛性）
                        printf("less correspondence! *************************************************\n");
                    }

                    TicToc t_solver;  // 初始化时间统计对象（用于统计优化求解耗时）
                    ceres::Solver::Options options;  // 定义Ceres求解器配置选项
                    // 设置线性求解器类型（DENSE_QR：适用于小规模稠密矩阵的QR分解）
                    options.linear_solver_type = ceres::DENSE_QR;
                    // 设置最大迭代次数（4次平衡计算效率与收敛性）
                    options.max_num_iterations = 4;
                    // 禁用优化过程输出（避免日志冗余）
                    options.minimizer_progress_to_stdout = false;
                    ceres::Solver::Summary summary;  // 定义优化结果摘要（存储优化统计信息）
                    // 执行Ceres优化（使用配置选项、待优化问题、存储结果摘要）
                    ceres::Solve(options, &problem, &summary);
                    // 输出优化求解耗时（单位：毫秒）
                    printf("solver time %f ms \n", t_solver.toc());
                }
                printf("optimization twice time %f \n", t_opt.toc());

                t_w_curr = t_w_curr + q_w_curr * t_last_curr;
                q_w_curr = q_w_curr * q_last_curr;
            }

            TicToc t_pub;

            // publish odometry
            nav_msgs::Odometry laserOdometry;
            // -------------ubuntu 18.04
            // laserOdometry.header.frame_id = "/camera_init";
            // -------------ubuntu 20.04
            // 设置激光里程计消息的坐标系信息：父坐标系为"camera_init"（初始相机坐标系），子坐标系为当前激光里程计坐标系
            laserOdometry.header.frame_id = "camera_init";
            laserOdometry.child_frame_id = "/laser_odom";
            // 设置里程计消息的时间戳（与次平坦平面点云时间戳同步）
            laserOdometry.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
            // 将当前帧到世界坐标系的四元数姿态写入里程计消息（方向部分）
            laserOdometry.pose.pose.orientation.x = q_w_curr.x();
            laserOdometry.pose.pose.orientation.y = q_w_curr.y();
            laserOdometry.pose.pose.orientation.z = q_w_curr.z();
            laserOdometry.pose.pose.orientation.w = q_w_curr.w();
            // 将当前帧到世界坐标系的平移向量写入里程计消息（位置部分）
            laserOdometry.pose.pose.position.x = t_w_curr.x();
            laserOdometry.pose.pose.position.y = t_w_curr.y();
            laserOdometry.pose.pose.position.z = t_w_curr.z();
            // 发布激光里程计消息到"/laser_odom_to_init"话题（用于外部节点订阅）
            pubLaserOdometry.publish(laserOdometry);

            // 创建位姿消息对象（用于路径记录）
            geometry_msgs::PoseStamped laserPose;
            // 继承里程计消息的头部信息（坐标系和时间戳）
            laserPose.header = laserOdometry.header;
            // 继承里程计消息的位姿数据（位置+方向）
            laserPose.pose = laserOdometry.pose.pose;
            // 更新路径消息的时间戳（与当前里程计时间同步）
            laserPath.header.stamp = laserOdometry.header.stamp;
            // 将当前位姿添加到路径消息的位姿列表中（用于可视化历史轨迹）
            laserPath.poses.push_back(laserPose);
            // -------------ubuntu 18.04
            // laserPath.header.frame_id = "/camera_init";
            // -------------ubuntu 20.04
            laserPath.header.frame_id = "camera_init";
            pubLaserPath.publish(laserPath);

            // transform corner features and plane features to the scan end point
            // 条件恒为假（0），此代码块永远不会执行（可能为开发中暂存的备用逻辑）
            if (0)
            {
                // 获取次尖锐角点点云数量（用于遍历）
                int cornerPointsLessSharpNum = cornerPointsLessSharp->points.size();
                // 遍历次尖锐角点，将每个点转换到扫描结束时刻坐标系
                for (int i = 0; i < cornerPointsLessSharpNum; i++)
                {
                    // 调用TransformToEnd函数：将点从扫描起始时刻转换到结束时刻（校正运动畸变）
                    TransformToEnd(&cornerPointsLessSharp->points[i], &cornerPointsLessSharp->points[i]);
                }

                // 获取次平坦平面点点云数量（用于遍历）
                int surfPointsLessFlatNum = surfPointsLessFlat->points.size();
                // 遍历次平坦平面点，将每个点转换到扫描结束时刻坐标系
                for (int i = 0; i < surfPointsLessFlatNum; i++)
                {
                    // 调用TransformToEnd函数：校正点坐标到扫描结束时刻
                    TransformToEnd(&surfPointsLessFlat->points[i], &surfPointsLessFlat->points[i]);
                }

                // 获取全分辨率点云数量（用于遍历）
                int laserCloudFullResNum = laserCloudFullRes->points.size();
                // 遍历全分辨率点云，将每个点转换到扫描结束时刻坐标系
                for (int i = 0; i < laserCloudFullResNum; i++)
                {
                    // 调用TransformToEnd函数：校正点坐标到扫描结束时刻（输出最终点云）
                    TransformToEnd(&laserCloudFullRes->points[i], &laserCloudFullRes->points[i]);
                }
            }

            // 交换当前帧次尖锐角点云与前一帧角点云指针（更新前一帧数据）
            pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsLessSharp;
            cornerPointsLessSharp = laserCloudCornerLast;  // 前一帧角点云指向当前帧的次尖锐角点
            laserCloudCornerLast = laserCloudTemp;         // 当前帧次尖锐角点作为下一帧的前一帧角点

            // 交换当前帧次平坦平面点云与前一帧平面点云指针（逻辑同上）
            laserCloudTemp = surfPointsLessFlat;
            surfPointsLessFlat = laserCloudSurfLast;       // 前一帧平面云指向当前帧的次平坦平面点
            laserCloudSurfLast = laserCloudTemp;           // 当前帧次平坦平面点作为下一帧的前一帧平面点

            // 记录更新后的前一帧角点/平面点云数量（用于调试和流程控制）
            laserCloudCornerLastNum = laserCloudCornerLast->points.size();
            laserCloudSurfLastNum = laserCloudSurfLast->points.size();

            // 输出前一帧角点/平面点云数量（监控数据状态）
            std::cout << "the size of corner last is " << laserCloudCornerLastNum << ", and the size of surf last is " << laserCloudSurfLastNum << '\n';

            // 若前一帧角点云非空：设置KD树输入点云（用于下一帧特征匹配的最近邻搜索）
            if (!laserCloudCornerLast->empty())
            {
                /* code */
                kdtreeCornerLast->setInputCloud(laserCloudCornerLast);  // 更新角点特征KD树的输入
            }
            // 若前一帧平面云非空：设置KD树输入点云（逻辑同上）
            if (!laserCloudSurfLast->empty())
            {
                /* code */
                kdtreeSurfLast->setInputCloud(laserCloudSurfLast);      // 更新平面特征KD树的输入
            }
            
            // （被注释的原始代码：直接设置KD树输入，未做空点云检查）
            // kdtreeCornerLast->setInputCloud(laserCloudCornerLast);
            // kdtreeSurfLast->setInputCloud(laserCloudSurfLast);

            // 控制发布频率（每跳过skipFrameNum帧发布一次点云）
            if (frameCount % skipFrameNum == 0)
            {
                frameCount = 0;  // 重置帧计数（避免溢出）

                // 发布前一帧角点云（转换为ROS消息）
                sensor_msgs::PointCloud2 laserCloudCornerLast2;
                pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLast2);  // PCL点云转ROS消息
                laserCloudCornerLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);  // 同步时间戳
                // 根据Ubuntu版本设置坐标系（18.04与20.04的差异）
                // -------------ubuntu 18.04
                // laserCloudCornerLast2.header.frame_id = "/camera";
                // -------------ubuntu 20.04
                laserCloudCornerLast2.header.frame_id = "camera";  // 设置坐标系ID
                pubLaserCloudCornerLast.publish(laserCloudCornerLast2);  // 发布角点云消息

                // 发布前一帧平面云（逻辑同上）
                sensor_msgs::PointCloud2 laserCloudSurfLast2;
                pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLast2);
                laserCloudSurfLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                // -------------ubuntu 18.04
                // laserCloudSurfLast2.header.frame_id = "/camera";
                // -------------ubuntu 20.04
                laserCloudSurfLast2.header.frame_id = "camera";
                pubLaserCloudSurfLast.publish(laserCloudSurfLast2);  // 发布平面云消息

                // 发布全分辨率点云（逻辑同上）
                sensor_msgs::PointCloud2 laserCloudFullRes3;
                pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
                laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                // -------------ubuntu 18.04
                // laserCloudFullRes3.header.frame_id = "/camera";
                // -------------ubuntu 20.04
                laserCloudFullRes3.header.frame_id = "camera";
                pubLaserCloudFullRes.publish(laserCloudFullRes3);  // 发布全分辨率点云消息
            }
            // 输出发布耗时（单位：毫秒）
            printf("publication time %f ms \n", t_pub.toc());
            // 输出整体激光里程计处理耗时（单位：毫秒）
            printf("whole laserOdometry time %f ms \n \n", t_whole.toc());
            // 若整体处理时间超过100ms，输出警告（提示性能瓶颈）
            if (t_whole.toc() > 100)
                ROS_WARN("odometry process over 100ms");

            frameCount++;  // 帧计数递增（控制发布频率）
        }
        rate.sleep();
    }
    return 0;
}