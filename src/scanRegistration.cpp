// This is an advanced implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

// C++标准库，引用三角函数
#include <cmath>
// 迭代器
#include <vector>
// 字符串
#include <string>
// 自定义的两个文件，在include/aloam_velodyne目录下
// 该文件自己写了度数和角度的转换函数
#include "aloam_velodyne/common.h"
// 该文件定义了时间的记录
#include "aloam_velodyne/tic_toc.h"
// nav_msgs包含导航功能包、调用其中的odometry里程计
#include <nav_msgs/Odometry.h>
// 调用opencv库
// 图像处理库
#include <opencv2/imgproc.hpp>
// 进行PCL文件的数据类型转换
#include <pcl_conversions/pcl_conversions.h>
// PCL点云文件
#include <pcl/point_cloud.h>
// PCL点云的类型
#include <pcl/point_types.h>
// PCL滤波（去除无效点）：使用的是体素滤波器
#include <pcl/filters/voxel_grid.h>
// KD树
// 参考https://oi-wiki.org/ds/kdt/
// k-D Tree 具有二叉搜索树的形态，
// 二叉搜索树上的每个结点都对应 k 维空间内的一个点。
// 其每个子树中的点都在一个 k 维的超长方体内，
// 这个超长方体内的所有点也都在这个子树中。
// 具体解释https://zhuanlan.zhihu.com/p/617234731
// 下面的点云索引用的就是这个技术
#include <pcl/kdtree/kdtree_flann.h>
// ros系统
#include <ros/ros.h>
// 传感器IMU（加速度和角速度）信息
#include <sensor_msgs/Imu.h>
// 传感器3D点云信息
#include <sensor_msgs/PointCloud2.h>
// 该头文件定义了如下数据
// Quaternion（四元数）, Vector（向量）, Point（点）, Pose（位姿）, Transform（转移矩阵）
#include <tf/transform_datatypes.h>
// 位姿广播
#include <tf/transform_broadcaster.h>

// 三角函数的命名空间
using std::atan2;
using std::cos;
using std::sin;
// 激光雷达的扫描频率
const double scanPeriod = 0.1;
// 系统延迟，等几帧再进行运行
const int systemDelay = 0;
// 若等待时间不为0，需要开始等待计数，到一定数目运行
int systemInitCount = 0;
// 系统初始化
// 不进行系统初始化
// 若有系统延迟，就要系统初始化
bool systemInited = false;
// 激光雷达的线数
int N_SCANS = 0;
// 每个点云都有的属性：有效点云
// 点云的索引，梯度，
// 统计点云梯度
float cloudCurvature[400000];
// 点云的索引
int cloudSortInd[400000];
// 记录该点是否打过标签
// 在程序中会对特征点附近的几个点也打标签，目的：防止特征点太密
int cloudNeighborPicked[400000];
// 存储点云的标签
int cloudLabel[400000];
// 按照comp函数排序：曲率从小到大
// 这个函数只是给出sort排序函数的约束：排序的规则
bool comp(int i, int j) { return (cloudCurvature[i] < cloudCurvature[j]); }

// 定义几个发布者，发布处理分类过的点云信息
// 全部点云信息
ros::Publisher pubLaserCloud;
// 角点信息
ros::Publisher pubCornerPointsSharp;
// 次角点信息
ros::Publisher pubCornerPointsLessSharp;
// 平面点信息
ros::Publisher pubSurfPointsFlat;
// 次平面点信息
ros::Publisher pubSurfPointsLessFlat;
// 移除的点云
ros::Publisher pubRemovePoints;
// 定义一个发布队列，内部都是Publisher
std::vector<ros::Publisher> pubEachScan;
// 是否发布每线的激光雷达数据，不发布
bool PUB_EACH_LINE = false;
// 去除较近的无效点时，距离原点的距离
double MINIMUM_RANGE = 0.1;
// 定义类模板
template <typename PointT>
// 去除无效点云的函数（输入点云的引用，输出点云的引用，距离）
void removeClosedPointCloud(const pcl::PointCloud<PointT> &cloud_in,
                            pcl::PointCloud<PointT> &cloud_out, float thres)
{
    // 假如输入输出点云不使用同一个变量，则需要将输出点云的时间戳和容器大小与输入点云同步
    // 如果输入点云和输出点云不同
    // 一般输出点云是空
    if (&cloud_in != &cloud_out)
    {
        // 同步点云的头（时间戳和依赖的ID）
        cloud_out.header = cloud_in.header;
        // 设置成大小相同
        cloud_out.points.resize(cloud_in.points.size());
    }
    // 输出点云计数（过滤掉后数目减小）
    size_t j = 0;
    // 遍历所有点云
    for (size_t i = 0; i < cloud_in.points.size(); ++i)
    {
        // 计算在半径为thres的球内的点云
        if (cloud_in.points[i].x * cloud_in.points[i].x + cloud_in.points[i].y * cloud_in.points[i].y + cloud_in.points[i].z * cloud_in.points[i].z < thres * thres)
            // 不进行记录
            continue;
        // 进行记录球外点云
        cloud_out.points[j] = cloud_in.points[i];
        // 和i++同步
        j++;
    }
    //
    if (j != cloud_in.points.size())
    {
        // 重新计算输出点云的大小
        cloud_out.points.resize(j);
    }
    // 设为一行，j列的矩阵
    // 无结构点云
    cloud_out.height = 1;
    //
    cloud_out.width = static_cast<uint32_t>(j);
    // 判断点云中的点是否包含 Inf/NaN这种值
    // 不包含为true
    cloud_out.is_dense = true;
    // 其实无效数据没有去除干净
}
// 当激光雷达发布一帧（sweep）数据，计入回调函数
void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
{
    // 系统是否是第一次运行
    // systemInited默认值是0
    if (!systemInited)
    {
        // 记录等待时间（非真实时间）
        systemInitCount++;
        // 大于设定的次数，则系统不缓冲雷达数据
        // systemDelay默认值：0
        if (systemInitCount >= systemDelay)
        {
            // 不再进行判断
            systemInited = true;
        }
        else
            return;
    }
    // 定义一个类用于计时
    //
    TicToc t_whole;
    // 同上，记录准备时间，尤其是点云的分类
    TicToc t_prepare;
    // 开始记录所有有效点的曲率
    // 曲率开始的索引值（2维数组：16线，每线一个开始值）
    std::vector<int> scanStartInd(N_SCANS, 0);
    // 同上，只不过现在是结束的索引值
    std::vector<int> scanEndInd(N_SCANS, 0);
    // 命名一个PCL格式的输入点云
    pcl::PointCloud<pcl::PointXYZ> laserCloudIn;
    // pcl::PointCloud<pcl::PointXYZRGB> laserCloudIn;
    // 将ROS点云转化为PCL点云
    pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);
    // 首先对点云滤波，去除NaN(不是数）值得无效点云，以及在Lidar坐标系原点MINIMUM_RANGE距离以内的点
    // 创建一个数组，记录去除的点云索引值
    std::vector<int> indices;
    // 去除无效点云（NAN点）
    // 分别为输入点云，输出点云及对应保留的索引
    pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);
    // 去除在原点附近的点云
    removeClosedPointCloud(laserCloudIn, laserCloudIn, MINIMUM_RANGE);
    // 没有去除错点，杂点

    // 将激光点分类
    // 根据激光雷达模型，获得每个激光点的scanID以及扫描时间
    // velodyne 16雷达每次返回的数据称为一帧（sweep），一帧由16条线组成（每条线称为一个scan），每个scan有很多点。
    // 如果将velodyne 16雷达的扫描频率设置为10Hz，那么一秒就返回10帧数据。
    // 工作在10Hz的频率下，这个雷达的水平扫描角度的分辨率是0.2°，我们可以算出来理论上一帧有360/0.2×16=28800360/0.2×16=28800个点，
    // 但是实际上你可以试试，每次的点数不是完全一样的，有时多一点有时少一点，在程序中将存储点的数组定义为40000个元素，这是选了一个保守的上限。
    // 过程：
    // 首先求出一完整sweep数据中激光的起始角度startOri和终止角度endOri。
    // 求出每一个激光点的垂直俯仰角，从而利用这个俯仰角求出该激光点属于哪一个scan，即scanID，
    // 然后求出该点的旋转角ori，并利用该旋转角ori以及起始startOri和终点角度endOri和一个扫描周期的时间scanPeriod（总时间）计算该点的时间
    // 利用时间来代替角度，另一个好处是和其他传感器对齐时间
    // 最后设置intensity参数：
    // float relTime = (ori - startOri) / (endOri - startOri);
    // point.intensity = scanID + scanPeriod * relTime;
    // 最后将激光点放置于laserCloudScans[scanID]中
    // 参考https://blog.csdn.net/qq_41586768/article/details/108123854

    // 获取点云数据量的大小
    int cloudSize = laserCloudIn.points.size();
    // lidar scan开始点的旋转角,atan2范围[-pi,+pi],计算旋转角时取负号是因为velodyne是顺时针旋转,atan2逆时针为正角
    // atan2()函数是atan(y， x)函数的增强版，不仅可以求取arctran(y/x)还能够确定象限
    float startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);
    // 由于机械抖动激光雷达扫描不一定180度，也不一定回归0度在计算，所以加上2pi以保证Zaire周期范围内（是修正）
    float endOri = -atan2(laserCloudIn.points[cloudSize - 1].y,
                          laserCloudIn.points[cloudSize - 1].x) +
                   2 * M_PI;
    // 处理  保证  M_PI < endOri - startOri < 3 * M_PI
    // 原因：机关雷达一帧数据一定是扫描一圈以上，理论上是整数圈，扰动等，有误差（错）
    // 考虑旋转开始角和结束的相限关系来理解（对）
    // 虽然 负角度代表反方向转，但是不能被用来分配相对时间
    // 为了分配相对时间，旋转角度必须为正（即使是错的）
    // 参考https://zhuanlan.zhihu.com/p/395196484
    // 扫描场在右半部分（一、四象限）的情况
    if (endOri - startOri > 3 * M_PI)
    {
        // 大于三PI，减一圈
        endOri -= 2 * M_PI;
    }
    // 扫描场在左半部分（二、三象限）的情况
    // 小于PI，加一圈
    else if (endOri - startOri < M_PI)
    {
        endOri += 2 * M_PI;
    }
    // 输出最后角度
    // printf("end Ori %f\n", endOri);
    // lidar扫描线是否旋转过半，先默认没有
    // 这里会导致旋转角度出现错误，但是只要是正数就可以
    bool halfPassed = false;
    // 点云大小
    int count = cloudSize;
    // 定义点云的一个点
    PointType point;
    // 定义雷达线束，动态数组集合
    // 生成16线点云
    // 这里N_SCANS来自launch文件
    std::vector<pcl::PointCloud<PointType>> laserCloudScans(N_SCANS);
    // 开始遍历每个点
    for (int i = 0; i < cloudSize; i++)
    {
        // 提取一个点的坐标
        point.x = laserCloudIn.points[i].x;
        point.y = laserCloudIn.points[i].y;
        point.z = laserCloudIn.points[i].z;
        // 设置RGB颜色
        // laserCloudIn.points[i].r = 100;
        // laserCloudIn.points[i].g = 100;
        // laserCloudIn.points[i].b = 100;
        // 计算俯仰角度
        float angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
        // 初始化scanID，就是该点属于那个线
        int scanID = 0;
        // 如果是16线的激光雷达，角度是-15到15度
        // 作者使用的是velodyne
        // 16线的角度：-15，-13，-11，-9，-7，-5，-3，-1，1，3，5，7，9，11，13，15
        if (N_SCANS == 16)
        {
            // 除2是因为每两个scan之间的间隔为2度，scanID设定为相差1
            // + 0.5 是用于四舍五入   因为 int 只会保留整数部分
            scanID = int((angle + 15) / 2 + 0.5);
            // 更新后的ID:0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5,15.5
            // 差值为1
            // 超过16线或者小于16线，则不标定
            if (scanID > (N_SCANS - 1) || scanID < 0)
            {
                count--;
                continue;
            }
        }
        // 这是32线的情况
        else if (N_SCANS == 32)
        {
            //
            scanID = int((angle + 92.0 / 3.0) * 3.0 / 4.0);
            //
            if (scanID > (N_SCANS - 1) || scanID < 0)
            {
                count--;
                continue;
            }
        }
        // 这是64线的情况
        else if (N_SCANS == 64)
        {
            //
            if (angle >= -8.83)
                scanID = int((2 - angle) * 3.0 + 0.5);
            //
            else
                scanID = N_SCANS / 2 + int((-8.83 - angle) * 2.0 + 0.5);

            // use [0 50]  > 50 remove outlies
            //
            if (angle > 2 || angle < -24.33 || scanID > 50 || scanID < 0)
            {
                count--;
                continue;
            }
        }
        // 如果不是16线的激光雷达，或32、64，就不能进行运算
        else
        {
            printf("wrong scan number\n");
            ROS_BREAK();
        }
        // 输出角度值和scanID
        // printf("angle %f scanID %d \n", angle, scanID);
        // 可以看到atan2( )默认返回逆时针角度，
        // 由于LiDAR通常是顺时针扫描，
        // 所以往往使用-atan2( )函数。
        // 计算平面角度，即在某条线上
        float ori = -atan2(point.y, point.x);
        // 是否转过半圈，halfPassed默认值：false
        // 扫描控制在pi到3pi之间，有可能是2pi多，及转了1圈多，到达当前点的位置
        // 当前点今计算角度，就有两个
        // 根据扫描线是否旋转过半选择与起始位置还是终止位置进行差值计算，从而进行补偿
        // 未过半圈，用起始位置算，已过半圈的用终止位置算
        // 先假设未过半圈
        if (!halfPassed)
        {
            //
            if (ori < startOri - M_PI / 2)
            {
                //
                ori += 2 * M_PI;
            }
            //
            else if (ori > startOri + M_PI * 3 / 2)
            {
                //
                ori -= 2 * M_PI;
            }
            //
            if (ori - startOri > M_PI)
            {
                //
                halfPassed = true;
            }
        }
        //
        else
        {
            //
            ori += 2 * M_PI;
            if (ori < endOri - M_PI * 3 / 2)
            {
                //
                ori += 2 * M_PI;
            }
            //
            else if (ori > endOri + M_PI / 2)
            {
                //
                ori -= 2 * M_PI;
            }
        }
        //
        float relTime = (ori - startOri) / (endOri - startOri);
        // 计算相对时间（相对于角度多了scanID和时间等多个参数组成的标签）
        // scanPeriod是扫描频率，0.1
        point.intensity = scanID + scanPeriod * relTime;
        // 将点云押入laserCloudScans
        laserCloudScans[scanID].push_back(point);
    }
    // 有点云在计算中超出范围：超出线数的俯仰角度，或者线转角度有问题
    // count记录的是绝对有效的点云个数
    cloudSize = count;
    printf("points size %d \n", cloudSize);
    // 特征点提取
    // 创建一个点云队列
    pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
    // 去除前5个点，后5个点
    for (int i = 0; i < N_SCANS; i++)
    {
        // 将原始点云第5个作为开始索引
        scanStartInd[i] = laserCloud->size() + 5;
        // 将该线的数据赋值给该线
        *laserCloud += laserCloudScans[i];
        // 将倒数第五个作为结束索引值
        scanEndInd[i] = laserCloud->size() - 6;
    }
    // 输出前面点云处理的时间
    // 将一帧无序点云转换成有序点云消耗的时间
    printf("prepare time %f \n", t_prepare.toc());
    // 基于当前线
    // 从第6个数据和倒数第6个数据
    for (int i = 5; i < cloudSize - 5; i++)
    {
        // 计算该点的梯度信息
        // 实际上就是同一条扫描线上的取目标点左右两侧各5个点，分别与目标点的坐标作差，得到的结果就是目标点的曲率。
        // 当目标点处在棱或角的位置时，自然与周围点的差值较大，得到的曲率较大；反之当目标点在平面上时，周围点与目标点的坐标相近，得到的曲率自然较小。
        // 缺点，没有跨线计算曲率
        float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x + laserCloud->points[i - 3].x + laserCloud->points[i - 2].x + laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x + laserCloud->points[i + 1].x + laserCloud->points[i + 2].x + laserCloud->points[i + 3].x + laserCloud->points[i + 4].x + laserCloud->points[i + 5].x;
        float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y + laserCloud->points[i - 3].y + laserCloud->points[i - 2].y + laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y + laserCloud->points[i + 1].y + laserCloud->points[i + 2].y + laserCloud->points[i + 3].y + laserCloud->points[i + 4].y + laserCloud->points[i + 5].y;
        float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z + laserCloud->points[i - 3].z + laserCloud->points[i - 2].z + laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z + laserCloud->points[i + 1].z + laserCloud->points[i + 2].z + laserCloud->points[i + 3].z + laserCloud->points[i + 4].z + laserCloud->points[i + 5].z;
        // 一方面保证曲率为正，另一方面遵从论文中的梯度公式
        cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;
        // 记录点云索引信息
        cloudSortInd[i] = i;
        // 点云周围点的选择
        cloudNeighborPicked[i] = 0;
        // 点云标签（后面对角点，平面点分类）
        cloudLabel[i] = 0;
    }
    // 声明一个处理点云分类的时间记录类
    // 记录点云分成4类的时间
    TicToc t_pts;
    // 声明一个角点空集和
    pcl::PointCloud<PointType> cornerPointsSharp;
    // 次角点
    pcl::PointCloud<PointType> cornerPointsLessSharp;
    // 平面点
    pcl::PointCloud<PointType> surfPointsFlat;
    // 次平面点
    pcl::PointCloud<PointType> surfPointsLessFlat;
    // 对每条线扫scan进行操作（曲率排序，选取对应特征点）
    // 记录时间
    float t_q_sort = 0;
    // 遍历每条线
    for (int i = 0; i < N_SCANS; i++)
    {
        // 如果一条线上开始到结束小于6个，无法做梯度，做差，计算曲率
        // 如果最后一个可算曲率的点与第一个的差小于6，说明无法分成6个扇区，跳过
        if (scanEndInd[i] - scanStartInd[i] < 6)
            continue;
        // 创造该类型指针，指向次平面点云集合（误解）
        //
        pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);
        // 为了使特征点均匀分布，将一个scan分成6个扇区
        for (int j = 0; j < 6; j++)
        {
            // 计算每个扇区的起点
            int sp = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * j / 6;
            // 计算每个扇区的终点
            int ep = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * (j + 1) / 6 - 1;
            // 创建类记录处理时间
            TicToc t_tmp;
            // 按照曲率进行升序排序
            // sort排序，comp函数中的方法排序
            std::sort(cloudSortInd + sp, cloudSortInd + ep + 1, comp);
            // 当前扇区的处理时间
            // t_q_sort累计每个扇区曲率排序时间总和
            t_q_sort += t_tmp.toc();
            // 选取极大边线点（2个）和次极大边线点（20个）
            // 最大选取数目
            int largestPickedNum = 0;
            // 遍历排序后的点云，从曲率最大开始
            // 从最大曲率往最小曲率遍历，寻找边线点，并要求大于0.1
            for (int k = ep; k >= sp; k--)
            {
                // 获取当前索引值
                int ind = cloudSortInd[k];
                // 默认的cloudNeighborPicked是0
                // 意思是未打标签
                // 若最大曲率小于0.1，可认为是平面点
                if (cloudNeighborPicked[ind] == 0 &&
                    cloudCurvature[ind] > 0.1)
                {
                    // 已获取最大曲率点一个
                    largestPickedNum++;
                    // 限制获取的最大曲率点超过两个
                    if (largestPickedNum <= 2)
                    {
                        // 曲率标签：-2，-1，0，1，2
                        // 设置曲率最大点的标签为2
                        cloudLabel[ind] = 2;
                        // 既放入极大边线点容器，也放入次极大边线点容器
                        cornerPointsSharp.push_back(laserCloud->points[ind]);
                        // 不懂为什么放入次角点
                        cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                    }
                    // 选取20个次角点
                    else if (largestPickedNum <= 20)
                    {
                        // 标签设置为1
                        cloudLabel[ind] = 1;
                        // 押入次角点点云集合
                        cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                    }
                    else
                    {
                        break;
                    }
                    // 标记该点已经打过标签
                    cloudNeighborPicked[ind] = 1;
                    // ID为ind的特征点的相邻scan点距离的平方 <= 0.05的点标记为选择过，避免特征点密集分布
                    // 为避免特征点密布，立刻计算附近的曲率，如果相差不大，则也标为标记过
                    // 为什么不用之前计算的曲率？查不到吗？
                    // 因为是按曲率大小排序的，现在计算的是前后点的距离差大不大
                    // 大的话，就不在同一地方，特征点就没有密集分布
                    // 往后数5个
                    for (int l = 1; l <= 5; l++)
                    {
                        // 计算两点间的距离（并没有计算这5个点与该点的距离）
                        // 计算的是该点和后一个点，后一个点和后两个点，......
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                        // 距离大于0.05，则不标记
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }
                        // 将该邻近点进行标记
                        cloudNeighborPicked[ind + l] = 1;
                    }
                    // 计算前5个点
                    for (int l = -1; l >= -5; l--)
                    {
                        //
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                        //
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }
                        //
                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
                // 点云标记已完成
            }
            // 最大角点编译完成
            // 最小值点计数
            int smallestPickedNum = 0;
            // 遍历该扇区
            // 问题：为什么不一起计算
            // 这次是曲率从小到大遍历（速度快，如果是在上一个，遍历时间长）
            // 遍历时忽略的条件不同
            for (int k = sp; k <= ep; k++)
            {
                // 获取当前索引值
                int ind = cloudSortInd[k];
                // 没有标记和曲率小于0.1
                if (cloudNeighborPicked[ind] == 0 &&
                    cloudCurvature[ind] < 0.1)
                {
                    // 标签设置为-1
                    cloudLabel[ind] = -1;
                    // 押入平面点点云
                    surfPointsFlat.push_back(laserCloud->points[ind]);
                    // 平面点计数
                    smallestPickedNum++;
                    // 平面点4个
                    if (smallestPickedNum >= 4)
                    {
                        break;
                    }
                    // 该点已选择
                    cloudNeighborPicked[ind] = 1;
                    // 同上操作：防止特征点过密
                    for (int l = 1; l <= 5; l++)
                    {
                        //
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                        //
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }
                        //
                        cloudNeighborPicked[ind + l] = 1;
                    }
                    //
                    for (int l = -1; l >= -5; l--)
                    {
                        //
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                        //
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }
                        //
                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
                // 完成最平面点的选择
            }
            // 选取次极小平面点，除了极大平面点、次极大平面点，剩下的都是次极小平面点
            // 平面点：-1，次平面点：0，次角点：1，角点：2
            // 上面每线选了4个平面点，余下的都是次平面点
            for (int k = sp; k <= ep; k++)
            {
                // 剩余的点全部为次平面点
                if (cloudLabel[k] <= 0)
                {
                    // 存储到次平面点扫描
                    surfPointsLessFlatScan->push_back(laserCloud->points[k]);
                }
            }
        }
        // 对每一条scan线上的次极小平面点进行一次降采样
        // 次平面点点云（太多了）
        // 降低特征点密度
        pcl::PointCloud<PointType> surfPointsLessFlatScanDS;
        // 降采样的滤波器
        // 算法名：下采样（约减少到原来的1/10）
        // voxelGrid类通过在点云数据中创建三维体素栅格，然后用每个体素的重心来近似表达体素中的其它点。
        // 优点：准确，缺点：速度太慢
        // 改进：用体素中心来逼近的方法
        pcl::VoxelGrid<PointType> downSizeFilter;
        // 载入原始点云
        downSizeFilter.setInputCloud(surfPointsLessFlatScan);
        // 滤波格子（三维正方体）的大小，多大空间里的点云转化为1个点
        // LeafSize设置的越大，体素格子也就越大，滤波效果也就越明显
        downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
        // 进行滤波的结果
        downSizeFilter.filter(surfPointsLessFlatScanDS);
        // 放入次平面点云
        surfPointsLessFlat += surfPointsLessFlatScanDS;
    }
    // 排序所用的时间
    printf("sort q time %f \n", t_q_sort);
    // 分类成四类点云的时间
    printf("seperate points time %f \n", t_pts.toc());
    // 声明发布的点云（新版：pcl::PointClound2）
    // 这是一个ROS点云
    sensor_msgs::PointCloud2 laserCloudOutMsg;
    // 转化为ROS点云
    // 将有序化点云转化为ROS点云
    pcl::toROSMsg(*laserCloud, laserCloudOutMsg);
    // 时间戳保持不变
    laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
    // 发布名字（从属的ID）
    // 没有发布的名称
    laserCloudOutMsg.header.frame_id = "/camera_init";
    // 发布
    // pubLaserCloud定义在main()中
    // 问题：为什么回调函数后面的变量，前面能调用
    pubLaserCloud.publish(laserCloudOutMsg);
    // 角点的发布过程
    sensor_msgs::PointCloud2 cornerPointsSharpMsg;
    //
    pcl::toROSMsg(cornerPointsSharp, cornerPointsSharpMsg);
    //
    cornerPointsSharpMsg.header.stamp = laserCloudMsg->header.stamp;
    //
    cornerPointsSharpMsg.header.frame_id = "/camera_init";
    //
    pubCornerPointsSharp.publish(cornerPointsSharpMsg);
    // 次角点的发布
    sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
    //
    pcl::toROSMsg(cornerPointsLessSharp, cornerPointsLessSharpMsg);
    //
    cornerPointsLessSharpMsg.header.stamp = laserCloudMsg->header.stamp;
    //
    cornerPointsLessSharpMsg.header.frame_id = "/camera_init";
    //
    pubCornerPointsLessSharp.publish(cornerPointsLessSharpMsg);
    // 平面点的发布
    sensor_msgs::PointCloud2 surfPointsFlat2;
    //
    pcl::toROSMsg(surfPointsFlat, surfPointsFlat2);
    //
    surfPointsFlat2.header.stamp = laserCloudMsg->header.stamp;
    //
    surfPointsFlat2.header.frame_id = "/camera_init";
    //
    pubSurfPointsFlat.publish(surfPointsFlat2);
    // 次平面点的发布
    sensor_msgs::PointCloud2 surfPointsLessFlat2;
    //
    pcl::toROSMsg(surfPointsLessFlat, surfPointsLessFlat2);
    //
    surfPointsLessFlat2.header.stamp = laserCloudMsg->header.stamp;
    //
    surfPointsLessFlat2.header.frame_id = "/camera_init";
    //
    pubSurfPointsLessFlat.publish(surfPointsLessFlat2);
    // 如果发布每条线
    // pub each scam
    if (PUB_EACH_LINE)
    {
        //
        for (int i = 0; i < N_SCANS; i++)
        {
            //
            sensor_msgs::PointCloud2 scanMsg;
            //
            pcl::toROSMsg(laserCloudScans[i], scanMsg);
            //
            scanMsg.header.stamp = laserCloudMsg->header.stamp;
            //
            scanMsg.header.frame_id = "/camera_init";
            //
            pubEachScan[i].publish(scanMsg);
        }
    }
    // 扫描处理时间为几ms
    printf("scan registration time %f ms *************\n", t_whole.toc());
    // 超过100ms发出警告
    // 大概因为激光雷达的帧率，0.1s可能第二帧激光雷达数据来了（错）
    // ros::spin();和ros::Rate loop_rate(10);有关
    // 每0.1s进一次处理消息队列
    if (t_whole.toc() > 100)
        ROS_WARN("scan registration process over 100ms");
}

int main(int argc, char **argv)
{
    // ROS初始化节点：scanRegistration
    ros::init(argc, argv, "scanRegistration");
    // 创建ROS句柄，将节点实例化
    ros::NodeHandle nh;
    // 从launch文件中读取N_SCANS，如果没有，则将16赋值给scan_line
    nh.param<int>("scan_line", N_SCANS, 16);
    // 道理同上，minimum_range == 0.1
    nh.param<double>("minimum_range", MINIMUM_RANGE, 0.1);
    // 输出N_SCANS
    printf("scan line number %d \n", N_SCANS);
    // 一般3D激光雷达分为16线，32，64
    if (N_SCANS != 16 && N_SCANS != 32 && N_SCANS != 64)
    {
        printf("only support velodyne with 16, 32 or 64 scan line!");
        return 0;
    }
    // 定义一个订阅者 subLaserCloud 接收从激光雷达或rosbag发布的话题:/velodyne_points
    // 调用laserCloudHandler这个回调函数对点云进行分类
    // laserCloudHandler并没有直接进入，在ros::spin()触发后才开始回调
    // 所以下面是发布函数的定义，但是回调函数中已经开始使用了
    // 开始运行函数时并没有调用laserCloudHandler，下面对发布者进行定义，当运行到ros::spin()时，进入消息队列，处理laserCloudHandler，才开始发布
    ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 100, laserCloudHandler);
    // 发布对象pubLaserCloud，发布点云数据，话题名/velodyne_cloud_2
    // 其实laserCloudHandler已经发布
    pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100);
    // 同上，这些点云是角点
    pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 100);
    // 同上，这是曲率次大一点的点云
    pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100);
    // 这是平面点云
    pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100);
    // 这是曲率次小一点的点云
    pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100);
    // 这是移除的点云
    pubRemovePoints = nh.advertise<sensor_msgs::PointCloud2>("/laser_remove_points", 100);
    // 是否发布每行数据
    // 存疑：这里是false
    if (PUB_EACH_LINE)
    {
        // 逐行发布
        for (int i = 0; i < N_SCANS; i++)
        {
            // 定义发布者，tmp 发布16行数据，话题名为/laser_scanid_1，/laser_scanid_2，。。。
            ros::Publisher tmp = nh.advertise<sensor_msgs::PointCloud2>("/laser_scanid_" + std::to_string(i), 100);
            // 添加到发布队列，一起同时发布
            // pubEachScan是全局变量
            pubEachScan.push_back(tmp);
        }
    }
    // 暂停，时间和ros::Rate loop_rate(10);的频率有关
    ros::spin();

    return 0;
}
