// 作者:   覃桐               qintonguav@gmail.com
//         曹绍祖            saozu.cao@connect.ust.hk

// 包含ceres优化库头文件
#include <ceres/ceres.h>
// 包含ceres旋转函数头文件
#include <ceres/rotation.h>
// 包含Eigen矩阵运算库头文件
#include <eigen3/Eigen/Dense>
// 包含PCL点云库头文件
#include <pcl/point_cloud.h>
// 包含PCL点类型头文件
#include <pcl/point_types.h>
// 包含PCL KD树头文件
#include <pcl/kdtree/kdtree_flann.h>
// 包含PCL与ROS消息转换头文件
#include <pcl_conversions/pcl_conversions.h>

// 激光雷达边缘特征残差项结构体（用于点到直线的约束）
struct LidarEdgeFactor
{
    // 构造函数：接收当前点、上一帧的两个边缘点、时间戳s
    LidarEdgeFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_a_,
                    Eigen::Vector3d last_point_b_, double s_)
        : curr_point(curr_point_), last_point_a(last_point_a_), last_point_b(last_point_b_), s(s_) {}

    // 残差计算函数（ceres自动求导核心）
    template <typename T>
    bool operator()(const T *q, const T *t, T *residual) const
    {
        // 将当前点转换为模板类型向量
        Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
        // 将上一帧的两个边缘点转换为模板类型向量
        Eigen::Matrix<T, 3, 1> lpa{T(last_point_a.x()), T(last_point_a.y()), T(last_point_a.z())};
        Eigen::Matrix<T, 3, 1> lpb{T(last_point_b.x()), T(last_point_b.y()), T(last_point_b.z())};

        // 构造上一帧到当前帧的四元数（初始为单位四元数）
		// Eigen::Quaternion<T> q_last_curr{q[3], T(s) * q[0], T(s) * q[1], T(s) * q[2]};
        Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
        Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
        // 基于时间戳s进行球面线性插值（处理运动畸变）
        q_last_curr = q_identity.slerp(T(s), q_last_curr);
        // 构造上一帧到当前帧的平移向量（时间戳s加权）
        Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};

        // 将当前点转换到上一帧坐标系
        Eigen::Matrix<T, 3, 1> lp;
        lp = q_last_curr * cp + t_last_curr;

        // 计算点到直线的距离残差（叉乘模长除以直线长度）
        Eigen::Matrix<T, 3, 1> nu = (lp - lpa).cross(lp - lpb);  // 叉乘得到平行四边形面积向量
        Eigen::Matrix<T, 3, 1> de = lpa - lpb;                  // 直线方向向量

        // 残差为面积向量在直线法平面上的投影（归一化处理）
        residual[0] = nu.x() / de.norm();
        residual[1] = nu.y() / de.norm();
        residual[2] = nu.z() / de.norm();

        return true;
    }

    // 工厂函数：创建ceres代价函数（自动求导）
    static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_a_,
                                       const Eigen::Vector3d last_point_b_, const double s_)
    {
        return (new ceres::AutoDiffCostFunction<
                LidarEdgeFactor, 3, 4, 3>(  // 残差维度3，参数块维度4（四元数）和3（平移）
            new LidarEdgeFactor(curr_point_, last_point_a_, last_point_b_, s_)));
    }

    // 成员变量：当前点、上一帧的两个边缘点、时间戳s
    Eigen::Vector3d curr_point, last_point_a, last_point_b;
    double s;
};

// 激光雷达平面特征残差项结构体（用于点到平面的约束）
struct LidarPlaneFactor
{
    // 构造函数：接收当前点、上一帧的三个平面点、时间戳s，并预计算平面法向量
    LidarPlaneFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d last_point_j_,
                     Eigen::Vector3d last_point_l_, Eigen::Vector3d last_point_m_, double s_)
        : curr_point(curr_point_), last_point_j(last_point_j_), last_point_l(last_point_l_),
          last_point_m(last_point_m_), s(s_)
    {
        // 计算平面法向量（两个向量叉乘）并归一化
        ljm_norm = (last_point_j - last_point_l).cross(last_point_j - last_point_m);
        ljm_norm.normalize();
    }

    // 残差计算函数（ceres自动求导核心）
    template <typename T>
    bool operator()(const T *q, const T *t, T *residual) const
    {
        // 将当前点转换为模板类型向量
        Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
        // 将上一帧的基准平面点转换为模板类型向量
        Eigen::Matrix<T, 3, 1> lpj{T(last_point_j.x()), T(last_point_j.y()), T(last_point_j.z())};
		// Eigen::Matrix<T, 3, 1> lpl{T(last_point_l.x()), T(last_point_l.y()), T(last_point_l.z())};
		// Eigen::Matrix<T, 3, 1> lpm{T(last_point_m.x()), T(last_point_m.y()), T(last_point_m.z())};
        // 将预计算的平面法向量转换为模板类型向量
        Eigen::Matrix<T, 3, 1> ljm{T(ljm_norm.x()), T(ljm_norm.y()), T(ljm_norm.z())};

        // 构造上一帧到当前帧的四元数（初始为单位四元数）
		// Eigen::Quaternion<T> q_last_curr{q[3], T(s) * q[0], T(s) * q[1], T(s) * q[2]};
        Eigen::Quaternion<T> q_last_curr{q[3], q[0], q[1], q[2]};
        Eigen::Quaternion<T> q_identity{T(1), T(0), T(0), T(0)};
        // 基于时间戳s进行球面线性插值（处理运动畸变）
        q_last_curr = q_identity.slerp(T(s), q_last_curr);
        // 构造上一帧到当前帧的平移向量（时间戳s加权）
        Eigen::Matrix<T, 3, 1> t_last_curr{T(s) * t[0], T(s) * t[1], T(s) * t[2]};

        // 将当前点转换到上一帧坐标系
        Eigen::Matrix<T, 3, 1> lp;
        lp = q_last_curr * cp + t_last_curr;

        // 残差为点到平面的有符号距离（点与基准点的差与法向量的点积）
        residual[0] = (lp - lpj).dot(ljm);

        return true;
    }

    // 工厂函数：创建ceres代价函数（自动求导）
    static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d last_point_j_,
                                       const Eigen::Vector3d last_point_l_, const Eigen::Vector3d last_point_m_,
                                       const double s_)
    {
        return (new ceres::AutoDiffCostFunction<
                LidarPlaneFactor, 1, 4, 3>(  // 残差维度1，参数块维度4（四元数）和3（平移）
            new LidarPlaneFactor(curr_point_, last_point_j_, last_point_l_, last_point_m_, s_)));
    }

    // 成员变量：当前点、上一帧的三个平面点、平面法向量、时间戳s
    Eigen::Vector3d curr_point, last_point_j, last_point_l, last_point_m;
    Eigen::Vector3d ljm_norm;
    double s;
};

// 激光雷达平面法向量残差项结构体（已知平面法向量时的约束）
struct LidarPlaneNormFactor
{
    // 构造函数：接收当前点、平面单位法向量、负的原点到平面的距离（OA·n的负值）
    LidarPlaneNormFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d plane_unit_norm_,
                         double negative_OA_dot_norm_) : curr_point(curr_point_), plane_unit_norm(plane_unit_norm_),
                                                         negative_OA_dot_norm(negative_OA_dot_norm_) {}

    // 残差计算函数（ceres自动求导核心）
    template <typename T>
    bool operator()(const T *q, const T *t, T *residual) const
    {
        // 构造当前帧到世界坐标系的四元数和平移向量
        Eigen::Quaternion<T> q_w_curr{q[3], q[0], q[1], q[2]};
        Eigen::Matrix<T, 3, 1> t_w_curr{t[0], t[1], t[2]};
        // 将当前点转换为模板类型向量
        Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
        // 将当前点转换到世界坐标系
        Eigen::Matrix<T, 3, 1> point_w;
        point_w = q_w_curr * cp + t_w_curr;

        // 将平面单位法向量转换为模板类型向量
        Eigen::Matrix<T, 3, 1> norm(T(plane_unit_norm.x()), T(plane_unit_norm.y()), T(plane_unit_norm.z()));
        // 残差为世界坐标系下点到平面的距离（点·法向量 + 原点距离）
        residual[0] = norm.dot(point_w) + T(negative_OA_dot_norm);
        return true;
    }

    // 工厂函数：创建ceres代价函数（自动求导）
    static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d plane_unit_norm_,
                                       const double negative_OA_dot_norm_)
    {
        return (new ceres::AutoDiffCostFunction<
                LidarPlaneNormFactor, 1, 4, 3>(  // 残差维度1，参数块维度4（四元数）和3（平移）
            new LidarPlaneNormFactor(curr_point_, plane_unit_norm_, negative_OA_dot_norm_)));
    }

    // 成员变量：当前点、平面单位法向量、负的原点到平面距离
    Eigen::Vector3d curr_point;
    Eigen::Vector3d plane_unit_norm;
    double negative_OA_dot_norm;
};

// 激光雷达点距离残差项结构体（点到点的直接距离约束）
struct LidarDistanceFactor
{
    // 构造函数：接收当前点和最近点
    LidarDistanceFactor(Eigen::Vector3d curr_point_, Eigen::Vector3d closed_point_)
        : curr_point(curr_point_), closed_point(closed_point_) {}

    // 残差计算函数（ceres自动求导核心）
    template <typename T>
    bool operator()(const T *q, const T *t, T *residual) const
    {
        // 构造当前帧到世界坐标系的四元数和平移向量
        Eigen::Quaternion<T> q_w_curr{q[3], q[0], q[1], q[2]};
        Eigen::Matrix<T, 3, 1> t_w_curr{t[0], t[1], t[2]};
        // 将当前点转换为模板类型向量
        Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
        // 将当前点转换到世界坐标系
        Eigen::Matrix<T, 3, 1> point_w;
        point_w = q_w_curr * cp + t_w_curr;

        // 残差为世界坐标系下当前点与最近点的坐标差
        residual[0] = point_w.x() - T(closed_point.x());
        residual[1] = point_w.y() - T(closed_point.y());
        residual[2] = point_w.z() - T(closed_point.z());
        return true;
    }

    // 工厂函数：创建ceres代价函数（自动求导）
    static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_, const Eigen::Vector3d closed_point_)
    {
        return (new ceres::AutoDiffCostFunction<
                LidarDistanceFactor, 3, 4, 3>(  // 残差维度3，参数块维度4（四元数）和3（平移）
            new LidarDistanceFactor(curr_point_, closed_point_)));
    }

    // 成员变量：当前点、最近点
    Eigen::Vector3d curr_point;
    Eigen::Vector3d closed_point;
};