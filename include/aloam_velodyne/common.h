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

// 为了避免同一个头文件被包含（include）多次，C/C++中有两种宏实现方式：一种是#ifndef方式；另一种是#pragma once方式
// #ifndef方式:宏名撞车，编译慢，C/C++都支持
// #pragma once方式，编译快，但只针对文件（文件中有的编译有的不编译就不能用），需要编译器支持（现在都支持）
#pragma once
// 包含C++的数学库
#include <cmath>
// 包含PCL点云的基本类型，如PointXYZ,PointXYZI,PointXYZRGBA,PointNormal等点云类型
#include <pcl/point_types.h>
// typedef
// 定义一种类型的别名，而不只是简单的宏替换。
// 可以用作同时声明指针型的多个对象。
// 在scanRegistration中有应用
typedef pcl::PointXYZI PointType;
// Members: float x, y, z, uin32_t label
//  弧度转换为角度
inline double rad2deg(double radians)
{
  //
  return radians * 180.0 / M_PI;
}
// 角度转换为弧度
inline double deg2rad(double degrees)
{
  //
  return degrees * M_PI / 180.0;
}
