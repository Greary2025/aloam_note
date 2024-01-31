// Author:   Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk
// 是一个比较常用的C/C++预处理指令，只要在头文件的最开始加入这条预处理指令，就能够保证头文件只被编译一次，防止头文件被重复引用。
#pragma once
// c++中的系统时间函数
#include <ctime>
// C++ 通用标准库头文件，也称为 C++ cstdlib，定义了用于数据类型转换、伪随机数生成、内存分配、搜索、排序、数学和处理宽或多字节字符的核心函数集。
#include <cstdlib>
// chrono是一个time library, 源于boost，现在已经是C++标准。
#include <chrono>
// 自定义类
class TicToc
{
public:
  // 构造函数
  TicToc()
  {
    // 开始就调用tic(),用于记录开始的时间
    tic();
  }
  // 获取系统时间
  void tic()
  {
    start = std::chrono::system_clock::now();
  }
  // 计算从开始到现在的时间
  double toc()
  {
    // 获取当前时间
    end = std::chrono::system_clock::now();
    // 计算时间间隔（std::chrono::duration有的返回单位是秒，这里好像是ms）
    std::chrono::duration<double> elapsed_seconds = end - start;
    // 返回秒
    return elapsed_seconds.count() * 1000;
  }

private:
  // 定义计时开始和结束的两个变量
  // 类模板 std::chrono::time_point 表示时间中的一个点。
  // 它被实现成如同存储一个 Duration 类型的自 Clock 的纪元起始开始的时间间隔的值。
  // <std::chrono::system_clock>是一种数据类型，像int,这里是类
  std::chrono::time_point<std::chrono::system_clock> start, end;
};
