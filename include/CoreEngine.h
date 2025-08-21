#pragma once

#include "WebSocketServer.h"
#include "pybind_bridge.h"  // <-- 【新增】引入我们的桥接头文件
#include <string>
#include <vector>
#include <queue>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <memory>              // <-- 【新增】为了使用 std::unique_ptr

// 定义一个包含客户端UUID和消息的数据包结构
struct DataPacket {
    std::string client_uuid;
    std::string message;
};

// 一个简单的线程安全队列，用于在网络线程和处理线程之间传递数据
template<typename T>
class ThreadSafeQueue {
public:
    void push(T value) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(value));
        cond_var_.notify_one();
    }

    bool try_pop(T& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return false;
        }
        value = std::move(queue_.front());
        queue_.pop();
        return true;
    }

private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cond_var_;
};


class CoreEngine {
public:
    CoreEngine();
    ~CoreEngine();

    // 启动引擎的主函数
    void run();

private:
    // --- 核心组件 ---
    WebSocketServer server_; 
    ThreadSafeQueue<DataPacket> incoming_data_queue_; 
    
    // --- 线程 ---
    std::thread processing_thread_; 
    std::atomic<bool> is_running_; // 【修改】使用 std::atomic<bool> 更适合多线程环境

    // --- 业务逻辑 ---
    // WebSocket 事件的回调处理函数
    void handle_connection_open(const std::string& client_uuid); 
    void handle_connection_message(const std::string& client_uuid, std::string_view message); 
    void handle_connection_close(const std::string& client_uuid); 

    // 数据处理线程的主循环
    void data_processing_loop(); 

    // --- 【新增】Python 桥接与解释器管理 ---
    // py::scoped_interpreter 必须是成员变量，它的生命周期将管理Python解释器的初始化和销毁
    py::scoped_interpreter guard{}; 

    // 使用智能指针管理 AIServiceManager 的生命周期
    std::unique_ptr<AIServiceManager> ai_services_;
};