#pragma once

#include "Client.h"
#include <string>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <vector> // <-- 【新增】包含vector头文件

// 管理所有客户端连接的单例类
class ClientManager {
public:
    // 获取单例实例
    static ClientManager& getInstance();

    // 【修改】更新addClient方法声明，以接收KWS所需的参数
    std::shared_ptr<Client> addClient(
        const std::string& uuid,
        const std::string& kws_model_path,
        const std::vector<std::string>& kws_keyword_paths
    );

    // 移除客户端 (保持不变)
    void removeClient(const std::string& uuid);

    // 获取客户端 (保持不变)
    std::shared_ptr<Client> getClient(const std::string& uuid);

private:
    // 私有构造/析构/拷贝，确保单例模式
    ClientManager() = default;
    ~ClientManager() = default;
    ClientManager(const ClientManager&) = delete;
    ClientManager& operator=(const ClientManager&) = delete;

    // 存储所有Client对象的映射 (保持不变)
    std::unordered_map<std::string, std::shared_ptr<Client>> clients_;

    // 用于保护 clients_ 这个 map 本身的互斥锁 (保持不变)
    std::mutex manager_mutex_;
};