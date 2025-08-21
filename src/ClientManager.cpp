#include "ClientManager.h"
#include <iostream>
#include <memory>

// getInstance() 方法保持不变
ClientManager& ClientManager::getInstance() {
    static ClientManager instance;
    return instance;
}

// 【修改】更新addClient方法的实现
std::shared_ptr<Client> ClientManager::addClient(
    const std::string& uuid,
    const std::string& kws_model_path,
    const std::vector<std::string>& kws_keyword_paths) 
{
    const std::lock_guard<std::mutex> lock(manager_mutex_);

    if (clients_.count(uuid)) {
        std::cerr << "[Warning] ClientManager: Attempted to add an existing client with UUID " << uuid << std::endl;
        return clients_.at(uuid);
    }

    // 【核心改动】使用新的Client构造函数创建实例，并传递所有参数
    auto new_client = std::make_shared<Client>(uuid, kws_model_path, kws_keyword_paths);
    clients_[uuid] = new_client;

    std::cout << "[Log] ClientManager: Added new client " << uuid << ". Total clients: " << clients_.size() << std::endl;

    return new_client;
}

// removeClient() 和 getClient() 方法保持不变
void ClientManager::removeClient(const std::string& uuid) {
    const std::lock_guard<std::mutex> lock(manager_mutex_);

    if (clients_.count(uuid)) {
        clients_.erase(uuid);
        std::cout << "[Log] ClientManager: Removed client " << uuid << ". Total clients: " << clients_.size() << std::endl;
    } else {
        std::cerr << "[Warning] ClientManager: Attempted to remove a non-existent client with UUID " << uuid << std::endl;
    }
}

std::shared_ptr<Client> ClientManager::getClient(const std::string& uuid) {
    const std::lock_guard<std::mutex> lock(manager_mutex_);

    auto it = clients_.find(uuid);
    if (it != clients_.end()) {
        return it->second;
    }
    return nullptr;
}