// include/WebSocketServer.h (最终修正版)

#pragma once

#include <string>
#include <functional>
#include <string_view>

// 【修复】不再使用错误的前向声明，而是直接包含 uWebSockets 的头文件。
// 这样可以确保编译器获得 uWS::App 最准确、唯一的定义。
#include <uwebsockets/App.h> 

class WebSocketServer {
public:
    // 类型定义
    using OpenCallback = std::function<void(const std::string&)>;
    using MessageCallback = std::function<void(const std::string&, std::string_view)>;
    using CloseCallback = std::function<void(const std::string&)>;

    // 构造函数和析构函数
    WebSocketServer();
    ~WebSocketServer();

    // 成员函数声明
    void run(int port);

    // 回调函数成员变量
    OpenCallback on_open;
    MessageCallback on_message;
    CloseCallback on_close;
};