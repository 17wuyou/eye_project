// src/WebSocketServer.cpp

#include "WebSocketServer.h" // 包含头文件
#include <uwebsockets/App.h>
#include <iostream>
#include <objbase.h> // Windows UUID

// 每个连接的用户数据
struct PerSocketData {
    std::string client_uuid;
};

// UUID 生成函数（Windows 平台）
std::string generate_uuid() {
    GUID guid;
    if (CoCreateGuid(&guid) == S_OK) {
        wchar_t wstr[39];
        StringFromGUID2(guid, wstr, 39);

        char str[39];
        size_t converted_chars;
        wcstombs_s(&converted_chars, str, 39, wstr, 38);

        return std::string(str + 1, str + 36); // 去掉 {}
    }
    return "uuid-generation-failed";
}

// 构造函数和析构函数的实现（即使为空也要提供）
WebSocketServer::WebSocketServer() {}
WebSocketServer::~WebSocketServer() {}

// run 方法的实现
void WebSocketServer::run(int port) {
    uWS::App app;

    uWS::App::WebSocketBehavior<PerSocketData> behavior;

    behavior.compression = uWS::DEDICATED_COMPRESSOR_3KB;
    behavior.maxPayloadLength = 16 * 1024 * 1024;
    behavior.idleTimeout = 60;

    behavior.open = [this](auto* ws) {
        PerSocketData* data = ws->getUserData();
        data->client_uuid = generate_uuid();
        std::cout << "[WebSocket] Client connected. UUID: " << data->client_uuid << std::endl;
        if (on_open) on_open(data->client_uuid);
        };

    behavior.message = [this](auto* ws, std::string_view msg, uWS::OpCode opCode) {
        PerSocketData* data = ws->getUserData();
        if (on_message) on_message(data->client_uuid, msg);
        else ws->send(msg, opCode); // 默认 echo
        };

    behavior.close = [this](auto* ws, int code, std::string_view message) {
        PerSocketData* data = ws->getUserData();
        std::cout << "[WebSocket] Client disconnected. UUID: " << data->client_uuid << std::endl;
        if (on_close) on_close(data->client_uuid);
        };

    app.ws<PerSocketData>("/*", std::move(behavior))
        .listen(port, [port](auto* listen_socket) {
        if (listen_socket)
            std::cout << "[WebSocket] Listening on port " << port << std::endl;
        else
            std::cerr << "[WebSocket] Failed to listen on port " << port << std::endl;
            })
        .run();

    std::cout << "[WebSocket] Server has shut down." << std::endl;
}