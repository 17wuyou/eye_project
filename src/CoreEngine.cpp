#include "CoreEngine.h"
#include "ClientManager.h"
#include "Client.h"
#include "KeyframeDetector.h"
#include <iostream>
#include <chrono>
#include <atomic>

// 引入JSON处理库
#include "nlohmann/json.hpp" 
using json = nlohmann::json;

// TODO: 您需要将Python的加密和Base64工具移植到C++。
//       在完成移植前，我们暂时使用占位函数。
namespace YourUtils {
    std::string decrypt_from_string(const std::string& encrypted_str) {
        return encrypted_str; 
    }
    std::string base64_decode(const std::string& encoded_str) {
        return encoded_str;
    }
}


// CoreEngine 构造函数
CoreEngine::CoreEngine() : is_running_(false) {
    server_.on_open = [this](const std::string& uuid) { this->handle_connection_open(uuid); };
    server_.on_message = [this](const std::string& uuid, std::string_view msg) { this->handle_connection_message(uuid, msg); };
    server_.on_close = [this](const std::string& uuid) { this->handle_connection_close(uuid); };

    std::cout << "[CoreEngine] Initializing AI Service Bridge..." << std::endl;
    ai_services_ = std::make_unique<AIServiceManager>();
    std::cout << "[CoreEngine] AI Service Bridge Initialized." << std::endl;
}

// CoreEngine 析构函数
CoreEngine::~CoreEngine() {
    is_running_ = false;
    if (processing_thread_.joinable()) {
        processing_thread_.join();
    }
}

// CoreEngine run 方法
void CoreEngine::run() {
    is_running_ = true;
    processing_thread_ = std::thread(&CoreEngine::data_processing_loop, this);

    int port = 5002; // TODO: 从配置文件读取
    std::cout << "[CoreEngine] Starting WebSocket server on port " << port << "..." << std::endl;
    server_.run(port);

    is_running_ = false;
    if (processing_thread_.joinable()) {
        processing_thread_.join();
    }
}

// handle_connection_open 方法
void CoreEngine::handle_connection_open(const std::string& client_uuid) {
    // TODO: 从配置中读取这些模型路径
    std::string kws_model_path = "kws_models/porcupine_params_zh.pv";
    std::vector<std::string> keyword_paths = {"kws_models/xiaotong_zh_linux_v3_0_0.ppn"};//truman这里要改
    
    ClientManager::getInstance().addClient(client_uuid, kws_model_path, keyword_paths);
}

// handle_connection_message 方法
void CoreEngine::handle_connection_message(const std::string& client_uuid, std::string_view message) {
    incoming_data_queue_.push({client_uuid, std::string(message)});
}

// handle_connection_close 方法
void CoreEngine::handle_connection_close(const std::string& client_uuid) {
    ClientManager::getInstance().removeClient(client_uuid);
}

// 【核心修改】data_processing_loop 的全新实现
void CoreEngine::data_processing_loop() {
    std::cout << "[CoreEngine] Data processing thread started." << std::endl;
    while (is_running_) {
        DataPacket packet;
        if (incoming_data_queue_.try_pop(packet)) {
            
            auto client = ClientManager::getInstance().getClient(packet.client_uuid);
            if (!client) {
                std::cerr << "[ProcessingThread] Could not find client for UUID: " << packet.client_uuid << std::endl;
                continue;
            }

            std::string decrypted_json_str = YourUtils::decrypt_from_string(packet.message);
            auto data = json::parse(decrypted_json_str, nullptr, false);
            if (data.is_discarded()) {
                std::cerr << "[ProcessingThread] JSON parse failed for message from " << packet.client_uuid << std::endl;
                continue;
            }
            
            double client_timestamp = data.value("timestamp", 0.0);

            // SUGGESTION: 在这里实现 "Smart Skip" 逻辑，检查队列长度和时间戳延迟，
            //             如果延迟过高，可以选择 continue 跳过处理此数据包。

            if (data.contains("audio_chunk")) {
                std::string audio_b64 = data["audio_chunk"];
                std::string audio_bytes_str = YourUtils::base64_decode(audio_b64);
                std::vector<char> audio_bytes(audio_bytes_str.begin(), audio_bytes_str.end());

                // FIX: 使用新的Client接口，它返回一个包含音频和KWS状态的optional<pair>
                auto process_result = client->processAudioChunk(audio_bytes, client_timestamp);

                if (process_result) {
                    // 解包结果
                    auto& [dialogue_bytes_vec, kws_was_triggered] = process_result.value();
                    
                    std::cout << "\n[CoreEngine] Complete dialogue ready for client " << packet.client_uuid 
                              << ". Sending " << dialogue_bytes_vec.size() << " bytes to Python ASR..." << std::endl;
                    
                    // 将vector<char>转换为string以传递给pybind
                    std::string dialogue_str(dialogue_bytes_vec.begin(), dialogue_bytes_vec.end());
                    
                    std::string transcript = ai_services_->transcribe(dialogue_str, 16000); // 跨语言调用

                    std::cout << "=================================================" << std::endl;
                    // FIX: 使用解包后的 kws_was_triggered 变量
                    std::cout << "[C++] ASR Result (" << (kws_was_triggered ? "KWS Triggered" : "No KWS") << "):" << std::endl;
                    std::cout << ">>> " << transcript << std::endl;
                    std::cout << "=================================================\n" << std::endl;

                    // TODO: 在这里根据ASR结果和KWS状态，决定是否调用LLM服务
                }
            }

            if (data.contains("video_frame")) {
                std::string video_b64 = data["video_frame"];
                std::string video_bytes_str = YourUtils::base64_decode(video_b64);
                std::vector<char> video_bytes(video_bytes_str.begin(), video_bytes_str.end());

                KeyframeDetector* detector = client->getKeyframeDetector();
                if (detector && detector->is_keyframe(video_bytes)) {
                    std::cout << "[CoreEngine] Keyframe DETECTED for client " << packet.client_uuid << "!" << std::endl;
                    // TODO: 在这里触发事件记录等后续操作
                }
                
                // TODO: 在这里可以添加调用人脸识别服务的逻辑
            }

        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    std::cout << "[CoreEngine] Data processing thread stopped." << std::endl;
}