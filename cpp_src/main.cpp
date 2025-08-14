#include <pybind11/pybind11.h>
#include <pybind11/stl.h>      // 用于 vector, pair, deque 等
#include <string_view>

#include "audio_utils.h"
#include "keyframe_detector.h"

namespace py = pybind11;

// 宏定义模块入口点
// "my_project_cpp" 必须与 CMakeLists.txt 中的模块名一致
PYBIND11_MODULE(my_project_cpp, m) {
    m.doc() = "High-performance C++ backend for the project";

    // 绑定 audio_utils.h 中的函数
    m.def(
        "is_audio_active_by_rms",
        [](const py::bytes& audio_bytes, int sample_width, double rms_threshold) {
            std::string_view bytes_view(audio_bytes);
            std::vector<unsigned char> byte_vec(bytes_view.begin(), bytes_view.end());
            return is_audio_active_by_rms(byte_vec, sample_width, rms_threshold);
        },
        py::arg("audio_chunk_bytes"),
        py::arg("sample_width"),
        py::arg("rms_threshold"),
        "A C++ implementation of RMS-based Voice Activity Detection."
    );

    // 绑定 KeyframeDetector 类
    py::class_<KeyframeDetector>(m, "KeyframeDetector")
        .def(py::init([](
            double threshold, int history_size, std::pair<int, int> resize_dim,
            double motion_threshold_percent, int motion_diff_threshold_value,
            double cooldown_period, bool use_h_channel,
            py::dict hist_params
            ) {
                // 从 Python dict 中解析直方图参数，提供灵活性
                std::vector<int> channels, size;
                std::vector<float> ranges;

                if (use_h_channel) {
                    channels = hist_params["hist_channels_h"].cast<std::vector<int>>();
                    size = hist_params["hist_size_h"].cast<std::vector<int>>();
                    py::list ranges_list = hist_params["hist_ranges_h"];
                    ranges.push_back(ranges_list[0].cast<float>());
                    ranges.push_back(ranges_list[1].cast<float>());
                }
                else {
                    channels = hist_params["hist_channels_gray"].cast<std::vector<int>>();
                    size = hist_params["hist_size_gray"].cast<std::vector<int>>();
                    py::list ranges_list = hist_params["hist_ranges_gray"];
                    ranges.push_back(ranges_list[0].cast<float>());
                    ranges.push_back(ranges_list[1].cast<float>());
                }

                return std::make_unique<KeyframeDetector>(
                    threshold, history_size, resize_dim,
                    motion_threshold_percent, motion_diff_threshold_value,
                    cooldown_period, use_h_channel,
                    channels, size, ranges
                );
            }),
            py::arg("threshold"), py::arg("history_size_config"),
            py::arg("resize_dim"), py::arg("motion_threshold_percent"),
            py::arg("motion_diff_threshold_value"), py::arg("cooldown_period"),
            py::arg("use_h_channel"), py::arg("hist_params"),
            "C++ implementation of the KeyframeDetector."
        )
        .def("is_keyframe",
            [](KeyframeDetector& self, const py::bytes& frame_bytes) {
                std::string_view bytes_view(frame_bytes);
                std::vector<unsigned char> byte_vec(bytes_view.begin(), bytes_view.end());
                return self.is_keyframe(byte_vec);
            },
            py::arg("current_frame_bytes"),
            py::call_guard<py::gil_scoped_release>(), // 关键！释放GIL，允许多线程
            "Checks if the current frame is a keyframe. Releases the GIL."
        );
}