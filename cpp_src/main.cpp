#include <pybind11/pybind11.h>
#include <pybind11/stl.h>      // ���� vector, pair, deque ��
#include <string_view>

#include "audio_utils.h"
#include "keyframe_detector.h"

namespace py = pybind11;

// �궨��ģ����ڵ�
// "my_project_cpp" ������ CMakeLists.txt �е�ģ����һ��
PYBIND11_MODULE(my_project_cpp, m) {
    m.doc() = "High-performance C++ backend for the project";

    // �� audio_utils.h �еĺ���
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

    // �� KeyframeDetector ��
    py::class_<KeyframeDetector>(m, "KeyframeDetector")
        .def(py::init([](
            double threshold, int history_size, std::pair<int, int> resize_dim,
            double motion_threshold_percent, int motion_diff_threshold_value,
            double cooldown_period, bool use_h_channel,
            py::dict hist_params
            ) {
                // �� Python dict �н���ֱ��ͼ�������ṩ�����
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
            py::call_guard<py::gil_scoped_release>(), // �ؼ����ͷ�GIL��������߳�
            "Checks if the current frame is a keyframe. Releases the GIL."
        );
}