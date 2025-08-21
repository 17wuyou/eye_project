#include <Python.h>
#include <iostream>
#include "CoreEngine.h"

int main() {
    std::cout << "--- Main: Initializing CoreEngine... ---" << std::endl;

    try {
        // 1️⃣ 指定 PythonHome 为完整安装目录
        Py_SetPythonHome(L"C:\\Users\\12429\\AppData\\Local\\Programs\\Python\\Python312");

        // 2️⃣ 初始化 Python
        if (!Py_IsInitialized()) {
            Py_Initialize();
        }

        // 3️⃣ 添加路径：
        // - 完整安装的 Lib 和 site-packages
        // - 虚拟环境 site-packages
        // - 项目模块
        PyRun_SimpleString(
            "import sys\n"
            "sys.path.append(r'C:\\Users\\12429\\AppData\\Local\\Programs\\Python\\Python312\\Lib')\n"
            "sys.path.append(r'C:\\Users\\12429\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages')\n"
            "sys.path.append(r'D:\\our_weax2_project\\Backend_lin\\glassEnv\\Lib\\site-packages')\n"
            "sys.path.append(r'D:\\eye8.18\\EyeTeach')\n"
        );

        // 4️⃣ 测试导入 _ctypes
        PyRun_SimpleString(
            "try:\n"
            "    import _ctypes\n"
            "    print('_ctypes imported successfully')\n"
            "except Exception as e:\n"
            "    print('Failed to import _ctypes:', e)\n"
        );

        // 5️⃣ 测试导入 modules
        PyRun_SimpleString(
            "try:\n"
            "    import modules\n"
            "    print('modules imported successfully')\n"
            "except Exception as e:\n"
            "    print('Failed to import modules:', e)\n"
        );

        // 6️⃣ 运行 CoreEngine
        CoreEngine engine;
        engine.run();

        // 7️⃣ 结束 Python
        Py_Finalize();
    }
    catch (const std::exception& e) {
        std::cerr << "An unhandled exception reached main: " << e.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "An unknown exception reached main." << std::endl;
        return 1;
    }

    std::cout << "--- Main: CoreEngine has shut down cleanly. ---" << std::endl;
    return 0;
}
