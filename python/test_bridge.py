# python/test_bridge.py
import sys
import os

# 将编译产物的路径添加到Python搜索路径中
# 这使得我们可以直接import C++模块
# 注意：根据您的操作系统和编译设置，build目录可能在上一级
build_path = os.path.join(os.path.dirname(__file__), '..', 'build')
sys.path.append(build_path)

try:
    # 导入我们用C++创建的模块，模块名在CMakeLists.txt中定义
    import my_project_bridge
    print("Python: 成功导入 C++ 模块 'my_project_bridge'。")
except ImportError as e:
    print(f"Python: 导入 C++ 模块失败！请确保：")
    print(f"1. 项目已成功编译。")
    print(f"2. 模块文件（.so 或 .pyd）位于 '{build_path}' 目录中。")
    print(f"错误信息: {e}")
    sys.exit(1)

print("\n--- 开始测试：从 Python 调用 C++ ---")
message_from_cpp = my_project_bridge.say_hello_from_cpp()
print(f"Python: 从 C++ 函数收到的消息: '{message_from_cpp}'")
print("--- 测试结束：从 Python 调用 C++ ---\n")