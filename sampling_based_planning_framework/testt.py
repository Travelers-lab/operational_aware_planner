import sys
import os
from pathlib import Path


def diagnose_import_issue(module_name: str):
    """诊断为什么无法导入模块"""
    print(f"🔍 Diagnosing import issue for: {module_name}")
    print("=" * 50)

    # 1. 检查当前目录
    current_dir = Path.cwd()
    print(f"Current directory: {current_dir}")
    print(f"In sys.path: {'' in sys.path or str(current_dir) in sys.path}")

    # 2. 检查模块目录是否存在
    module_dir = current_dir / module_name
    print(f"\nModule directory: {module_dir}")
    print(f"Exists: {module_dir.exists()}")
    print(f"Is directory: {module_dir.is_dir() if module_dir.exists() else 'N/A'}")

    # 3. 检查 __init__.py 文件
    if module_dir.exists() and module_dir.is_dir():
        init_file = module_dir / "__init__.py"
        print(f"__init__.py exists: {init_file.exists()}")

        # 4. 检查目录内容
        print(f"\nContents of {module_name}/:")
        for item in module_dir.iterdir():
            if item.is_file():
                print(f"  📄 {item.name}")
            elif item.is_dir():
                print(f"  📁 {item.name}/")

    # 5. 尝试不同的导入方式
    print(f"\n🧪 Testing import approaches:")

    # 方式1: 直接导入包
    try:
        module = __import__(module_name)
        print(f"✅ Direct import: SUCCESS - {module}")
        print(f"   Module file: {getattr(module, '__file__', 'Unknown')}")
        return True
    except ImportError as e:
        print(f"❌ Direct import: FAILED - {e}")

    # 方式2: 使用 importlib
    try:
        import importlib
        module = importlib.import_module(module_name)
        print(f"✅ Importlib import: SUCCESS - {module}")
        return True
    except ImportError as e:
        print(f"❌ Importlib import: FAILED - {e}")

    # 方式3: 检查是否是子模块问题
    try:
        # 假设我们要从 motion_intent.recognizer 导入 MotionIntentRecognizer
        module = __import__(f"{module_name}.recognizer", fromlist=['MotionIntentRecognizer'])
        print(f"✅ Submodule import: SUCCESS - {module}")
        return True
    except ImportError as e:
        print(f"❌ Submodule import: FAILED - {e}")

    return False


# 运行诊断
diagnose_import_issue("model/map_generation")