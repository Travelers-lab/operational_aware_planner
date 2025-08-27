import sys
print(sys.path)
from sampling_based_planning_framework.sampling_based_interactive_planner
def diagnose_and_fix_path_issues():
    """
    Comprehensive diagnosis and fix for Python path issues.
    """
    import sys
    from pathlib import Path

    print("🔍 Python Path Diagnosis")
    print("=" * 50)

    # 当前路径分析
    current_dir = Path.cwd()
    print(f"Current directory: {current_dir}")
    print(f"In sys.path: {'✅' if '' in sys.path or str(current_dir) in sys.path else '❌'}")

    # 检查项目结构
    print("\n📁 Project structure:")
    python_dirs = []
    for item in current_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            py_files = list(item.glob("*.py"))
            has_init = (item / "__init__.py").exists()
            status = "🐍" if py_files or has_init else "📁"
            print(f"  {status} {item.name}/ - {len(py_files)} py files, init: {has_init}")
            if py_files or has_init:
                python_dirs.append(item)

    # 添加缺失的路径
    print("\n➕ Adding missing paths:")
    paths_to_add = [current_dir] + python_dirs
    added_count = 0

    for path in paths_to_add:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
            print(f"  ✅ Added: {path_str}")
            added_count += 1
        else:
            print(f"  ℹ️  Already in path: {path_str}")

    print(f"\nTotal paths added: {added_count}")

    # 测试导入
    print("\n🧪 Testing imports:")
    modules_to_test = ["motion_intent", "multi_dimension_perception"]  # 添加您需要导入的模块

    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"  ✅ {module_name}: SUCCESS")
        except ImportError as e:
            print(f"  ❌ {module_name}: FAILED - {e}")

    return added_count > 0


# 运行诊断和修复
# if __name__ == "__main__":
#     success = diagnose_and_fix_path_issues()
#     if success:
#         print("\n🎉 Path issues fixed! You can now import your modules.")
#     else:
#         print("\n⚠️  Some issues may still exist. Check your project structure.")