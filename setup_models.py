#!/usr/bin/env python3
"""
下载 argostranslate 翻译语言包（仅需运行一次，之后完全离线）
"""
from argostranslate import package

PAIRS = [
    ("en", "zh"),  # 英文 → 中文
    ("zh", "en"),
    ("ja", "en"),  # 日文 → 英文（中转到中文）
    ("ko", "en"),
    ("fr", "en"),
    ("de", "en"),
    ("es", "en"),
    ("it", "en"),
    ("pt", "en"),
    ("ru", "en"),
    ("ar", "en"),
]

print("正在获取语言包列表（需要网络，仅此一次）…")
package.update_package_index()
available = package.get_available_packages()

installed = 0
for pkg in available:
    pair = (pkg.from_code, pkg.to_code)
    if pair in PAIRS:
        print(f"  安装 {pkg.from_name} → {pkg.to_name} …", end=" ", flush=True)
        package.install_from_path(pkg.download())
        print("✓")
        installed += 1

print(f"\n完成，共安装 {installed} 个语言包。之后可完全离线运行。")
