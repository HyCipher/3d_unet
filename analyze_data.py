"""
数据分析脚本：检查正类样本的比例，帮助调整损失函数参数
以及分析3D tif文件的维度顺序
"""
import glob
import os
import numpy as np
import tifffile as tiff
import sys


def analyze_tif_dimensions(tif_path):
    """
    分析3D tif文件的维度顺序，帮助确定XYZ轴
    """
    if not os.path.exists(tif_path):
        print(f"❌ 文件不存在: {tif_path}")
        return None

    print(f"\n📐 分析 3D TIF 文件维度: {os.path.basename(tif_path)}\n")
    print("=" * 70)

    try:
        data = tiff.imread(tif_path).astype(np.float32)
        print(f"✅ 成功加载文件")
        print(f"\n📊 维度信息:")
        print(f"   - 原始形状 (shape): {data.shape}")
        print(f"   - 数据类型: {data.dtype}")
        print(f"   - 总体素数: {data.size:,}")
        print(f"   - 数据范围: [{data.min():.2f}, {data.max():.2f}]")

        # 分析三个维度
        dims = data.shape
        if len(dims) == 3:
            d0, d1, d2 = dims
            print(f"\n📌 三个维度分析:")
            print(f"   - 维度0 (axis 0): 大小 {d0}")
            print(f"   - 维度1 (axis 1): 大小 {d1}")
            print(f"   - 维度2 (axis 2): 大小 {d2}")

            # 根据大小关系给出建议
            dim_list = [(0, d0), (1, d1), (2, d2)]
            dim_list_sorted = sorted(dim_list, key=lambda x: x[1])

            print(f"\n💡 按大小排序:")
            print(f"   - 最小维度 (axis {dim_list_sorted[0][0]}): 大小 {dim_list_sorted[0][1]} <- 可能是 Z轴 (depth)")
            print(f"   - 中间维度 (axis {dim_list_sorted[1][0]}): 大小 {dim_list_sorted[1][1]} <- 可能是 Y轴 (height)")
            print(f"   - 最大维度 (axis {dim_list_sorted[2][0]}): 大小 {dim_list_sorted[2][1]} <- 可能是 X轴 (width)")

            print(f"\n🔄 当前训练代码中的转换:")
            print(f"   - 输入 TIF 形状: (H, W, Z)")
            print(f"   - 转换后形状: (Z, H, W) 使用 np.transpose(vol, (2, 0, 1))")
            print(f"   - 即: axis 2 -> axis 0 (Z layer作为第一个维度)")

            print(f"\n📋 你的文件对应:")
            if d0 < d1 and d0 < d2:
                print(f"   ✓ axis 0 ({d0}) -> H (height/行)")
                print(f"   ✓ axis 1 ({d1}) -> W (width/列)")
                print(f"   ✓ axis 2 ({d2}) -> Z (depth/层) <- 最大维度")
                order_interpretation = "(H, W, Z)"
            else:
                print(f"   ❓ axis 0 ({d0})")
                print(f"   ❓ axis 1 ({d1})")
                print(f"   ❓ axis 2 ({d2})")
                order_interpretation = "需要人工确认"

            print(f"\n✅ 当前文件的维度顺序: {order_interpretation}")
            print(f"   形状: {data.shape}")

            # 显示切片预览
            print(f"\n🔍 切片信息预览:")
            mid_z = d0 // 2
            print(f"   - 第 {mid_z} 层切片形状: {data[mid_z].shape}")
            print(f"   - 该切片统计: mean={data[mid_z].mean():.2f}, std={data[mid_z].std():.2f}")

        else:
            print(f"❌ 不是3D数据，维度数: {len(dims)}")

    except Exception as e:
        print(f"❌ 读取文件错误: {e}")

    print("=" * 70)


def analyze_class_distribution(label_dir):
    """Analyze the distribution of positive vs negative samples."""
    label_paths = sorted(glob.glob(os.path.join(label_dir, "*.tif")))

    total_pixels = 0
    positive_pixels = 0
    volume_stats = []

    print(f"\n📊 分析 {len(label_paths)} 个标签文件\n")
    print(f"{'Volume':<40} {'Positive %':>12} {'Pixels':>15}")
    print("-" * 68)

    for label_path in label_paths:
        lab = tiff.imread(label_path).astype(np.float32)
        total = lab.size
        positive = np.sum(lab > 0)

        pos_percent = 100 * positive / total
        volume_stats.append(pos_percent)
        total_pixels += total
        positive_pixels += positive

        print(f"{os.path.basename(label_path):<40} {pos_percent:>11.2f}% {total:>15}")

    overall_positive_ratio = positive_pixels / total_pixels
    print("-" * 68)
    print(f"{'OVERALL':<40} {100*overall_positive_ratio:>11.2f}% {total_pixels:>15}\n")

    print(f"✅ 统计信息:")
    print(f"   - 总像素数: {total_pixels:,}")
    print(f"   - 正类像素数: {positive_pixels:,}")
    print(f"   - 负类像素数: {total_pixels - positive_pixels:,}")
    print(f"   - 类不平衡比例: {(1-overall_positive_ratio)/overall_positive_ratio:.1f}:1 (负:正)")

    # 建议参数
    print(f"\n💡 参数建议:")
    if overall_positive_ratio < 0.01:
        print(f"   ⚠️  严重类不平衡 (正类 < 1%)")
        print(f"   推荐使用: Focal Loss (gamma=2.0, alpha=0.25)")
        print(f"   或者: DiceFocal Loss (dice_weight=0.8, focal_weight=1.0)")
    elif overall_positive_ratio < 0.05:
        print(f"   ⚠️  类不平衡 (正类 < 5%)")
        print(f"   推荐使用: DiceFocal Loss (dice_weight=0.6, focal_weight=1.0)")
    elif overall_positive_ratio < 0.2:
        print(f"   ℹ️  中度类不平衡 (正类 < 20%)")
        print(f"   推荐使用: DiceFocal Loss (dice_weight=0.5, focal_weight=1.0)")
    else:
        print(f"   ✓  类相对均衡")
        print(f"   推荐使用: BCE或简单Focal Loss")

    return overall_positive_ratio, volume_stats


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🔬 3D UNet 数据分析工具")
    print("=" * 70)

    if len(sys.argv) > 1:
        # 命令行模式
        if sys.argv[1] == "dims" and len(sys.argv) > 2:
            # 分析维度: python analyze_data.py dims <tif_file_path>
            tif_path = sys.argv[2]
            analyze_tif_dimensions(tif_path)

        elif sys.argv[1] == "class":
            # 分析类分布: python analyze_data.py class <label_dir>
            label_dir = sys.argv[2] if len(sys.argv) > 2 else "data/training/labels"
            ratio, stats = analyze_class_distribution(label_dir)

        elif sys.argv[1] == "all":
            # 运行全部分析
            print("\n📊 运行完整分析...\n")
            
            # 分析一个示例文件
            example_file = "data/training/images/dauer_img_160_180.tif"
            if os.path.exists(example_file):
                analyze_tif_dimensions(example_file)
            
            # 分析类分布
            ratio, stats = analyze_class_distribution("data/training/labels")
            print(f"\n🔄 运行完成！根据上述建议更新 config.py 中的损失函数参数。")

        else:
            print("❌ 未知命令")
            print("\n💡 使用方法:")
            print("   python analyze_data.py dims <tif_file_path>  # 分析某个TIF文件的维度")
            print("   python analyze_data.py class [label_dir]    # 分析类不平衡情况")
            print("   python analyze_data.py all                  # 运行全部分析")

    else:
        # 默认模式：运行完整分析
        print("\n📊 运行完整分析...\n")

        # 分析一个示例文件
        example_file = "data/training/images/dauer_img_160_180.tif"
        if os.path.exists(example_file):
            analyze_tif_dimensions(example_file)
        else:
            print(f"⚠️  示例文件不存在: {example_file}")

        # 分析类分布
        ratio, stats = analyze_class_distribution("data/training/labels")
        print(f"\n🔄 运行完成！根据上述建议更新 config.py 中的损失函数参数。")

        print("\n💡 快速命令:")
        print("   python analyze_data.py dims <你的tif文件>  # 分析特定文件的维度")
        print("=" * 70)
