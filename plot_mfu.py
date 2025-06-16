#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from pathlib import Path


def parse_log_line(line):
    """解析日志行，提取迭代次数、elapsed time、token throughput、mfu和lm loss"""
    # 匹配新的日志格式
    iteration_match = re.search(r'iteration\s+(\d+)/\s*\d+', line)
    elapsed_time_match = re.search(r'elapsed time per iteration \(ms\): ([\d.]+)', line)
    token_throughput_match = re.search(r'token throughput per GPU \(tokens/s/GPU\): ([\d.]+)', line)
    lm_loss_match = re.search(r'lm loss: ([\d.]+E[+-]\d+)', line)
    # 尝试匹配直接提供的MFU值，兼容两种格式：MFU: X.X% 或 某前缀MFU: X.X%
    mfu_match = re.search(r'(?:(\w+))?MFU: ([\d.]+)%', line)
    
    if not (iteration_match and elapsed_time_match and token_throughput_match):
        return None
    
    iteration = int(iteration_match.group(1))
    elapsed_time_ms = float(elapsed_time_match.group(1))
    token_throughput = float(token_throughput_match.group(1))
    
    # 转换为秒
    elapsed_time_s = elapsed_time_ms / 1000.0
    
    result = {
        'iteration': iteration,
        'elapsed_time_s': elapsed_time_s,
        'token_throughput': token_throughput,
        'has_mfu': False  # 标志是否直接从日志中读取了MFU值
    }
    
    # 如果找到直接提供的MFU值
    if mfu_match:
        mfu_value = float(mfu_match.group(2)) / 100.0  # 转换为小数
        result['mfu'] = mfu_value
        result['has_mfu'] = True
        mfu_type = mfu_match.group(1) or ""  # 如果group(1)为None，则使用空字符串
        result['mfu_type'] = mfu_type  # 记录MFU的类型
    
    # 如果找到lm_loss，则添加到结果中
    if lm_loss_match:
        result['lm_loss'] = float(lm_loss_match.group(1))
    
    return result


def calculate_mfu(elapsed_time_s, token_throughput):
    """计算MFU值
    MFU = (6*21000000000*2400*4096)/(989000000000000*240*X)+(6.144*4096*60)*Y*10000/989000000000000
    其中X是elapsed time per iteration in second，Y是token throughput per GPU
    """
    term1 = (6 * 21000000000 * 2400 * 4096) / (989000000000000 * 240 * elapsed_time_s)
    term2 = (6.144 * 4096 * 60) * token_throughput * 10000 / 989000000000000
    return term1 + term2


def moving_average(data, window_size=50):
    """计算移动平均"""
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')


def parse_log_file(file_path, show_lm_loss=False):
    """解析日志文件，提取迭代次数、MFU值和lm loss值"""
    iterations = []
    mfu_values = []
    lm_loss_values = [] if show_lm_loss else None
    mfu_source = None  # 用于记录MFU的来源：'direct' 或 'calculated'
    
    with open(file_path, 'r') as f:
        for line in f:
            parsed = parse_log_line(line)
            if parsed:
                if parsed['has_mfu']:
                    # 如果日志中直接提供了MFU值，则使用它
                    mfu = parsed['mfu']
                    if mfu_source is None:
                        mfu_source = f"direct ({parsed['mfu_type']})"
                else:
                    # 否则计算MFU值
                    mfu = calculate_mfu(parsed['elapsed_time_s'], parsed['token_throughput'])
                    if mfu_source is None:
                        mfu_source = 'calculated'
                
                iterations.append(parsed['iteration'])
                mfu_values.append(mfu)
                
                if show_lm_loss and 'lm_loss' in parsed:
                    lm_loss_values.append(parsed['lm_loss'])
    
    print(f"文件 {file_path} 中的MFU来源: {mfu_source}")
    
    if show_lm_loss:
        return iterations, mfu_values, lm_loss_values
    else:
        return iterations, mfu_values


def calculate_ema(data, alpha=0.9):
    """计算指数移动平均 (EMA)
    alpha: 平滑系数，范围[0,1]，值越大表示对最近的数据权重越大
    """
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    return ema


def plot_mfu_curves(file_paths, title, show_lm_loss=False, show_mfu=True, lm_loss_ylim=None, show_loss_ema=False):
    """绘制多个文件的MFU曲线和lm loss曲线"""
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # 创建第二个y轴（如果需要显示lm_loss）
    ax2 = ax1.twinx() if show_lm_loss else None
    
    # 根据文件名前缀（_分割的第一部分）对文件进行分组
    file_groups = {}
    for file_path in file_paths:
        file_name = Path(file_path).name
        prefix = file_name.split('_')[0] if '_' in file_name else file_name
        if prefix not in file_groups:
            file_groups[prefix] = []
        file_groups[prefix].append(file_path)
    
    # 为每个组分配一个基础颜色
    base_colors = plt.cm.tab10.colors
    
    # 绘制每个组的文件
    group_index = 0
    for prefix, group_files in file_groups.items():
        base_color = base_colors[group_index % len(base_colors)]
        
        # 根据组内文件数量计算颜色深浅的步长
        num_files = len(group_files)
        
        for i, file_path in enumerate(group_files):
            # 计算颜色深浅，从浅到深
            alpha = 0.3 + 0.7 * (i / max(1, num_files - 1)) if num_files > 1 else 1.0
            color = to_rgba(base_color, alpha)
            
            # 解析日志文件
            if show_lm_loss:
                iterations, mfu_values, lm_loss_values = parse_log_file(file_path, show_lm_loss=True)
            else:
                iterations, mfu_values = parse_log_file(file_path, show_lm_loss=False)
            
            if not iterations:
                print(f"警告: 文件 {file_path} 中未找到有效数据")
                continue
            
            # 获取文件名作为图例
            file_name = Path(file_path).name
            
            # 绘制MFU曲线（在第一个y轴）
            if show_mfu:
                # 将MFU值转换为百分比
                mfu_values_percent = [mfu * 100 for mfu in mfu_values]
                line1 = ax1.plot(iterations, mfu_values_percent, color=color, linewidth=2, label=f'MFU: {file_name}')
                
                # 找出最高点并标注
                if mfu_values_percent:
                    max_index = np.argmax(mfu_values_percent)
                    max_iteration = iterations[max_index]
                    max_mfu = mfu_values_percent[max_index]
                    
                    # 每隔50轮标注一次MFU值
                    for idx, (iter_num, mfu_val) in enumerate(zip(iterations, mfu_values_percent)):
                        if iter_num % 250 == 0:
                            ax1.annotate(f'{mfu_val:.1f}%', 
                                        xy=(iter_num, mfu_val),
                                        xytext=(0, 10),  # 文本偏移量（向上偏移）
                                        textcoords='offset points',
                                        color=color,
                                        fontsize=8)
            
            # 绘制lm loss曲线（在第二个y轴，如果需要）
            if show_lm_loss and lm_loss_values and len(lm_loss_values) == len(iterations):
                # 使用相同颜色但透明度更低
                loss_color = to_rgba(base_color, alpha * (0.3 if show_loss_ema else 0.5))
                # 只在显示EMA时不显示原始loss的图例
                line2 = ax2.plot(iterations, lm_loss_values, color=loss_color, linewidth=1.5, linestyle='--', 
                               label=f'Loss: {file_name}' if not show_loss_ema else None)
                
                # 如果启用了loss EMA，绘制EMA曲线
                if show_loss_ema:
                    loss_ema_values = calculate_ema(lm_loss_values)
                    # 使用相同颜色但更深的线条绘制EMA
                    loss_ema_color = to_rgba(base_color, min(alpha * 0.5 + 0.3, 1.0))
                    ax2.plot(iterations, loss_ema_values, color=loss_ema_color, linewidth=1.0, 
                            label=f'Loss: {file_name}', linestyle='-')
                
                # 如果显示lm_loss，每隔50轮标注一次lm_loss值
                for idx, (iter_num, lm_loss_val) in enumerate(zip(iterations, lm_loss_values)):
                    if iter_num % 250 == 0:
                        ax2.annotate(f'{lm_loss_val:.4f}', 
                                    xy=(iter_num, lm_loss_val),
                                    xytext=(0, -10),  # 文本偏移量（向下偏移）
                                    textcoords='offset points',
                                    color=loss_color,
                                    fontsize=8)
        
        group_index += 1
    
    # 设置第一个y轴（MFU）
    if show_mfu:
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('MFU (%)')
        ax1.set_ylim(0.0, 35.0)  # 改为百分比范围
        # 设置y轴刻度格式为百分比
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}%'))
        ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 设置第二个y轴（lm loss，如果需要）
    if show_lm_loss and ax2:
        ax2.set_ylabel('LM Loss')
        if lm_loss_ylim:
            ax2.set_ylim(lm_loss_ylim)
    
    # 设置标题
    plt.title(title)
    
    # 合并两个轴的图例
    if show_lm_loss and ax2:
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    else:
        ax1.legend(loc='upper right')
    
    # 保存图像
    plt.savefig('mfu_comparison.png', dpi=300, bbox_inches='tight')
    print(f"图像已保存为 mfu_comparison.png")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='绘制多个日志文件的MFU曲线')
    parser.add_argument('title', help='图表标题')
    parser.add_argument('files', nargs='+', help='日志文件路径')
    parser.add_argument('--show-lm-loss', action='store_true', default=False, 
                        help='是否显示lm_loss曲线（默认不显示）')
    parser.add_argument('--no-show-mfu', action='store_true', default=False,
                        help='是否隐藏MFU曲线（默认显示）')
    parser.add_argument('--lm-loss-ylim', type=float, nargs=2, metavar=('MIN', 'MAX'),
                        help='设置lm-loss的y轴范围，例如：--lm-loss-ylim 0.5 2.0')
    parser.add_argument('--show-loss-ema', action='store_true', default=False,
                        help='是否显示loss的指数移动平均曲线（默认不显示）')
    args = parser.parse_args()
    
    plot_mfu_curves(args.files, args.title, 
                    show_lm_loss=args.show_lm_loss, 
                    show_mfu=not args.no_show_mfu,
                    lm_loss_ylim=args.lm_loss_ylim,
                    show_loss_ema=args.show_loss_ema)


if __name__ == "__main__":
    main()
