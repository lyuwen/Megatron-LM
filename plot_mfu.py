#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from pathlib import Path


def format_consumed_tokens(tokens):
    """格式化consumed tokens的显示"""
    if tokens >= 1e12:
        return f'{tokens/1e12:.1f}T'
    elif tokens >= 1e9:
        return f'{tokens/1e9:.1f}B'
    elif tokens >= 1e6:
        return f'{tokens/1e6:.1f}M'
    elif tokens >= 1e3:
        return f'{tokens/1e3:.1f}K'
    else:
        return f'{int(tokens)}'


def parse_log_line(line):
    """解析日志行，提取迭代次数、elapsed time、token throughput、mfu和lm loss"""
    # 匹配新的日志格式
    iteration_match = re.search(r'iteration\s+(\d+)/\s*\d+', line)
    elapsed_time_match = re.search(r'elapsed time per iteration \(ms\): ([\d.]+)', line)
    token_throughput_match = re.search(r'token throughput per GPU \(tokens/s/GPU\): ([\d.]+)', line)
    lm_loss_match = re.search(r'lm loss: ([\d.]+E[+-]\d+)', line)
    # 尝试匹配直接提供的MFU值，兼容两种格式：MFU: X.X% 或 某前缀MFU: X.X%
    mfu_match = re.search(r'(?:(\w+))?MFU: ([\d.]+)%', line)
    # 尝试匹配consumed tokens (支持逗号分隔的数字)
    consumed_tokens_match = re.search(r'consumed tokens:\s*([\d,]+)', line)
    # 尝试匹配global batch size
    global_batch_size_match = re.search(r'global batch size: (\d+)', line)
    
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
    
    # 如果找到consumed tokens，则添加到结果中
    if consumed_tokens_match:
        consumed_str = consumed_tokens_match.group(1)
        # 移除逗号并转换为数字
        consumed_str_clean = consumed_str.replace(',', '')
        # 处理K, M, B后缀
        if consumed_str_clean.endswith('K'):
            result['consumed_tokens'] = float(consumed_str_clean[:-1]) * 1e3
        elif consumed_str_clean.endswith('M'):
            result['consumed_tokens'] = float(consumed_str_clean[:-1]) * 1e6
        elif consumed_str_clean.endswith('B'):
            result['consumed_tokens'] = float(consumed_str_clean[:-1]) * 1e9
        else:
            result['consumed_tokens'] = float(consumed_str_clean)
    
    # 如果找到global batch size，则添加到结果中
    if global_batch_size_match:
        result['global_batch_size'] = int(global_batch_size_match.group(1))
    
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


def parse_log_file(file_path, show_lm_loss=False, global_batch_size=None):
    """解析日志文件，提取迭代次数、MFU值和lm loss值"""
    iterations = []
    consumed_tokens = []
    mfu_values = []
    lm_loss_values = [] if show_lm_loss else None
    mfu_source = None  # 用于记录MFU的来源：'direct' 或 'calculated'
    detected_global_batch_size = None
    
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
                
                # 尝试获取global_batch_size
                if 'global_batch_size' in parsed and detected_global_batch_size is None:
                    detected_global_batch_size = parsed['global_batch_size']
                
                # 计算consumed tokens
                if 'consumed_tokens' in parsed:
                    # 如果日志中直接有consumed tokens，直接使用
                    consumed_token_value = parsed['consumed_tokens']
                else:
                    # 否则通过iteration * global_batch_size计算
                    batch_size = global_batch_size or detected_global_batch_size
                    if batch_size:
                        consumed_token_value = parsed['iteration'] * batch_size
                    else:
                        # 如果没有batch size信息，使用iteration作为备选（会在后面提示用户）
                        consumed_token_value = parsed['iteration']
                
                iterations.append(parsed['iteration'])
                consumed_tokens.append(consumed_token_value)
                mfu_values.append(mfu)
                
                if show_lm_loss and 'lm_loss' in parsed:
                    lm_loss_values.append(parsed['lm_loss'])
    
    print(f"文件 {file_path} 中的MFU来源: {mfu_source}")
    
    # 检查是否成功计算了consumed tokens
    has_consumed_tokens = any('consumed_tokens' in (parse_log_line(line) or {}) for line in open(file_path, 'r'))
    if not global_batch_size and not detected_global_batch_size and not has_consumed_tokens:
        print(f"警告: 文件 {file_path} 中未找到global_batch_size或consumed_tokens信息，X轴将显示为iteration数值")
    
    if show_lm_loss:
        return consumed_tokens, mfu_values, lm_loss_values
    else:
        return consumed_tokens, mfu_values


def calculate_ema(data, alpha=0.9):
    """计算指数移动平均 (EMA)
    alpha: 平滑系数，范围[0,1]，值越大表示对最近的数据权重越大
    """
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    return ema


def plot_mfu_curves(file_paths, title, show_lm_loss=False, show_mfu=True, lm_loss_ylim=None, show_loss_ema=False, global_batch_size=None):
    """绘制多个文件的MFU曲线和lm loss曲线"""
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # ax1用于显示lm loss（左侧），不需要第二个坐标轴
    ax2 = None
    
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
                consumed_tokens, mfu_values, lm_loss_values = parse_log_file(file_path, show_lm_loss=True, global_batch_size=global_batch_size)
            else:
                consumed_tokens, mfu_values = parse_log_file(file_path, show_lm_loss=False, global_batch_size=global_batch_size)
            
            if not consumed_tokens:
                print(f"警告: 文件 {file_path} 中未找到有效数据")
                continue
            
            # 获取文件名作为图例
            file_name = Path(file_path).name
            
            # 绘制lm loss曲线（在左侧y轴）
            if show_lm_loss and lm_loss_values and len(lm_loss_values) == len(consumed_tokens):
                # 使用相同颜色但透明度更低
                loss_color = to_rgba(base_color, alpha * (0.3 if show_loss_ema else 0.5))
                # 只在显示EMA时不显示原始loss的图例
                line2 = ax1.plot(consumed_tokens, lm_loss_values, color=loss_color, linewidth=1.5, linestyle='--', 
                               label=f'Loss: {file_name}' if not show_loss_ema else None)
                
                # 如果启用了loss EMA，绘制EMA曲线
                if show_loss_ema:
                    loss_ema_values = calculate_ema(lm_loss_values)
                    # 使用相同颜色但更深的线条绘制EMA
                    loss_ema_color = to_rgba(base_color, min(alpha * 0.5 + 0.3, 1.0))
                    ax1.plot(consumed_tokens, loss_ema_values, color=loss_ema_color, linewidth=1.0, 
                            label=f'Loss: {file_name}', linestyle='-')
                
                                # 在lm_loss的最低点标注一次数值
                if lm_loss_values:
                    min_loss_idx = np.argmin(lm_loss_values)
                    min_loss_value = lm_loss_values[min_loss_idx]
                    min_loss_tokens = consumed_tokens[min_loss_idx]
                    ax1.annotate(f'Min: {min_loss_value:.4f}', 
                                xy=(min_loss_tokens, min_loss_value),
                                xytext=(10, 10),  # 文本偏移量（向右上偏移）
                                textcoords='offset points',
                                color=loss_color,
                                fontsize=9,
                                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor=loss_color))
                                #arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color=loss_color))
        
        group_index += 1
    
    # 设置坐标轴
    ax1.set_xlabel('Consumed Tokens')
    if show_lm_loss:
        ax1.set_ylabel('LM Loss')
        if lm_loss_ylim:
            ax1.set_ylim(lm_loss_ylim)
        ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 格式化x轴标签
    import matplotlib.ticker as ticker
    def format_func(x, pos):
        return format_consumed_tokens(x)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
    
    # 设置标题
    plt.title(title)
    
    # 设置图例
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
    parser.add_argument('--global-batch-size', type=int, 
                        help='全局批次大小，用于计算consumed tokens。如果不提供，程序会尝试从日志中自动检测')
    args = parser.parse_args()
    
    plot_mfu_curves(args.files, args.title, 
                    show_lm_loss=args.show_lm_loss, 
                    show_mfu=not args.no_show_mfu,
                    lm_loss_ylim=args.lm_loss_ylim,
                    show_loss_ema=args.show_loss_ema,
                    global_batch_size=args.global_batch_size)


if __name__ == "__main__":
    main()
