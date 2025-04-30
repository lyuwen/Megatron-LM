
import plotly.graph_objects as go
from typing import Dict, List
import os
from itertools import groupby
import argparse


def main() -> None:

    parser = argparse.ArgumentParser(description='生成调度策略可视化图')

    parser.add_argument('--input', '-i',
                        type=str, 
                        default='timecond',
                        help='Schedule timecond input path.')
    parser.add_argument('--output', '-o', 
                        type=str, 
                        default='schedule_visualization.jpg',
                        help='Output path.')
    parser.add_argument('--iteration',
                        type=int, 
                        default=10,
                        help='If set, visualize the scheduling timecond for the nth iteration of the process.')                    
    
    args = parser.parse_args()

    run_plot_timecond(args.input, args.iteration)


def parse_timecond_file(file_path: str, iteration: int) -> Dict[int, List[Dict]]:
    """Parse the timecond file into a visualization format."""
    visualization_data = {}
    
    # First pass: collect all operations
    iteration -=1
    operations = []
    read_time = False
    with open(file_path, 'r') as f:
        for line in f:
            if 'iteration ' + str(iteration) in line:
                read_time = True
            elif 'iteration' in line and 'iteration ' + str(iteration) not in line:
                read_time = False
            if read_time:
                parts = line.strip().split()
                if len(parts) != 6:
                    continue
            else:
                continue
                
            stage = int(parts[1])
            batch_type = parts[2]  # F or B
            batch_num = int(parts[3])
            event_type = parts[4]  # begin or end
            time = float(parts[5]) 
            
            if time < -500000:
                time = time + 613725
            else:
                time = time + 500000

            
            operations.append({
                'stage': stage,
                'batch_type': batch_type,
                'batch_num': batch_num,
                'event_type': event_type,
                'time': time
            })
    
    # Second pass: organize by stage
    for stage in set(op['stage'] for op in operations):
        visualization_data[stage] = []
        
        # Get all operations for this stage
        stage_ops = [op for op in operations if op['stage'] == stage]
        
        # Group by batch type and number
        for _, group in groupby(stage_ops, key=lambda x: (x['batch_type'], x['batch_num'])):
            group = list(group)
            if len(group) != 2:  # Need both begin and end
                continue
                
            begin = next(op for op in group if op['event_type'] == 'begin')
            end = next(op for op in group if op['event_type'] == 'end')
            
            visualization_data[stage].append({
                'type': begin['batch_type'],
                'batch': begin['batch_num'],
                'start_time': begin['time'],
                'duration': end['time'] - begin['time']
            })
    
    return visualization_data

def get_color(op_type: str, stage_id: int, num_devices: int):
    """Get color for an operation type."""
    # A more harmonious blue palette with low saturation and high brightness
    forward_colors = [
        "#0a5aff",  # Intense blue
        "#4c88ff",  # Blue (deeper)
        "#7aa7ff",  # Medium blue
        "#a8c5ff",  # Soft blue
        "#d6e4ff",  # Very light blue
    ]

    # Orange palette for backward operations with low saturation and high brightness
    backward_colors = [
        "#f47b00",  # Intense orange
        "#ffa952",  # Orange
        "#ffc78e",  # Light orange
        "#ffe6cc",  # Very light orange
    ]
    backward_attn_D_colors = [
        "#2dff54",  # Intense green
        "#00c800",  # Green
        "#a8ff9e",  # Light green
        "#e5ffe0",  # Very light green
    ]

    backward_attn_D_colors = [
        "#2dff54",  # Intense green
        "#00c800",  # Green
        "#a8ff9e",  # Light green
        "#e5ffe0",  # Very light green
    ]
    
    backward_W_colors = [
        "#9d00ff",  # Intense  purple
        "#a952ff",  #  Purple
        "#d9a0ff",  # Light purple
        "#f2e0ff",  # Very light purple
    ]
    
    # 通信颜色 - 前向使用浅黄色，后向使用浅粉色
    forward_communication_color = "#FFEF99"  # 浅黄色
    backward_communication_color = "#FFD6E7"  # 浅粉色
    
    virtual_stage = stage_id // num_devices
    color_index = virtual_stage % len(forward_colors)

    if op_type == "F":
        return forward_colors[color_index]
    elif op_type == "ATTNF":
        return forward_colors[color_index]
    elif op_type == "MLPF":
        return forward_colors[color_index]
    elif op_type == "B":
        return backward_colors[color_index % len(backward_colors)]
    elif op_type == "ATTND":
        return backward_D_colors[color_index % len(backward_colors)]
    elif op_type == "ATTNW":
        return backward_W_colors[color_index % len(backward_colors)]
    elif op_type == "MLPD":
        return backward_D_colors[color_index % len(backward_colors)]
    elif op_type == "MLPW":
        return backward_W_colors[color_index % len(backward_colors)]
    elif op_type == "communication":
        return forward_communication_color
    else:
        raise ValueError(f"Invalid operation type: {op_type}")

def create_timeline_figure(visualization_data: Dict[int, List[Dict]], max_time=None):
    """Create a Plotly figure for the timeline visualization."""
    # Find the maximum time if not provided
    if max_time is None:
        max_time = 0
        for stage in visualization_data:
            for op in visualization_data[stage]:
                end_time = op['start_time'] + op['duration']
                if end_time > max_time:
                    max_time = end_time
    
    # Create figure
    fig = go.Figure()
    
    # Create a custom y-axis with no gaps between stages
    y_spacing = 1.0  # Use 1.0 for no gaps
    
    # Batch processing for increased performance
    shapes = []
    annotations = []
    hover_traces = []
    
    # Add rectangles for each operation
    for stage in visualization_data:
        stage_idx_reversed = max(visualization_data.keys()) - stage
        
        # 首先分析这个阶段上的所有任务，找出计算和通信任务
        computation_tasks = []
        communication_tasks = []
        
        for task in visualization_data[stage]:
            if task.get("type") == "communication":
                communication_tasks.append(task)
            else:
                computation_tasks.append(task)
                
        # 对所有计算和通信任务排序
        computation_tasks.sort(key=lambda t: t["start_time"])
        communication_tasks.sort(key=lambda t: t["start_time"])
        
        # 绘制所有计算任务
        for task in computation_tasks:
            y_pos = stage_idx_reversed * y_spacing
            start_time = task["start_time"]
            duration = task["duration"]
            
            # Regular operation
            op_type = task["type"]
            color = get_color(op_type, stage, len(visualization_data))
            text_color = "white" if op_type == "F" else "black"
            
            shapes.append(
                dict(
                    type="rect",
                    xref="x",
                    yref="y",
                    x0=start_time,
                    y0=y_pos - 0.5,
                    x1=start_time + duration,
                    y1=y_pos + 0.5,
                    line=dict(color="black", width=0.5),
                    fillcolor=color,
                    layer="above",
                )
            )
            
            # Add batch number label
            annotations.append(
                dict(
                    x=start_time + duration / 2,
                    y=y_pos,
                    text=f"{task['batch']}",
                    showarrow=False,
                    font=dict(color=text_color, size=12, family="Arial, bold"),
                )
            )
            
            hover_text = (
                f"Batch: {task['batch']}<br>"
                f"Stage: {stage}<br>"
                f"Type: {op_type}<br>"
                f"Start: {start_time:.2f}<br>"
                f"End: {start_time + duration:.2f}<br>"
                f"Duration: {duration:.2f}"
            )
            
            hover_traces.append(
                dict(
                    x=[start_time + duration / 2],
                    y=[y_pos],
                    mode="markers",
                    marker=dict(opacity=0),  # Invisible marker
                    hoverinfo="text",
                    text=hover_text,
                    showlegend=False,
                )
            )
    
    # Add all shapes at once for better performance
    fig.update_layout(shapes=shapes)
    
    # Add all annotations at once
    fig.update_layout(annotations=annotations)
    
    # Add all hover traces at once
    for trace in hover_traces:
        fig.add_trace(go.Scatter(**trace))
    
    # Update layout
    fig.update_layout(
        title='Timeline Visualization',
        xaxis_title='Time',
        yaxis_title='Stage',
        showlegend=True,
        height=400,  # 调整高度
        width=2000,  # 调整宽度
        hovermode='closest',
        yaxis=dict(
            tickmode='array',
            tickvals=[(max(visualization_data.keys()) - i) * y_spacing for i in range(max(visualization_data.keys()) + 1)],
            ticktext=[f'Stage {i}' for i in range(max(visualization_data.keys()) + 1)],
            showgrid=False,
            zeroline=False,
        ),
        plot_bgcolor='white',
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.02,
            xanchor="right",
            x=1.20,
            title=dict(text="<b>Operation Types:</b>"),
            itemsizing="constant",
            tracegroupgap=0,
        )
    )
    
    return fig

def run_plot_timecond(input_file, iteration) -> None:
    """Run visualization directly from timecond file."""
    if not os.path.exists(input_file):
        raise FileNotFoundError("shecdule timecond file not found in current directory")
    
    # Parse the timecond file
    visualization_data = parse_timecond_file(input_file, iteration)
    import plotly.io as pio
    fig = create_timeline_figure(visualization_data)

    try:
        pio.write_image(fig, 'timeline_visualization.jpg', format='jpeg', engine='kaleido')
        print("图表已保存为 timelene_visualization.jpg")
    except Exception as e:
        print(f"保存图片失败: {str(e)}")

if __name__ == "__main__":
    main()
