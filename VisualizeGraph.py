'''
Author: letianyu letianyu@tencent.com
Date: 2024-11-29 10:41:23
LastEditors: letianyu letianyu@tencent.com
LastEditTime: 2024-11-29 18:18:13
FilePath: \weibo-crawler\VisualizeGraph.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import networkx as nx
import csv
import plotly.graph_objects as go

graph_data_path = "./weibo/searchedGraph.txt"
crawled_users_path = "./weibo/users.csv"
page_data_path = "./weibo/searchedPages.csv"

G = nx.DiGraph()

def split_string_with_newlines(input_string, max_length = 100):
    """
    将输入的字符串按照指定的最大长度分段并添加换行符。

    :param input_string: 要处理的输入字符串
    :param max_length: 每段的最大长度
    :return: 处理好的带有换行符的字符串
    """
    result = ""
    for i in range(0, len(input_string), max_length):
        result += "<br>" + input_string[i:i + max_length]
    return result

# 打开文件并读取数据
with open(graph_data_path, 'r') as f:
    graph_lines = f.readlines()

with open(crawled_users_path, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    user_list_data = list(reader)

with open(page_data_path, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    page_list_data = list(reader)

# Read UserInfo
user_info_dict = {}
for row in user_list_data:
    user_info_dict[row["\ufeff用户id"]] = row
    temp_lines = []
    temp_lines.append("用户id:"+row["\ufeff用户id"])
    temp_lines.append("昵称:"+row["昵称"])
    #for key, value in row.items():
    #    temp_lines.append(f"{key}: {value}")
    user_info_dict[row["\ufeff用户id"]]["formated_str"] = "<br>".join(temp_lines)
    user_info_dict[row["\ufeff用户id"]]["raw_weibo"] = []

# Read Page Datas
for row in page_list_data:
    user_id = row["uid"]
    if user_id in user_info_dict:
        user_info_dict[user_id]["raw_weibo"].append(row)

# cook page datas
for row in user_list_data:
    temp_lines = []
    if len(user_info_dict[row["\ufeff用户id"]]["raw_weibo"]) > 0:
        for raw_weibo_line in user_info_dict[row["\ufeff用户id"]]["raw_weibo"]:
            temp_lines.append("原:" + split_string_with_newlines(raw_weibo_line["正文"]) + "转:" + split_string_with_newlines(raw_weibo_line["源微博正文"]))
        user_info_dict[row["\ufeff用户id"]]["formated_str"] = user_info_dict[row["\ufeff用户id"]]["formated_str"] + "<br>".join(temp_lines)

# 遍历每一行数据
for line in graph_lines:
    if line.startswith('node:'):
        node_id = line.split(':')[1].strip()
        G.add_node(node_id)
    elif line.startswith('follower:'):
        line_data_part = line.split(':')[1]
        parts = line_data_part.split(' ')
        source_node_id = parts[1].strip()
        target_node_id = parts[0].strip()
        G.add_edge(source_node_id, target_node_id, label = "follower")
    elif line.startswith('redirect'):
        line_data_part = line.split(':')[1]
        parts = line_data_part.split(' ')
        source_node_id = parts[0].strip()
        target_node_id = parts[1].strip()
        G.add_edge(source_node_id, target_node_id, label = "redirect")

# 使用Spring布局算法计算节点位置（也可尝试其他布局算法）
pos = nx.spring_layout(G)

# 创建边的坐标列表
edge_follower_x = []
edge_follower_y = []
edge_redirect_x = []
edge_redirect_y = []

for edge in G.edges(data=True):
    source_node = edge[0]
    target_node = edge[1]
    edge_attrs = edge[2]  # 这里包含了边的属性字典
    
    if (source_node in user_info_dict) and (target_node in user_info_dict):
        x0, y0 = pos[source_node]
        x1, y1 = pos[target_node]

        if edge_attrs.get("label") == "follower":
            edge_follower_x.extend([x0, x1, None])
            edge_follower_y.extend([y0, y1, None])
        elif edge_attrs.get("label") == "redirect":
            edge_redirect_x.extend([x0, x1, None])
            edge_redirect_y.extend([y0, y1, None])

# 创建节点的坐标列表和标签列表
node_x = []
node_y = []
node_text = []
node_hover = []
node_weibo_x = []
node_weibo_y = []
node_weibo_text = []
node_weibo_hover = []
for node in G.nodes():
    x, y = pos[node]
    node_has_weibo = False
    if str(node) in user_info_dict:
        if len(user_info_dict[str(node)]["raw_weibo"]) > 0:
            node_has_weibo = True
    if node_has_weibo:
        if str(node) in user_info_dict:
            node_weibo_x.append(x)
            node_weibo_y.append(y)
            node_weibo_text.append(user_info_dict[str(node)]["昵称"])
            node_weibo_hover.append(user_info_dict[str(node)]["formated_str"])
    else:
        if str(node) in user_info_dict:
            node_x.append(x)
            node_y.append(y)
            node_text.append(user_info_dict[str(node)]["昵称"])
            node_hover.append(user_info_dict[str(node)]["formated_str"])

# 创建边的轨迹
edge_trace_follower = go.Scatter(
    x=edge_follower_x,
    y=edge_follower_y,
    line=dict(width=1, color='lightblue'),
    hoverinfo='none',
    mode='lines',  # 添加'markers' 模式以便显示箭头标记
)
edge_trace_redirect = go.Scatter(
    x=edge_redirect_x,
    y=edge_redirect_y,
    line=dict(width=1, color='royalblue'),
    hoverinfo='none',
    mode='lines',  # 添加'markers' 模式以便显示箭头标记
)

# 创建节点的轨迹
node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    text=node_text,
    hovertext=node_hover,
    mode='markers+text',
    marker=dict(
        size=20,
        color='orange'
    ),
    hoverinfo='text',
    selected=dict(
        marker=dict(
            size=24,
            color='red'
        )
    )
)

weibo_node_trace = go.Scatter(
    x=node_weibo_x,
    y=node_weibo_y,
    text=node_weibo_text,
    hovertext=node_weibo_hover,
    mode='markers+text',
    marker=dict(
        size=20,
        color='white'
    ),
    hoverinfo='text',
    selected=dict(
        marker=dict(
            size=24,
            color='red'
        )
    )
)

# 创建图形对象并添加轨迹
fig = go.Figure(data=[edge_trace_redirect, edge_trace_follower, node_trace, weibo_node_trace])

# 设置图形布局
fig.update_layout(
    showlegend=False,
    hovermode='closest',
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
)

# 显示图形
fig.show()