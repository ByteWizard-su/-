import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from difflib import SequenceMatcher
from collections import defaultdict
import jieba

# ==================== 设置中文字体 ====================
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'STHeiti']
matplotlib.rcParams['axes.unicode_minus'] = False

# ==================== 读取标注结果 ====================
df = pd.read_excel("data/附件2_标注结果.xlsx")
print("=" * 60)
print("诉求演化分析 - 事件聚合")
print("=" * 60)

# 只保留有效诉求
df_valid = df[df['is_valid'] == True].copy()
print(f"\n有效诉求数量：{len(df_valid)} 条")


# ==================== 1. 地点相似度计算函数 ====================
def location_similarity(loc1, loc2):
    """计算两个地点字符串的相似度"""
    if pd.isna(loc1) or pd.isna(loc2):
        return 0
    loc1 = str(loc1).strip()
    loc2 = str(loc2).strip()
    if loc1 == loc2:
        return 1.0
    # 包含关系
    if loc1 in loc2 or loc2 in loc1:
        return 0.85
    # 序列匹配
    return SequenceMatcher(None, loc1, loc2).ratio()


def merge_locations(locations, threshold=0.7):
    """将相似的地点合并为同一地点组"""
    locations = [str(loc).strip() for loc in locations if pd.notna(loc) and str(loc).strip()]
    if not locations:
        return {}

    # 去重
    unique_locs = list(set(locations))

    # 聚类
    groups = []
    used = set()

    for i, loc1 in enumerate(unique_locs):
        if i in used:
            continue
        group = [loc1]
        used.add(i)
        for j, loc2 in enumerate(unique_locs):
            if j in used:
                continue
            if location_similarity(loc1, loc2) >= threshold:
                group.append(loc2)
                used.add(j)
        groups.append(group)

    # 为每组选择一个代表名称（最短的作为代表）
    result = {}
    for group in groups:
        representative = min(group, key=len)
        for loc in group:
            result[loc] = representative
    return result


# ==================== 2. 关键词相似度计算 ====================
def keyword_similarity(kw1, kw2):
    """计算两个关键词的相似度"""
    if pd.isna(kw1) or pd.isna(kw2):
        return 0
    kw1 = str(kw1).strip()
    kw2 = str(kw2).strip()
    if kw1 == kw2:
        return 1.0
    # 包含关系（如"供水安全"包含"供水"）
    if kw1 in kw2 or kw2 in kw1:
        return 0.8
    # 同义词映射
    synonym_map = {
        '施工安全': ['施工隐患', '施工扰民', '工地安全'],
        '物业纠纷': ['物业收费', '物业不作为', '物业管理'],
        '供水安全': ['二次供水', '水压', '停水'],
        '环境卫生': ['垃圾', '卫生', '清洁'],
        '设施损坏': ['路灯', '井盖', '道路损坏']
    }
    for standard, synonyms in synonym_map.items():
        if kw1 == standard or kw1 in synonyms:
            if kw2 == standard or kw2 in synonyms:
                return 0.9
    return SequenceMatcher(None, kw1, kw2).ratio()


def merge_keywords(keywords, threshold=0.7):
    """将相似的关键词合并"""
    keywords = [str(kw).strip() for kw in keywords if pd.notna(kw) and str(kw).strip()]
    if not keywords:
        return {}

    unique_kws = list(set(keywords))
    groups = []
    used = set()

    for i, kw1 in enumerate(unique_kws):
        if i in used:
            continue
        group = [kw1]
        used.add(i)
        for j, kw2 in enumerate(unique_kws):
            if j in used:
                continue
            if keyword_similarity(kw1, kw2) >= threshold:
                group.append(kw2)
                used.add(j)
        groups.append(group)

    result = {}
    for group in groups:
        representative = group[0]
        for kw in group:
            result[kw] = representative
    return result


# ==================== 3. 基于地点的聚合 ====================
print("\n" + "=" * 60)
print("3.1 基于地点的聚合")
print("=" * 60)

# 获取所有非空地点
valid_locations = df_valid['location'].dropna().tolist()
print(f"有地点信息的诉求数量：{len(valid_locations)} 条")

# 合并相似地点
location_merge_map = merge_locations(valid_locations, threshold=0.7)
df_valid['location_group'] = df_valid['location'].apply(
    lambda x: location_merge_map.get(str(x).strip(), x) if pd.notna(x) else None
)

# 统计每个地点组的留言数量
location_counts = df_valid[df_valid['location_group'].notna()]['location_group'].value_counts()
print(f"不同地点组数量：{len(location_counts)}")
print(f"\n地点组留言数量TOP10：")
for loc, count in location_counts.head(10).items():
    print(f"  {loc}: {count} 条")

# ==================== 4. 基于关键词的聚合 ====================
print("\n" + "=" * 60)
print("3.2 基于关键词的聚合")
print("=" * 60)

# 获取所有非空关键词
valid_keywords = df_valid['event_type'].dropna().tolist()
print(f"有关键词信息的诉求数量：{len(valid_keywords)} 条")

# 合并相似关键词
keyword_merge_map = merge_keywords(valid_keywords, threshold=0.7)
df_valid['keyword_group'] = df_valid['event_type'].apply(
    lambda x: keyword_merge_map.get(str(x).strip(), x) if pd.notna(x) else None
)

# 统计每个关键词组的留言数量
keyword_counts = df_valid[df_valid['keyword_group'].notna()]['keyword_group'].value_counts()
print(f"不同关键词组数量：{len(keyword_counts)}")
print(f"\n关键词组留言数量TOP10：")
for kw, count in keyword_counts.head(10).items():
    print(f"  {kw}: {count} 条")

# ==================== 5. 取交集：地点+关键词 事件聚合 ====================
print("\n" + "=" * 60)
print("3.3 地点+关键词 事件聚合（取交集）")
print("=" * 60)

# 创建事件ID：地点组 + 关键词组
df_valid['event_id'] = df_valid.apply(
    lambda row: f"{row['location_group']}|{row['keyword_group']}"
    if pd.notna(row['location_group']) and pd.notna(row['keyword_group'])
    else None, axis=1
)

# 统计每个事件的留言数量
event_counts = df_valid[df_valid['event_id'].notna()]['event_id'].value_counts()
print(f"不同事件数量：{len(event_counts)}")
print(f"\n事件留言数量TOP10：")
for event_id, count in event_counts.head(10).items():
    loc, kw = event_id.split('|')
    print(f"  [{kw}] {loc}: {count} 条")

# ==================== 6. 选取TOP3典型事件 ====================
print("\n" + "=" * 60)
print("3.4 典型事件选取（TOP3）")
print("=" * 60)

top3_events = event_counts.head(3)
selected_events = []

for i, (event_id, count) in enumerate(top3_events.items(), 1):
    loc, kw = event_id.split('|')

    # 获取该事件的所有留言
    event_df = df_valid[df_valid['event_id'] == event_id].copy()
    event_df['留言时间'] = pd.to_datetime(event_df['留言时间'])
    event_df = event_df.sort_values('留言时间')

    # 计算时间跨度
    time_span = (event_df['留言时间'].max() - event_df['留言时间'].min()).days
    user_count = event_df['留言用户'].nunique()

    print(f"\n典型事件{i}：{kw} - {loc}")
    print(f"  留言数量：{count} 条")
    print(f"  涉及用户：{user_count} 人")
    print(f"  时间跨度：{time_span} 天")
    print(f"  时间范围：{event_df['留言时间'].min().date()} 至 {event_df['留言时间'].max().date()}")

    # 显示代表性留言
    sample_text = event_df.iloc[0]['留言详情'][:100]
    print(f"  代表性留言：{sample_text}...")

    # 分析演化阶段
    daily_counts = event_df.groupby(event_df['留言时间'].dt.date).size()
    print(f"  每日分布：{dict(daily_counts)}")

    selected_events.append({
        'rank': i,
        'location': loc,
        'keyword': kw,
        'count': count,
        'user_count': user_count,
        'time_span': time_span,
        'start_date': event_df['留言时间'].min(),
        'end_date': event_df['留言时间'].max(),
        'event_df': event_df
    })

# ==================== 7. 绘制散点图 ====================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 图1：地点聚合散点图
ax1 = axes[0]
location_sorted = location_counts.head(30)
x1 = range(len(location_sorted))
y1 = location_sorted.values
ax1.scatter(x1, y1, s=50, c='#2E86AB', alpha=0.7)
ax1.set_xlabel('地点组 (按数量排序)', fontsize=12)
ax1.set_ylabel('留言数量', fontsize=12)
ax1.set_title('基于地点的诉求聚合分布', fontsize=14, fontweight='bold')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)

# 图2：关键词聚合散点图
ax2 = axes[1]
keyword_sorted = keyword_counts.head(30)
x2 = range(len(keyword_sorted))
y2 = keyword_sorted.values
ax2.scatter(x2, y2, s=50, c='#06A77D', alpha=0.7)
ax2.set_xlabel('关键词组 (按数量排序)', fontsize=12)
ax2.set_ylabel('留言数量', fontsize=12)
ax2.set_title('基于关键词的诉求聚合分布', fontsize=14, fontweight='bold')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)

# 图3：事件聚合散点图（地点+关键词）
ax3 = axes[2]
event_sorted = event_counts.head(30)
x3 = range(len(event_sorted))
y3 = event_sorted.values
ax3.scatter(x3, y3, s=50, c='#F18F01', alpha=0.7)
ax3.set_xlabel('事件组 (按数量排序)', fontsize=12)
ax3.set_ylabel('留言数量', fontsize=12)
ax3.set_title('地点+关键词 事件聚合分布', fontsize=14, fontweight='bold')
ax3.set_yscale('log')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("data/事件聚合散点图.png", dpi=300, bbox_inches='tight')
print("\n散点图已保存至：data/事件聚合散点图.png")
plt.show()

# ==================== 8. 保存事件聚合结果 ====================
# 保存事件聚合表
event_summary = []
for event_id, count in event_counts.head(50).items():
    loc, kw = event_id.split('|')
    event_df = df_valid[df_valid['event_id'] == event_id]
    time_span = (pd.to_datetime(event_df['留言时间']).max() - pd.to_datetime(event_df['留言时间']).min()).days
    user_count = event_df['留言用户'].nunique()
    event_summary.append({
        '地点': loc,
        '关键词': kw,
        '留言数量': count,
        '涉及用户数': user_count,
        '时间跨度(天)': time_span,
        '最早时间': event_df['留言时间'].min(),
        '最晚时间': event_df['留言时间'].max()
    })

event_summary_df = pd.DataFrame(event_summary)
event_summary_df.to_excel("data/事件聚合结果.xlsx", index=False)
print("\n事件聚合结果已保存至：data/事件聚合结果.xlsx")

# 输出统计信息
print("\n" + "=" * 60)
print("3.5 聚合统计汇总")
print("=" * 60)
print(f"有地点的诉求：{len(valid_locations)} 条")
print(f"合并后地点组数量：{len(location_counts)}")
print(f"合并前地点唯一值：{len(set(valid_locations))}")
print(f"\n有关键词的诉求：{len(valid_keywords)} 条")
print(f"合并后关键词组数量：{len(keyword_counts)}")
print(f"合并前关键词唯一值：{len(set(valid_keywords))}")
print(f"\n成功聚合的事件数量：{len(event_counts)}")
print(f"总事件留言数：{event_counts.sum()} 条")

print("\n" + "=" * 60)
print("分析完成！TOP3典型事件已选出")
print("=" * 60)