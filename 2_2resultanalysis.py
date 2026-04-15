import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ==================== 设置中文字体 ====================
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'STHeiti']
matplotlib.rcParams['axes.unicode_minus'] = False

# ==================== 读取标注结果 ====================
df = pd.read_excel("data/附件2_标注结果.xlsx")
print("=" * 60)
print("诉求识别结果分析报告（含准确率评估）")
print("=" * 60)

print(f"\n【数据概览】")
print(f"总数据量：{len(df)} 条")

# ==================== 构建真实标签 ====================
# 原始数据中：有具体标签（城乡建设等）→ 有效诉求
# 噪声数据（标签为"噪声"）→ 无效诉求
df['true_label'] = df['一级标签'].apply(lambda x: True if x != '噪声' else False)
df['pred_label'] = df['is_valid']

# ==================== 1. 准确率评估 ====================
print("\n" + "=" * 60)
print("1. 模型准确率评估（以人工标注为基准）")
print("=" * 60)

# 计算各项指标
accuracy = accuracy_score(df['true_label'], df['pred_label'])
precision = precision_score(df['true_label'], df['pred_label'])
recall = recall_score(df['true_label'], df['pred_label'])
f1 = f1_score(df['true_label'], df['pred_label'])

print(f"准确率 (Accuracy)：{accuracy*100:.2f}%")
print(f"精确率 (Precision)：{precision*100:.2f}%")
print(f"召回率 (Recall)：{recall*100:.2f}%")
print(f"F1分数 (F1-Score)：{f1*100:.2f}%")

# 混淆矩阵
cm = confusion_matrix(df['true_label'], df['pred_label'])
tn, fp, fn, tp = cm.ravel()

print(f"\n混淆矩阵：")
print(f"  真阳性 (TP)：{tp} 条（模型正确识别为有效诉求）")
print(f"  真阴性 (TN)：{tn} 条（模型正确识别为噪声）")
print(f"  假阳性 (FP)：{fp} 条（模型误判为有效诉求，实际是噪声）")
print(f"  假阴性 (FN)：{fn} 条（模型漏判，实际是有效诉求）")

# ==================== 2. 按标签类别分析准确率 ====================
print("\n" + "=" * 60)
print("2. 按原始标签类别的准确率分析")
print("=" * 60)

# 排除噪声，只对有效诉求的各标签类别进行分析
df_valid_only = df[df['true_label'] == True]
label_accuracy = df_valid_only.groupby('一级标签').apply(
    lambda x: (x['true_label'] == x['pred_label']).mean()
).sort_values(ascending=False)

print("\n各标签类别识别准确率：")
for label, acc in label_accuracy.items():
    print(f"  {label}：{acc*100:.1f}%")

# ==================== 3. 有效诉求统计 ====================
print("\n" + "=" * 60)
print("3. 有效诉求与噪声分布")
print("=" * 60)

valid_counts = df['pred_label'].value_counts()
valid_pct = df['pred_label'].mean() * 100

print(f"模型识别有效诉求数量：{valid_counts.get(True, 0)} 条 ({valid_pct:.1f}%)")
print(f"模型识别噪声数量：{valid_counts.get(False, 0)} 条 ({100-valid_pct:.1f}%)")

# 真实分布
true_valid_counts = df['true_label'].value_counts()
true_valid_pct = df['true_label'].mean() * 100
print(f"\n实际有效诉求数量：{true_valid_counts.get(True, 0)} 条 ({true_valid_pct:.1f}%)")
print(f"实际噪声数量：{true_valid_counts.get(False, 0)} 条 ({100-true_valid_pct:.1f}%)")

# ==================== 4. 情绪分数分布 ====================
print("\n" + "=" * 60)
print("4. 情绪分数分布（模型识别结果）")
print("=" * 60)

emotion_dist = df['emotion_score'].value_counts().sort_index()
print(emotion_dist)
print(f"\n平均情绪分数：{df['emotion_score'].mean():.2f}")

emotion_labels = {1: "平静", 2: "轻微不满", 3: "明显不满", 4: "愤怒/急切", 5: "极度愤怒"}
print("\n情绪分数含义：")
for score, label in emotion_labels.items():
    count = emotion_dist.get(score, 0)
    pct = count / len(df) * 100
    print(f"  {score}分（{label}）：{count} 条 ({pct:.1f}%)")

# ==================== 5. 事件类型分布 ====================
print("\n" + "=" * 60)
print("5. 事件类型分布（仅有效诉求）")
print("=" * 60)

df_valid = df[df['pred_label'] == True]
event_dist = df_valid['event_type'].value_counts()
print(event_dist)

# ==================== 6. 升级信号统计 ====================
print("\n" + "=" * 60)
print("6. 升级信号统计")
print("=" * 60)

escalation_counts = df['escalation_signal'].value_counts()
print(f"包含升级信号：{escalation_counts.get(True, 0)} 条 ({escalation_counts.get(True, 0)/len(df)*100:.1f}%)")

valid_escalation = df_valid['escalation_signal'].sum()
print(f"有效诉求中包含升级信号：{valid_escalation} 条 ({valid_escalation/len(df_valid)*100:.1f}%)")

# ==================== 7. 地点抽取情况 ====================
print("\n" + "=" * 60)
print("7. 地点实体抽取情况")
print("=" * 60)

location_extracted = df_valid['location'].notna().sum()
location_rate = location_extracted / len(df_valid) * 100
print(f"有效诉求中成功抽取地点的数量：{location_extracted} 条 ({location_rate:.1f}%)")

# ==================== 8. 文本长度分析 ====================
print("\n" + "=" * 60)
print("8. 有效诉求 vs 噪声 文本长度对比")
print("=" * 60)

df['text_len'] = df['留言详情'].astype(str).str.len()
valid_len = df[df['pred_label'] == True]['text_len']
noise_len = df[df['pred_label'] == False]['text_len']

print(f"有效诉求平均长度：{valid_len.mean():.0f} 字符")
print(f"噪声平均长度：{noise_len.mean():.0f} 字符")

# ==================== 9. 输出样例 ====================
print("\n" + "=" * 60)
print("9. 模型输出样例")
print("=" * 60)

print("\n【正确识别的有效诉求】")
correct_valid = df[(df['true_label'] == True) & (df['pred_label'] == True)].head(3)
for _, row in correct_valid.iterrows():
    text = str(row['留言详情'])[:80]
    print(f"\n留言：{text}...")
    print(f"  地点：{row['location']} | 事件：{row['event_type']} | 情绪：{row['emotion_score']}分")

print("\n【错误识别的案例】")
wrong_cases = df[df['true_label'] != df['pred_label']].head(3)
for _, row in wrong_cases.iterrows():
    text = str(row['留言详情'])[:80]
    true_label = "有效诉求" if row['true_label'] else "噪声"
    pred_label = "有效诉求" if row['pred_label'] else "噪声"
    print(f"\n留言：{text}...")
    print(f"  真实标签：{true_label} | 模型预测：{pred_label}")

# ==================== 10. 保存统计结果 ====================
summary_data = {
    '指标': ['总数据量', '准确率', '精确率', '召回率', 'F1分数',
             '真阳性(TP)', '真阴性(TN)', '假阳性(FP)', '假阴性(FN)',
             '模型识别有效诉求数量', '实际有效诉求数量',
             '平均情绪分数', '地点抽取成功率', '有效诉求平均长度', '噪声平均长度'],
    '数值': [len(df), f"{accuracy*100:.2f}%", f"{precision*100:.2f}%",
             f"{recall*100:.2f}%", f"{f1*100:.2f}%",
             tp, tn, fp, fn,
             valid_counts.get(True, 0), true_valid_counts.get(True, 0),
             f"{df['emotion_score'].mean():.2f}", f"{location_rate:.1f}%",
             f"{valid_len.mean():.0f} 字符", f"{noise_len.mean():.0f} 字符"]
}
summary_df = pd.DataFrame(summary_data)
summary_df.to_excel("data/诉求识别统计结果.xlsx", index=False)
print("\n统计结果已保存至：data/诉求识别统计结果.xlsx")

# ==================== 11. 绘制图表 ====================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 图1：混淆矩阵热力图
ax1 = axes[0, 0]
cm_matrix = np.array([[tn, fp], [fn, tp]])
im = ax1.imshow(cm_matrix, cmap='Blues')
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.set_xticklabels(['预测噪声', '预测有效'])
ax1.set_yticklabels(['真实噪声', '真实有效'])
for i in range(2):
    for j in range(2):
        ax1.text(j, i, cm_matrix[i, j], ha='center', va='center', fontsize=16)
ax1.set_title('混淆矩阵', fontsize=14, fontweight='bold')

# 图2：有效诉求 vs 噪声 饼图
ax2 = axes[0, 1]
colors_valid = ['#2E86AB', '#A23B72']
ax2.pie([valid_counts.get(True, 0), valid_counts.get(False, 0)],
        labels=['有效诉求', '噪声'], autopct='%1.1f%%', colors=colors_valid, startangle=90)
ax2.set_title('模型识别结果分布', fontsize=14, fontweight='bold')

# 图3：情绪分数分布柱状图
ax3 = axes[0, 2]
emotion_dist.plot(kind='bar', color='#F18F01', ax=ax3)
ax3.set_xlabel('情绪分数', fontsize=12)
ax3.set_ylabel('数量', fontsize=12)
ax3.set_title('情绪分数分布', fontsize=14, fontweight='bold')
ax3.set_xticklabels(['1', '2', '3', '4', '5'], rotation=0)

# 图4：事件类型TOP8
ax4 = axes[1, 0]
top_events = event_dist.head(8)
ax4.barh(range(len(top_events)), top_events.values, color='#06A77D')
ax4.set_yticks(range(len(top_events)))
ax4.set_yticklabels(top_events.index)
ax4.set_xlabel('数量', fontsize=12)
ax4.set_title('事件类型TOP8', fontsize=14, fontweight='bold')
ax4.invert_yaxis()

# 图5：文本长度对比箱线图
ax5 = axes[1, 1]
box_data = [valid_len, noise_len]
bp = ax5.boxplot(box_data, labels=['有效诉求', '噪声'], patch_artist=True)
bp['boxes'][0].set_facecolor('#2E86AB')
bp['boxes'][1].set_facecolor('#A23B72')
ax5.set_ylabel('文本长度（字符）', fontsize=12)
ax5.set_title('文本长度对比', fontsize=14, fontweight='bold')

# 图6：各类别准确率柱状图
ax6 = axes[1, 2]
label_accuracy.head(8).plot(kind='bar', color='#9B5DE5', ax=ax6)
ax6.set_xlabel('标签类别', fontsize=12)
ax6.set_ylabel('准确率', fontsize=12)
ax6.set_title('各标签类别识别准确率', fontsize=14, fontweight='bold')
ax6.set_ylim(0, 1)
ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig("data/诉求识别分析图表.png", dpi=300, bbox_inches='tight')
print("图表已保存至：data/诉求识别分析图表.png")
plt.show()

print("\n" + "=" * 60)
print("分析完成！")
print("=" * 60)