import pandas as pd
import random
from datetime import datetime, timedelta

# ==================== 配置 ====================
# 文件路径
DATA_PATH = "data/附件2.xlsx"  # 你的城乡建设数据
OUTPUT_PATH = "data/附件2_混合数据.xlsx"  # 输出文件

# 噪声数据数量
NOISE_COUNT = 1000

# 起始编号（假设当前最大编号为 99999999，即8位数）
START_ID = 100000000  # 从9位数开始，避免与原有编号冲突

# ==================== 1. 读取原始城乡建设数据 ====================
df_original = pd.read_excel(DATA_PATH)
print(f"原始城乡建设数据量：{len(df_original)} 条")
print(f"原始数据字段：{df_original.columns.tolist()}")

# ==================== 2. 生成噪声数据 ====================

# 2.1 噪声文本库（模拟评论区常见的非诉求内容）
NOISE_TEXTS = {
    "纯情绪表达": [
        "太差了！",
        "无语了已经",
        "呵呵",
        "这是什么操作？",
        "服了",
        "真的假的？",
        "厉害了",
        "？？？",
        "顶",
        "支持一下",
        "路过看看",
        "已阅",
        "呵呵呵呵",
        "无语",
        "醉了",
        "笑死",
        "太离谱了",
        "emmm...",
        "哎",
        "唉，就这样吧"
    ],
    "广告/推广": [
        "专业疏通下水道，电话138xxxxxxx",
        "装修找我们，价格优惠",
        "XX楼盘热销中，联系电话xxxx",
        "需要搬家请联系我",
        "二手回收，上门服务",
        "专业保洁，随叫随到",
        "XX培训学校，火热招生中",
        "需要办理贷款请联系",
        "XX保险，为您保驾护航",
        "专业防水补漏，免费上门勘察"
    ],
    "无意义刷屏": [
        "111111",
        "aaaaaaaa",
        "......",
        "？？？？？？",
        "！！！！！！",
        "asdfghjkl",
        "666666",
        "哈哈哈哈",
        "路过",
        "打卡",
        "占楼",
        "前排",
        "沙发",
        "板凳",
        "地板"
    ],
    "纯转发/引用": [
        "转发微博",
        "//@用户123: 支持",
        "分享给更多人看看",
        "已转发",
        "扩散",
        "求扩散",
        "顶上去让大家看到",
        "Mark"
    ],
    "与政府无关的闲聊": [
        "今天天气真好",
        "晚饭吃什么呢",
        "有人一起打游戏吗",
        "明天放假吗",
        "这个视频真好看",
        "博主好帅",
        "小姐姐真漂亮",
        "求背景音乐",
        "求原图",
        "已关注，互关吗"
    ]
}

# 将所有噪声文本展平为一个列表
all_noise_texts = []
for category, texts in NOISE_TEXTS.items():
    all_noise_texts.extend(texts)

# 2.2 生成噪声数据
noise_data = []
start_date = datetime(2019, 1, 1)
end_date = datetime(2020, 12, 31)
date_range = (end_date - start_date).days

# 噪声用户ID池（模拟匿名用户或非实名用户）
noise_users = [f"N{random.randint(10000, 99999)}" for _ in range(200)]

for i in range(NOISE_COUNT):
    # 生成编号（从START_ID开始递增）
    record_id = START_ID + i

    # 随机选择用户
    user_id = random.choice(noise_users)

    # 随机生成时间（2019-2020年间）
    random_days = random.randint(0, date_range)
    comment_time = start_date + timedelta(days=random_days)
    # 随机添加时分秒
    comment_time = comment_time.replace(
        hour=random.randint(0, 23),
        minute=random.randint(0, 59),
        second=random.randint(0, 59)
    )

    # 随机选择噪声文本（可以组合多条，模拟真实评论区）
    if random.random() < 0.3:  # 30%的概率组合多条
        text_count = random.randint(2, 4)
        selected_texts = random.sample(all_noise_texts, min(text_count, len(all_noise_texts)))
        detail = " ".join(selected_texts)
    else:
        detail = random.choice(all_noise_texts)

    # 一级标签：噪声数据没有真实标签，标记为"噪声"
    label = "噪声"

    noise_data.append({
        "留言编号": record_id,
        "留言用户": user_id,
        "留言时间": comment_time,
        "留言详情": detail,
        "一级标签": label
    })

df_noise = pd.DataFrame(noise_data)
print(f"生成的噪声数据量：{len(df_noise)} 条")

# ==================== 3. 混合数据 ====================
df_mixed = pd.concat([df_original, df_noise], ignore_index=True)

# 打乱顺序（随机重排）
df_mixed = df_mixed.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n混合后总数据量：{len(df_mixed)} 条")
print(f"原始数据占比：{len(df_original) / len(df_mixed) * 100:.1f}%")
print(f"噪声数据占比：{len(df_noise) / len(df_mixed) * 100:.1f}%")

# ==================== 4. 保存 ====================
df_mixed.to_excel(OUTPUT_PATH, index=False)
print(f"\n混合数据已保存至：{OUTPUT_PATH}")

# ==================== 5. 验证输出 ====================
print("\n=== 标签分布 ===")
print(df_mixed['一级标签'].value_counts())

print("\n=== 噪声数据样例 ===")
for i in range(5):
    print(f"\n--- 噪声样例{i + 1} ---")
    print(f"编号：{df_mixed[df_mixed['一级标签'] == '噪声']['留言编号'].iloc[i]}")
    print(f"内容：{df_mixed[df_mixed['一级标签'] == '噪声']['留言详情'].iloc[i]}")

print("\n=== 原始数据样例（保留） ===")
original_samples = df_mixed[df_mixed['一级标签'] != '噪声'].head(2)
for _, row in original_samples.iterrows():
    print(f"\n--- 编号 {row['留言编号']} ---")
    print(f"标签：{row['一级标签']}")
    print(f"内容：{str(row['留言详情'])[:100]}...")