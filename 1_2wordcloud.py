import pandas as pd
import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib

# ==================== 设置中文字体 ====================
# 解决中文显示问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'STHeiti']
matplotlib.rcParams['axes.unicode_minus'] = False

# ==================== 停用词表 ====================
stopwords = set([
    # 常见无意义词
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '也', '被',
    '这', '那', '到', '说', '去', '与', '而', '及', '或', '对', '从', '为', '以', '于', '由',
    '但', '却', '只', '可', '并', '还', '更', '很', '太', '最', '多', '少', '几', '些', '各',
    '某', '每', '等', '将', '使', '让', '给', '把', '被', '得', '地', '着', '过', '吧', '吗',
    '呢', '啊', '哦', '嗯', '哈', '啦', '呀', '哇', '呵', '嘿', '哎', '喂', '哦哦', '嗯嗯',
    # 时间相关
    '今天', '明天', '昨天', '现在', '过去', '以前', '以后', '之前', '之后', '当时', '目前',
    '近日', '近期', '日前', '至今', '已经', '正在', '即将', '快要', '刚刚', '刚才',
    # 程度/语气
    '非常', '十分', '特别', '相当', '比较', '有点', '一些', '一定', '必须', '应该', '可以',
    '需要', '希望', '请求', '要求', '强烈', '实在', '真的', '真是', '真是', '确实', '确实',
    # 连接词
    '因为', '所以', '但是', '然而', '虽然', '尽管', '而且', '并且', '或者', '还是', '于是',
    '然后', '接着', '另外', '此外', '总之', '也就是说', '这样一来',
    # 其他常见
    '什么', '怎么', '为什么', '如何', '哪', '哪个', '哪些', '哪里', '这儿', '那儿', '这里',
    '那里', '这个', '那个', '这些', '那些', '一样', '这样', '那样', '这么', '那么', '这么样',
    '那么样', '如此', '而已', '罢了', '啦', '咯', '喽', '啰', '吖', '嘛', '呗', '勒', '嚒',
    # 代词
    '你', '您', '他', '她', '它', '我们', '你们', '他们', '她们', '它们', '自己', '别人',
    '大家', '人家', '谁', '各位', '诸位', '本人', '我方', '你方', '对方',
    # 无意义词
    '东西', '事情', '情况', '时候', '时间', '地方', '问题', '原因', '结果', '方式', '方法',
    '方面', '程度', '范围', '过程', '步骤', '环节', '阶段', '部分', '整体', '所有', '一切',
    '全部', '唯一', '主要', '重要', '关键', '基本', '具体', '详细', '充分', '足够',
    # 噪声特有
    '转发微博', '已阅', '路过', '沙发', '板凳', '顶', '支持一下', '打卡', '占楼', '前排',
    'mark', '收藏', '赞', '哈哈哈', '呵呵', '哦哦', '嗯嗯', '好的', '收到', '明白', '666', '111111',
    'aaaaaa', '......', '？？？', '！！！', 'asdfghjkl', '哈哈哈哈', '路过', '打卡'
])

# ==================== 读取数据 ====================
df = pd.read_excel("data/附件2_混合数据.xlsx")
print(f"数据量：{len(df)} 条")

# ==================== 合并所有文本 ====================
text = " ".join(df['留言详情'].astype(str))

# ==================== 分词并过滤停用词 ====================
print("正在分词...")
words = jieba.cut(text)

# 过滤：长度>1、不是纯数字/字母、不在停用词表中
filtered_words = []
for word in words:
    word = word.strip()
    # 过滤条件
    if len(word) <= 1:
        continue
    if word.isdigit():
        continue
    if word.isalpha() and word.isascii():
        continue
    if word in stopwords:
        continue
    filtered_words.append(word)

print(f"分词后词数：{len(filtered_words)}")
text_cut = " ".join(filtered_words)

# ==================== 生成词云 ====================
print("正在生成词云...")
wc = WordCloud(
    font_path="C:/Windows/Fonts/simhei.ttf",  # 使用黑体
    width=1600,
    height=1000,
    background_color='white',
    max_words=150,
    collocations=False  # 不显示搭配词
)
wc.generate(text_cut)

# ==================== 绘图并保存 ====================
plt.figure(figsize=(16, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title("混合数据词云图（去停用词后）", fontsize=24, fontweight='bold')
plt.tight_layout(pad=0)
plt.savefig("data/词云图.png", dpi=300, bbox_inches='tight')
print("词云图已保存至：data/词云图.png")

# 显示
plt.show()

# ==================== 输出高频词TOP20 ====================
from collections import Counter
word_counts = Counter(filtered_words)
print("\n=== 高频词 TOP20 ===")
for word, count in word_counts.most_common(20):
    print(f"  {word}: {count}")