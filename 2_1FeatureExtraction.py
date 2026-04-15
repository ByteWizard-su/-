import pandas as pd
import requests
import json
import re
import time
from tqdm import tqdm
import os

# ==================== 配置 ====================
API_KEY = "sk-pgsezrgsbyhwrznnsebhwxkcjclmqnwupsnkhzsfhtssvedh"

API_URL = "https://api.siliconflow.cn/v1/chat/completions"

# 换用更大的14B模型
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"

INPUT_PATH = "data/附件2_混合数据.xlsx"
OUTPUT_PATH = "data/附件2_标注结果.xlsx"

BATCH_SIZE = 100


# ==================== 规则预过滤 ====================
def is_obvious_noise(text):
    """规则过滤明显的噪声"""
    text = text.strip()

    # 纯数字/字母/符号（短文本）
    if re.match(r'^[\d\w\s\.\,\?\!\(\)\-\+\*\&\^\%\$\#\@\~]+$', text) and len(text) < 30:
        return True

    # 纯表情或极短无意义
    if len(text) <= 3 and not re.search(r'[\u4e00-\u9fff]', text):
        return True

    # 常见噪声关键词
    noise_keywords = [
        '转发微博', '已阅', '路过', '沙发', '板凳', '顶', '支持一下',
        '打卡', '占楼', '前排', 'mark', '收藏', '赞', '666', '哈哈哈',
        '呵呵', '哦哦', '嗯嗯', '好的', '收到', '明白'
    ]
    for kw in noise_keywords:
        if kw == text or text.startswith(kw) and len(text) < 20:
            return True

    # 广告特征
    ad_keywords = ['电话', '联系', '微信', 'qq', '专业', '服务', '上门']
    ad_count = sum(1 for kw in ad_keywords if kw in text)
    if ad_count >= 2 and len(text) < 50:
        return True

    return False


# ==================== Few-shot 提示词 ====================
def build_prompt(text):
    """构建带Few-shot示例的提示词"""
    if len(text) > 1500:
        text = text[:1500] + "..."

    prompt = '''你是政务留言分类助手。参考以下例子，判断新留言。

【有效诉求定义】描述了具体的、需要政府解决的实际问题：道路损坏、井盖缺失、垃圾堆积、停水停电、物业纠纷、施工扰民、安全隐患、供水问题、收费问题等。

【无效内容】纯情绪、广告、无意义刷屏、纯转发、闲聊。

--- 例子1（有效诉求）---
留言：华庭小区二次供水水箱长年不洗，水有严重霉味，请环保局来检测。
输出：{"is_valid": true, "location": "华庭小区", "time_entity": "长年", "event_type": "供水安全", "emotion_score": 4, "escalation_signal": true}

--- 例子2（有效诉求）---
留言：A3区大道西行便道，人行道被施工围墙圈占，每天人流车流极多，安全隐患非常大。
输出：{"is_valid": true, "location": "A3区大道西行便道", "time_entity": null, "event_type": "施工安全", "emotion_score": 4, "escalation_signal": false}

--- 例子3（有效诉求）---
留言：小区物业未经业主同意强收停车费，300元一个月，反映多次没人管。
输出：{"is_valid": true, "location": null, "time_entity": null, "event_type": "物业纠纷", "emotion_score": 4, "escalation_signal": false}

--- 例子4（无效：纯情绪）---
留言：太差了！无语
输出：{"is_valid": false, "location": null, "time_entity": null, "event_type": null, "emotion_score": 2, "escalation_signal": false}

--- 例子5（无效：广告）---
留言：专业疏通下水道，电话138xxxxxxx
输出：{"is_valid": false, "location": null, "time_entity": null, "event_type": null, "emotion_score": 1, "escalation_signal": false}

--- 例子6（无效：无意义刷屏）---
留言：111111
输出：{"is_valid": false, "location": null, "time_entity": null, "event_type": null, "emotion_score": 1, "escalation_signal": false}

--- 例子7（无效：纯转发）---
留言：转发微博
输出：{"is_valid": false, "location": null, "time_entity": null, "event_type": null, "emotion_score": 1, "escalation_signal": false}

现在判断这条留言，只输出JSON，不要有其他文字：

留言：''' + text + '''

输出：'''

    return prompt


# ==================== 解析响应 ====================
def parse_response(content):
    """解析模型返回的JSON"""
    result = {
        "is_valid": False,
        "location": None,
        "time_entity": None,
        "event_type": None,
        "emotion_score": 3,
        "escalation_signal": False
    }

    # 提取JSON
    json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
    if json_match:
        json_str = json_match.group()
    else:
        json_str = content

    json_str = json_str.replace('\n', ' ')

    # 尝试解析
    try:
        parsed = json.loads(json_str)
        if 'is_valid' in parsed:
            result['is_valid'] = bool(parsed['is_valid'])
        if 'location' in parsed and parsed['location'] and parsed['location'] != 'null':
            result['location'] = str(parsed['location'])
        if 'time_entity' in parsed and parsed['time_entity'] and parsed['time_entity'] != 'null':
            result['time_entity'] = str(parsed['time_entity'])
        if 'event_type' in parsed and parsed['event_type'] and parsed['event_type'] != 'null':
            result['event_type'] = str(parsed['event_type'])
        if 'emotion_score' in parsed:
            score = int(parsed['emotion_score'])
            result['emotion_score'] = max(1, min(5, score))
        if 'escalation_signal' in parsed:
            result['escalation_signal'] = bool(parsed['escalation_signal'])
        return result
    except:
        pass

    # 正则兜底
    valid_match = re.search(r'is_valid["\s:]+(true|false)', json_str, re.IGNORECASE)
    if valid_match:
        result['is_valid'] = valid_match.group(1).lower() == 'true'

    loc_match = re.search(r'location["\s:]+"([^"]+)"', json_str)
    if loc_match and loc_match.group(1) != 'null':
        result['location'] = loc_match.group(1)

    event_match = re.search(r'event_type["\s:]+"([^"]+)"', json_str)
    if event_match and event_match.group(1) != 'null':
        result['event_type'] = event_match.group(1)

    score_match = re.search(r'emotion_score["\s:]+(\d+)', json_str)
    if score_match:
        result['emotion_score'] = max(1, min(5, int(score_match.group(1))))

    return result


# ==================== 调用大模型 ====================
def call_llm(text, max_retries=3):
    """调用硅基流动API"""
    # 规则预过滤
    if is_obvious_noise(text):
        return {
            "is_valid": False,
            "location": None,
            "time_entity": None,
            "event_type": None,
            "emotion_score": 1,
            "escalation_signal": False
        }

    prompt = build_prompt(text)

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "你是政务留言分类助手。只输出JSON，不要输出解释。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 300
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)

            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                parsed = parse_response(content)
                return parsed
            else:
                print(f"\nAPI错误 {response.status_code}")

        except Exception as e:
            print(f"\n请求错误: {e}")

        time.sleep(2)

    return {
        "is_valid": True,  # 保守：不确定时判为有效
        "location": None,
        "time_entity": None,
        "event_type": None,
        "emotion_score": 3,
        "escalation_signal": False
    }


# ==================== 测试 ====================
def test_single():
    print("\n=== 测试API ===")
    test_cases = [
        ("华庭小区二次供水水箱长年不洗，水有霉味，请环保局检测", "应该判为有效"),
        ("太差了", "应该判为无效"),
        ("111111", "应该判为无效"),
    ]

    for text, expected in test_cases:
        print(f"\n测试: {text}")
        print(f"预期: {expected}")
        result = call_llm(text)
        print(f"结果: {result}")


# ==================== 主程序 ====================
def main():
    print("=" * 50)
    print("诉求识别 - 大模型批量标注 (Few-shot版)")
    print("=" * 50)

    test_single()

    user_input = input("\n测试成功？按回车继续，输入 n 退出：")
    if user_input.lower() == 'n':
        return

    print(f"\n读取数据: {INPUT_PATH}")
    df = pd.read_excel(INPUT_PATH)
    print(f"总数据量: {len(df)} 条")

    if os.path.exists(OUTPUT_PATH):
        df_existing = pd.read_excel(OUTPUT_PATH)
        start_idx = len(df_existing)
        print(f"已处理: {start_idx} 条，继续")
        df_result = df_existing.copy()
    else:
        start_idx = 0
        df_result = df.copy()
        for col in ['is_valid', 'location', 'time_entity', 'event_type', 'emotion_score', 'escalation_signal']:
            df_result[col] = None

    total = len(df)
    print(f"\n开始处理...\n")

    with tqdm(total=total, initial=start_idx, desc="进度") as pbar:
        for idx in range(start_idx, total):
            text = str(df.iloc[idx]['留言详情'])
            result = call_llm(text)

            df_result.loc[idx, 'is_valid'] = result['is_valid']
            df_result.loc[idx, 'location'] = result['location']
            df_result.loc[idx, 'time_entity'] = result['time_entity']
            df_result.loc[idx, 'event_type'] = result['event_type']
            df_result.loc[idx, 'emotion_score'] = result['emotion_score']
            df_result.loc[idx, 'escalation_signal'] = result['escalation_signal']

            pbar.update(1)

            if (idx + 1) % BATCH_SIZE == 0:
                df_result.to_excel(OUTPUT_PATH, index=False)
                print(f"\n[保存] {idx + 1}条")

            time.sleep(0.3)

    df_result.to_excel(OUTPUT_PATH, index=False)
    print(f"\n完成！输出: {OUTPUT_PATH}")

    valid_count = df_result['is_valid'].sum()
    print(f"有效诉求: {valid_count}/{len(df_result)} ({valid_count / len(df_result) * 100:.1f}%)")


if __name__ == "__main__":
    main()