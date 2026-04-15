from neo4j import GraphDatabase
import difflib

# ==================== 配置 ====================
NEO4J_URI = "bolt://localhost:7687"  # 默认地址
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"  # 请替换为你的密码

# 预定义问题关键词列表（用于相似度匹配）
PROBLEM_KEYWORDS = [
    '施工安全', '工地隐患', '扬尘污染', '施工扰民', '夜间施工',
    '物业纠纷', '物业收费', '停车费', '群租房', '违规出租',
    '房屋质量', '房屋漏水', '竣工验收', '老旧小区改造', '加装电梯',
    '公积金贷款', '公积金提取', '道路损坏', '井盖缺失', '路灯故障',
    '垃圾堆积', '环境卫生', '停水', '水压不足', '供暖问题', '暖气不热',
    '违章建筑', '违法建设', '房产证办理', '停车管理', '消防通道堵塞',
    '安全隐患', '拖欠工资', '绿化问题', '电梯故障'
]


class DepartmentQuery:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def find_similar_keyword(self, input_text, threshold=0.4):
        """计算输入文本与关键词的相似度，返回最匹配的关键词"""
        # 使用difflib计算相似度
        matches = []
        for keyword in PROBLEM_KEYWORDS:
            # 检查包含关系
            if keyword in input_text or input_text in keyword:
                similarity = 0.9
            else:
                similarity = difflib.SequenceMatcher(None, input_text, keyword).ratio()

            if similarity >= threshold:
                matches.append((keyword, similarity))

        # 按相似度降序排序
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[0] if matches else (None, 0)

    def query_department(self, keyword):
        """根据关键词查询责任部门"""
        with self.driver.session() as session:
            # 查询主责部门
            result_primary = session.run(
                """
                MATCH (p:Problem {keyword: $keyword})-[r:HAS_RESPONSIBILITY {type: '主责'}]->(d:Department)
                RETURN d.name AS department, r.confidence AS confidence, r.type AS type
                ORDER BY r.confidence DESC
                """,
                keyword=keyword
            )

            # 查询协同部门
            result_collab = session.run(
                """
                MATCH (p:Problem {keyword: $keyword})-[r:HAS_RESPONSIBILITY {type: '协同'}]->(d:Department)
                RETURN d.name AS department, r.confidence AS confidence, r.type AS type
                ORDER BY r.confidence DESC
                """,
                keyword=keyword
            )

            primary_depts = [dict(record) for record in result_primary]
            collab_depts = [dict(record) for record in result_collab]

            return primary_depts, collab_depts

    def search(self, user_input):
        """主查询函数：输入文本，输出责任部门"""
        print(f"\n{'='*60}")
        print(f"用户输入: {user_input}")
        print(f"{'='*60}")

        # 1. 相似度匹配
        matched_keyword, similarity = self.find_similar_keyword(user_input)

        if not matched_keyword:
            print("\n⚠️ 未找到匹配的问题类型，请尝试更具体的关键词。")
            print(f"建议的关键词示例: {', '.join(PROBLEM_KEYWORDS[:10])}...")
            return

        print(f"\n🔍 匹配到问题类型: 【{matched_keyword}】（相似度: {similarity:.2f}）")

        # 2. 查询责任部门
        primary_depts, collab_depts = self.query_department(matched_keyword)

        # 3. 输出结果
        print(f"\n📋 责任部门：")
        if primary_depts:
            print("   【主责部门】")
            for dept in primary_depts:
                print(f"      → {dept['department']}（置信度: {dept['confidence']*100:.0f}%）")
        else:
            print("   【主责部门】未找到，建议联系街道办事处或12345热线")

        if collab_depts:
            print("\n   【协同部门】")
            for dept in collab_depts:
                print(f"      → {dept['department']}（置信度: {dept['confidence']*100:.0f}%）")


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 连接数据库
    query_tool = DepartmentQuery(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    print("=" * 60)
    print("北京城乡建设领域诉求智能分拨系统")
    print("=" * 60)
    print("\n支持的问题类型：")
    print(", ".join(PROBLEM_KEYWORDS))

    while True:
        print("\n" + "-" * 40)
        user_input = input("请输入诉求内容（输入 q 退出）: ").strip()

        if user_input.lower() == 'q':
            break

        if not user_input:
            print("请输入有效内容")
            continue

        query_tool.search(user_input)

    query_tool.close()
    print("\n感谢使用！")


# ==================== 测试用例 ====================
def test():
    """测试函数"""
    query_tool = DepartmentQuery(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    test_cases = [
        "华庭小区二次供水水箱长年不洗，水有严重霉味",
        "小区物业乱收停车费，300块一个月",
        "工地半夜还在施工，吵得睡不着",
        "公积金贷款办不下来，说没钱放贷",
        "门口井盖丢了，晚上看不见很危险",
        "开发商盖的房子漏水，质量太差"
    ]

    for case in test_cases:
        query_tool.search(case)
        print("\n" + "-" * 40)

    query_tool.close()


# 取消注释运行测试
# test()