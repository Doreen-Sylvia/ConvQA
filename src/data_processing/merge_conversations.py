import json
import random
from typing import List, Dict, Any


class DialogueMerger:
    def __init__(self, data_file: str):
        """初始化对话合并器"""
        with open(data_file, 'r', encoding='utf-8') as f:
            self.original_data = json.load(f)

        # 按种子实体分组对话
        self.dialogues_by_entity = self._group_dialogues_by_entity()

    def _group_dialogues_by_entity(self) -> Dict[str, List[Dict]]:
        """按种子实体文本分组对话"""
        groups = {}
        for dialogue in self.original_data:
            entity_text = dialogue.get('seed_entity_text', 'unknown')
            if entity_text not in groups:
                groups[entity_text] = []
            groups[entity_text].append(dialogue)
        return groups

    def get_available_entities(self) -> List[str]:
        """获取所有可用的实体"""
        return list(self.dialogues_by_entity.keys())

    def merge_two_topics(self, entity1: str, entity2: str) -> Dict:
        """合并两个主题的对话，只生成一个合并结果"""
        # 获取两个主题的对话
        dialogues1 = self.dialogues_by_entity.get(entity1, [])
        dialogues2 = self.dialogues_by_entity.get(entity2, [])

        if not dialogues1 or not dialogues2:
            print(f"警告: 未找到实体 '{entity1}' 或 '{entity2}' 的对话")
            return None

        # 随机选择两个对话
        dialogue1 = random.choice(dialogues1)
        dialogue2 = random.choice(dialogues2)

        # 创建合并的对话
        merged_conv = self._create_merged_conversation(
            dialogue1, dialogue2, entity1, entity2
        )

        return merged_conv

    def _create_merged_conversation(self, dialogue1: Dict, dialogue2: Dict, entity1: str, entity2: str) -> Dict:
        """创建合并的对话，使用固定的交替模式"""
        try:
            # 提取两个对话的问题
            questions1 = dialogue1['questions']
            questions2 = dialogue2['questions']

            # 固定使用交替模式，不随机选择
            pattern = 'alternate'

            # 固定主题顺序（按字母顺序排序，确保一致性）
            topics = sorted([dialogue1['seed_entity_text'], dialogue2['seed_entity_text']])

            # 确定哪个对话对应哪个主题
            if dialogue1['seed_entity_text'] == topics[0]:
                first_dialogue, second_dialogue = dialogue1, dialogue2
                first_questions, second_questions = questions1, questions2
                first_entity, second_entity = entity1, entity2
            else:
                first_dialogue, second_dialogue = dialogue2, dialogue1
                first_questions, second_questions = questions2, questions1
                first_entity, second_entity = entity2, entity1

            # 创建合并的问题序列
            merged_questions = []
            turn = 0

            if pattern == 'alternate':
                # 交替模式: 1-2-1-2-1
                max_turns = min(len(first_questions), len(second_questions))
                for i in range(max_turns):
                    # 第一个主题的问题
                    if i < len(first_questions):
                        q1 = self._adapt_question(first_questions[i], topics[0], turn, is_transition=(i == 0))
                        if q1:
                            merged_questions.append(q1)
                            turn += 1

                    # 第二个主题的问题（带过渡）
                    if i < len(second_questions):
                        q2 = self._adapt_question(second_questions[i], topics[1], turn, is_transition=True)
                        if q2:
                            merged_questions.append(q2)
                            turn += 1

            # 确保至少有问题
            if not merged_questions:
                return None

            # 构建合并的对话
            merged_dialogue = {
                "domain": "books",
                "conv_id": f"mixed_{topics[0].replace(' ', '_')}_{topics[1].replace(' ', '_')}",
                "topics": topics,
                "questions": merged_questions,
                "seed_entities": [
                    {"entity": first_entity, "text": topics[0]},
                    {"entity": second_entity, "text": topics[1]}
                ]
            }

            return merged_dialogue

        except Exception as e:
            print(f"创建合并对话时出错: {e}")
            return None

    def _adapt_question(self, original_question: Dict, topic: str, turn: int, is_transition: bool = False) -> Dict:
        """调整问题以适应合并对话"""
        try:
            # 使用完整问题（如果存在），否则使用原始问题
            question_text = original_question.get('completed_question') or original_question['question']

            # 如果是主题切换，添加过渡短语
            if is_transition:
                transition_phrases = [
                    f"Speaking of {topic}, ",
                    f"On the topic of {topic}, ",
                    f"Switching to {topic}, ",
                    f"Regarding {topic}, ",
                    f"About {topic}, ",
                    f"Now let's talk about {topic}. ",
                    f"Moving to {topic}, ",
                    f"Another interesting book is {topic}. ",
                    f"Let's discuss {topic}. ",
                    f"Changing subjects to {topic}, "
                ]
                transition = random.choice(transition_phrases)
                # 确保问题首字母小写（如果过渡短语以句号结束）
                if transition.endswith('. '):
                    question_text = question_text[0].lower() + question_text[1:]
                question_text = transition + question_text
            else:
                # 对于连续问题，可能添加连贯性短语
                coherence_phrases = [
                    "Continuing with this, ",
                    "Also, ",
                    "Additionally, ",
                    "Furthermore, ",
                    "On a related note, ",
                    "Building on that, ",
                    "Next, ",
                    "Following up, "
                ]
                if random.random() < 0.3:  # 30%的概率添加连贯性短语
                    coherence = random.choice(coherence_phrases)
                    question_text = coherence + question_text[0].lower() + question_text[1:]

            # 处理答案字段 - 保持原始结构
            answer_data = {
                "answer": original_question['answer'],
                "answer_text": original_question['answer_text']
            }

            # 保留原始问题ID和其他可能存在的字段
            result = {
                "turn": turn,
                "topic": topic,
                "original_question": original_question['question'],
                "question": question_text,
                "answer": original_question['answer'],
                "answer_text": original_question['answer_text'],
                "question_id": f"mixed_{original_question['question_id']}"
            }

            # 保留原始问题中的其他字段（如completed_question等）
            for key, value in original_question.items():
                if key not in ['turn', 'question', 'question_id']:
                    result[f"original_{key}"] = value

            return result

        except Exception as e:
            print(f"调整问题时出错: {e}, 问题数据: {original_question}")
            return None

    def create_comprehensive_dataset(self) -> List[Dict]:
        """创建包含多个主题对的综合数据集，每对主题只生成一个合并对话"""
        entities = self.get_available_entities()
        print(f"找到 {len(entities)} 个实体: {entities}")

        all_merged = []
        # 使用集合来跟踪已经处理过的主题对
        processed_pairs = set()

        # 为每一对不同的实体生成合并对话
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                topic1 = entities[i]
                topic2 = entities[j]

                # 创建一个标准化的主题对标识符（按字母顺序排序）
                pair_key = tuple(sorted([topic1, topic2]))

                # 检查是否已经处理过这个主题对
                if pair_key in processed_pairs:
                    continue

                # 标记这个主题对已经处理过
                processed_pairs.add(pair_key)

                print(f"合并 {topic1} 和 {topic2}...")
                merged = self.merge_two_topics(topic1, topic2)
                if merged:
                    all_merged.append(merged)
                    print(f"  生成了 1 个对话")
                else:
                    print(f"  生成失败")

        return all_merged


def main():
    # 使用示例
    merger = DialogueMerger('../../data/row/ConvQuestions_train/train_set/train_set_books.json')

    # 查看可用的实体
    entities = merger.get_available_entities()
    print("可用的实体:", entities)

    #创建综合数据集（使用所有主题对）
    print("\n创建综合数据集...")
    comprehensive_data = merger.create_comprehensive_dataset()

    if comprehensive_data:
        with open("../../data/merged_dialogues/comprehensive_merged_dialogues.json", 'w', encoding='utf-8') as f:
            json.dump(comprehensive_data, f, indent=2, ensure_ascii=False)
        print(f"综合数据集包含 {len(comprehensive_data)} 个对话，保存到 comprehensive_merged_dialogues.json")


if __name__ == "__main__":
    main()