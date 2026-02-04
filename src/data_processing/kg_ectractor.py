# coding:utf-8
import json
from collections import defaultdict
import os

class DialogueToKGConverter:
    """
    将对话数据转换为知识图谱三元组格式
    """

    def __init__(self, input_file):
        self.input_file = input_file
        self.entities = set()
        self.relations = set()
        self.triples = []

    def extract_triples_from_dialogue(self, dialogue):
        """
        从单个对话中提取三元组
        """
        triples = []

        # 创建从问题到答案的映射
        for question in dialogue['questions']:
            # 实体: 书籍、作者等
            # 关系: 问题类型对应的属性关系

            # 提取实体
            topic_entity = question['topic']
            self.entities.add(topic_entity)

            # 根据问题类型创建关系和三元组
            original_question = question['original_question'].lower()
            answer_text = question['answer_text']

            # 添加答案实体
            if answer_text and answer_text != "":
                self.entities.add(answer_text)

                # 根据问题类型确定关系
                if 'author' in original_question:
                    relation = 'author'
                elif 'nationality' in original_question or 'country' in original_question:
                    relation = 'nationality'
                elif 'year' in original_question:
                    relation = 'publication_year'
                elif 'title' in original_question:
                    relation = 'book_title'
                elif 'award' in original_question:
                    relation = 'award'
                elif 'publisher' in original_question:
                    relation = 'publisher'
                elif 'genre' in original_question:
                    relation = 'genre'
                else:
                    relation = 'related_to'

                self.relations.add(relation)
                triples.append((topic_entity, relation, answer_text))

        return triples

    def convert(self):
        """
        转换整个数据集
        """
        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for dialogue in data:
            triples = self.extract_triples_from_dialogue(dialogue)
            self.triples.extend(triples)

        # 去重
        self.triples = list(set(self.triples))

        return self.triples, self.entities, self.relations

    def save_dataset(self, output_dir):
        """
        保存为训练数据格式
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 转换数据
        triples, entities, relations = self.convert()

        # 分割数据集 (简单地按8:1:1分割)
        n = len(triples)
        train_end = int(0.8 * n)
        valid_end = int(0.9 * n)

        train_triples = triples[:train_end]
        valid_triples = triples[train_end:valid_end]
        test_triples = triples[valid_end:]

        # 保存三元组文件
        self._save_triples(os.path.join(output_dir, 'train.txt'), train_triples)
        self._save_triples(os.path.join(output_dir, 'valid.txt'), valid_triples)
        self._save_triples(os.path.join(output_dir, 'test.txt'), test_triples)

        # 保存实体和关系字典
        self._save_dict(os.path.join(output_dir, 'entities.dict'), entities)
        self._save_dict(os.path.join(output_dir, 'relations.dict'), relations)

        print(f"数据集已保存到 {output_dir}")
        print(f"训练集: {len(train_triples)} 三元组")
        print(f"验证集: {len(valid_triples)} 三元组")
        print(f"测试集: {len(test_triples)} 三元组")
        print(f"实体数: {len(entities)}")
        print(f"关系数: {len(relations)}")

    def _save_triples(self, filename, triples):
        """
        保存三元组到文件
        """
        with open(filename, 'w', encoding='utf-8') as f:
            for head, relation, tail in triples:
                f.write(f"{head}\t{relation}\t{tail}\n")

    def _save_dict(self, filename, items):
        """
        保存字典到文件
        """
        with open(filename, 'w', encoding='utf-8') as f:
            for idx, item in enumerate(items):
                f.write(f"{item}\t{idx}\n")

def main():
    """
    主函数：执行数据预处理
    """
    # 输入文件路径
    input_file = "../../data/merged_dialogues/comprehensive_merged_dialogues.json"

    # 输出目录
    output_dir = "../../data/preprocessed/dialogue_kg/"

    # 创建转换器并执行转换
    converter = DialogueToKGConverter(input_file)
    converter.save_dataset(output_dir)


if __name__ == "__main__":
    main()
