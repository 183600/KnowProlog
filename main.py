# main_app.py
import os
from pyswip import Prolog
# 修改导入，使用 zhipuai
from zhipuai import ZhipuAI

# --- 配置 ---
# 你的智谱AI API Key (JWT)
# !!! 警告: 直接硬编码 API Key 是不安全的实践。请使用环境变量。!!!
# 为了演示，我们暂时直接使用，但强烈建议修改为从环境变量读取。
ZHIPU_API_KEY = "b6e23abec9b746adb20cb0f3ae116f8e.LIlANJZWSUCbXLnX"
# os.environ["ZHIPU_API_KEY"] = "你的实际API Key" # 推荐方式

try:
    # 初始化 ZhipuAI 客户端
    # SDK 会自动处理 API 地址，我们不需要显式传入 https://open.bigmodel.cn/api/paas/v4/
    client = ZhipuAI(api_key=ZHIPU_API_KEY)
except Exception as e: # ZhipuAI SDK 可能抛出特定的错误，但 Exception 能捕获通用问题
    print(f"ZhipuAI API key not found or invalid, or client initialization failed.")
    print(f"Error: {e}")
    print("Please ensure the ZHIPU_API_KEY is set correctly if using environment variables,")
    print("or that the hardcoded key is correct and valid.")
    exit()

PROLOG_KB_FILE = "knowledge_base.pl"
LLM_MODEL = "glm-4-flash" # 你指定的模型

# --- Prolog 接口 (与之前相同) ---
class PrologInterface:
    def __init__(self, kb_file):
        self.prolog = Prolog()
        try:
            self.prolog.consult(kb_file)
            print(f"Prolog knowledge base '{kb_file}' loaded successfully.")
        except Exception as e:
            print(f"Error loading Prolog knowledge base '{kb_file}': {e}")
            raise

    def query(self, prolog_query_string):
        print(f"Executing Prolog query: {prolog_query_string}")
        try:
            results = list(self.prolog.query(prolog_query_string))
            print(f"Prolog raw results: {results}")

            if not results:
                return {"success": False, "solutions": []}
            elif results == [{}]:
                return {"success": True, "solutions": [{}]}
            else:
                decoded_results = []
                for res_dict in results:
                    decoded_dict = {}
                    for k, v in res_dict.items():
                        if isinstance(v, bytes):
                            decoded_dict[k] = v.decode('utf-8')
                        else:
                            decoded_dict[k] = v
                    decoded_results.append(decoded_dict)
                return {"success": True, "solutions": decoded_results}
        except Exception as e:
            print(f"Error executing Prolog query '{prolog_query_string}': {e}")
            return {"success": False, "error": str(e), "solutions": []}

# --- LLM 接口 (修改为使用 ZhipuAI) ---
class LLMConverter:
    def __init__(self, model_name):
        self.model_name = model_name

    def _get_prolog_schema(self):
        # 这个函数保持不变
        return """
        Available Prolog predicates:
        - parent(Parent, Child): Parent is a parent of Child.
        - male(Person): Person is male.
        - female(Person): Person is female.
        - father(Father, Child): Father is the father of Child.
        - mother(Mother, Child): Mother is the mother of Child.
        - grandparent(Grandparent, Grandchild): Grandparent is a grandparent of Grandchild.
        - grandfather(Grandfather, Grandchild): Grandfather is a grandfather of Grandchild.
        - grandmother(Grandmother, Grandchild): Grandmother is a grandmother of Grandchild.
        - sibling(Sibling1, Sibling2): Sibling1 and Sibling2 are siblings.
        - brother(Brother, Sibling): Brother is the brother of Sibling.
        - sister(Sister, Sibling): Sister is the sister of Sibling.
        - likes(Person, Thing): Person likes Thing.

        Constants (names, things) should be lowercase atoms (e.g., john, mary, apples).
        Variables should be uppercase (e.g., X, Parent, Child).
        Queries must end with a period (.).
        """

    def nl_to_prolog(self, natural_language_question):
        schema = self._get_prolog_schema()
        # 提示词结构基本保持，但根据模型特性可能需要微调
        # ZhipuAI 的模型通常对中文提示词有良好支持
        # 我们可以将 system prompt 和 user prompt 分开
        system_prompt = f"""
        你是一位 Prolog 专家程序员。你的任务是根据提供的模式，将自然语言问题转换成一个有效的 Prolog 查询。

Prolog 模式：
{schema}

说明：

    仅使用模式中定义的谓词。
    确保所有常量（例如名字）均为小写。
    确保所有变量均为大写（例如：X, Y, Person）。
    查询必须以句点（.）结尾。
    如果问题是“是/否”类型的问题，Prolog 查询应反映这一点（例如：parent(john, mary).）。
    如果问题询问“谁”、“什么”等，则使用变量（例如：father(X, mary).）。
    仅输出 Prolog 查询。不要添加任何解释或介绍性文字。
    如果你无法根据模式自信地将问题转换为有效的查询，则输出确切的字符串 "CONVERSION_FAILED"。
        """
        user_prompt = f"""
        Natural Language Question:
        "{natural_language_question}"

        Prolog Query:
        """
        print(f"\n--- NL to Prolog Conversion ---")
        print(f"LLM Input (System Prompt for NL to Prolog):\n{system_prompt}")
        print(f"LLM Input (User Prompt for NL to Prolog):\n{user_prompt}")

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1 # 低温以获得更确定的输出
            )
            prolog_query = response.choices[0].message.content.strip()
            print(f"LLM Output (Prolog Query): {prolog_query}")

            if prolog_query == "CONVERSION_FAILED" or not prolog_query.endswith('.'):
                return None
            return prolog_query
        except Exception as e:
            print(f"Error during NL to Prolog conversion with ZhipuAI: {e}")
            return None

    def prolog_results_to_nl(self, original_question, prolog_query, prolog_results):
        system_prompt = "You are a helpful assistant."
        user_prompt = f"""
        The user asked the following question:
        "{original_question}"

        This question was translated into the Prolog query:
        `{prolog_query}`

        The Prolog query returned the following results:
        {prolog_results}

        Based on these results, provide a concise and natural language answer to the user's original question.
        - If `success` is `false` in the results, it might mean the query had an error or no information was found.
          If `error` is present, mention it. Otherwise, state that no information was found or the statement is false.
        - If `solutions` is an empty list `[]` (and success is true), it means the query is false or no answer was found.
        - If `solutions` is `{{}}` (and success is true), it means the query is true (for yes/no questions).
        - If `solutions` contains dictionaries with variable bindings, use them to construct the answer.
          For example, if results are `{{'success': True, 'solutions': [{{'X': 'john'}}]}}` for a query like `father(X, mary).`,
          a good answer would be "John is the father of Mary." or "The father of Mary is John.".
        - If there are multiple solutions, list them clearly or summarize.

        Natural Language Answer:
        """
        print(f"\n--- Prolog Results to NL Conversion ---")
        print(f"LLM Input (System Prompt for Prolog Results to NL):\n{system_prompt}")
        print(f"LLM Input (User Prompt for Prolog Results to NL):\n{user_prompt}")

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7
            )
            nl_answer = response.choices[0].message.content.strip()
            print(f"LLM Output (Natural Language Answer): {nl_answer}")
            return nl_answer
        except Exception as e:
            print(f"Error during Prolog results to NL conversion with ZhipuAI: {e}")
            return "I encountered an error while trying to formulate the answer."

# --- Main Application Logic (与之前相同) ---
def main():
    print("Initializing system...")
    try:
        prolog_interface = PrologInterface(PROLOG_KB_FILE)
    except Exception:
        print("Failed to initialize Prolog interface. Exiting.")
        return

    llm_converter = LLMConverter(LLM_MODEL)
    print(f"System initialized with ZhipuAI model: {LLM_MODEL}. You can now ask questions.")
    print("Type 'exit' or 'quit' to end the session.")

    while True:
        user_question = input("\nAsk a question: ")
        if user_question.lower() in ['exit', 'quit']:
            break

        if not user_question:
            continue

        # 1. NL to Prolog
        prolog_query = llm_converter.nl_to_prolog(user_question)

        if not prolog_query:
            print("LLM: I couldn't convert your question into a Prolog query. Please try rephrasing.")
            continue
        if prolog_query == "CONVERSION_FAILED":
            print("LLM: I'm unable to convert this specific question to a Prolog query based on my current knowledge. Please try rephrasing or ask something different.")
            continue

        # 2. Execute Prolog Query
        prolog_results = prolog_interface.query(prolog_query)

        # 3. Prolog Results to NL
        natural_language_answer = llm_converter.prolog_results_to_nl(user_question, prolog_query, prolog_results)

        # 4. Display Answer
        print(f"\nAnswer: {natural_language_answer}")

if __name__ == "__main__":
    main()