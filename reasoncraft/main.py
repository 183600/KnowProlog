import os
from pyswip import Prolog
from openai import OpenAI # Renamed from openai to avoid conflict if you have the old one
import re

# --- 配置 ---
API_KEY = "sk-sieqwktkdspwgkbibsygcopuyxgmqgobixrxnjodsowpbggi" # 你的智谱AI API Key
MODEL_NAME = "Qwen/Qwen3-8B"
API_ENDPOINT = "https://api.siliconflow.cn/v1"

PROLOG_KB_PATH = "kb.pl"

# 初始化 OpenAI 客户端 (适用于兼容 OpenAI API 的服务)
client = OpenAI(
    api_key=API_KEY,
    base_url=API_ENDPOINT
)

def load_prolog_kb_metadata(kb_path):
    """
    从Prolog知识库文件中提取元数据信息。
    元数据以特定格式的注释形式存在，例如:
    % predicate: name/arity - description.
    % type: name(type1, type2).
    % rule: name/arity - rule_definition. - description.
    """
    metadata = {
        "predicates": [],
        "types": [],
        "rules": []
    }
    if not os.path.exists(kb_path):
        print(f"Error: Knowledge base file {kb_path} not found.")
        return metadata

    try:
        with open(kb_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith("% predicate:"):
                    match = re.match(r"% predicate:\s*(\w+/\d+)\s*-\s*(.*)", line)
                    if match:
                        metadata["predicates"].append({"signature": match.group(1), "description": match.group(2)})
                elif line.startswith("% type:"):
                    match = re.match(r"% type:\s*(\w+\(.*\))\s*\.?\s*(.*)", line) # Allow optional description
                    if match:
                         metadata["types"].append({"signature": match.group(1), "description": match.group(2) or "No description."})
                elif line.startswith("% rule:"):
                    match = re.match(r"% rule:\s*(\w+/\d+)\s*-\s*(.+?)\s*-\s*(.*)", line)
                    if match:
                        metadata["rules"].append({
                            "signature": match.group(1),
                            "definition_summary": match.group(2),
                            "description": match.group(3)
                        })
    except Exception as e:
        print(f"Error reading or parsing metadata from {kb_path}: {e}")
    return metadata

def format_metadata_for_llm(metadata):
    """将提取的元数据格式化为LLM易于理解的文本。"""
    formatted_text = "Prolog Knowledge Base Schema:\n\n"
    
    if metadata["predicates"]:
        formatted_text += "Available Predicates (fact structure):\n"
        for p in metadata["predicates"]:
            formatted_text += f"- {p['signature']}: {p['description']}\n"
        formatted_text += "\n"

    if metadata["types"]:
        formatted_text += "Parameter Types (examples):\n"
        for t in metadata["types"]:
            formatted_text += f"- {t['signature']}: {t['description']}\n"
        formatted_text += "\n"

    if metadata["rules"]:
        formatted_text += "Available Rules (derived predicates):\n"
        for r in metadata["rules"]:
            formatted_text += f"- {r['signature']} (e.g., `{r['definition_summary']}`): {r['description']}\n"
        formatted_text += "\n"
    
    formatted_text += "You MUST use ONLY the predicates and rules listed above. \n"
    formatted_text += "Variables should be uppercase (e.g., X, Country). Atoms (strings) should be lowercase and enclosed in single quotes if they contain spaces or special characters, otherwise quotes are optional but good practice for consistency (e.g., 'paris', france, 'Guido van Rossum').\n"
    return formatted_text

def nl_to_prolog_query(natural_language_question, kb_metadata_str):
    """使用LLM将自然语言问题转换为Prolog查询。"""
    system_prompt = (
        "You are an expert Prolog programmer. Your task is to convert a natural language question "
        "into a syntactically correct Prolog query that can be executed against a given knowledge base. "
        "You will be provided with a description of the available predicates, their types, and any important rules."
    )
    
    user_prompt = (
        f"{kb_metadata_str}\n"
        f"Natural language question: \"{natural_language_question}\"\n\n"
        "Based on the schema and the question, provide ONLY the Prolog query. "
        "For example, if the question is 'What is the capital of France?', "
        "a valid query would be `capital_of(Capital, france).` or `capital_of(X, france).` "
        "If the question is 'Who is the father of Mary?', a valid query would be `father(Father, mary).`"
        "Prolog query:"
    )

    print("--- Sending to LLM for Prolog Query Generation ---")
    print(f"System Prompt: {system_prompt[:200]}...") # Print snippet
    print(f"User Prompt: {user_prompt[:500]}...") # Print snippet

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2, # Lower temperature for more deterministic output
            max_tokens=100
        )
        prolog_query = response.choices[0].message.content.strip()
        # Clean up potential markdown code blocks or extra text
        prolog_query = prolog_query.replace("`", "").replace("prolog", "").strip()
        if not prolog_query.endswith('.'):
            prolog_query += '.'
        print(f"LLM generated Prolog query: {prolog_query}")
        return prolog_query
    except Exception as e:
        print(f"Error calling LLM for query generation: {e}")
        return None

def execute_prolog_query(prolog_instance, query_string):
    """执行Prolog查询并返回结果。"""
    print(f"Executing Prolog query: {query_string}")
    try:
        results = list(prolog_instance.query(query_string))
        if not results: # query returned False (no solutions)
            return []
        if results == [{}]: # query returned True (e.g. for a fact check with no variables)
             return [{"result": True}]
        return results
    except Exception as e:
        print(f"Error executing Prolog query '{query_string}': {e}")
        return [{"error": str(e)}]


def translate_results_to_nl(original_question, prolog_query, results):
    """使用LLM将Prolog查询结果翻译回自然语言。"""
    if not results or "error" in results[0]:
        return "The query could not be answered due to an error or no results."

    system_prompt = (
        "You are a helpful assistant. Your task is to explain the results of a Prolog query "
        "in natural language, in response to an original user question."
    )
    
    user_prompt = (
        f"The user asked: \"{original_question}\"\n"
        f"This was translated to the Prolog query: `{prolog_query}`\n"
        f"The query returned the following results (list of dictionaries, where each dictionary is a solution):\n{results}\n\n"
        "Please provide a clear and concise natural language explanation of these results. "
        "If there are multiple results, list them or summarize them appropriately. "
        "If the result is `[{'result': True}]`, it means the query without variables was true. "
        "If the result is `[]`, it means the query was false or found no matches."
    )
    
    print("\n--- Sending to LLM for Result Translation ---")
    # print(f"User Prompt for translation: {user_prompt}") # Can be verbose

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,
            max_tokens=300
        )
        explanation = response.choices[0].message.content.strip()
        print(f"LLM translated explanation: {explanation}")
        return explanation
    except Exception as e:
        print(f"Error calling LLM for result translation: {e}")
        return "Could not translate results due to an API error."

def main():
    # 1. 加载Prolog知识库
    prolog = Prolog()
    try:
        prolog.consult(PROLOG_KB_PATH)
        print(f"Successfully consulted Prolog knowledge base: {PROLOG_KB_PATH}")
    except Exception as e:
        print(f"Error consulting Prolog KB: {e}")
        return

    # 2. 加载并格式化KB元数据给LLM
    kb_metadata = load_prolog_kb_metadata(PROLOG_KB_PATH)
    if not kb_metadata["predicates"] and not kb_metadata["rules"]: # Basic check
        print("Warning: No metadata (predicates/rules) found. LLM might struggle to generate queries.")
    
    kb_metadata_str = format_metadata_for_llm(kb_metadata)
    # print("\n--- Knowledge Base Schema for LLM ---")
    # print(kb_metadata_str) # For debugging

    print("\nWelcome to the Prolog+LLM Q&A System!")
    print("Type 'exit' to quit.")

    while True:
        user_question = input("\nAsk a question: ")
        if user_question.lower() == 'exit':
            break

        # 3. NL -> Prolog Query (via LLM)
        prolog_query_str = nl_to_prolog_query(user_question, kb_metadata_str)

        if not prolog_query_str:
            print("Could not generate Prolog query.")
            continue
        
        # 4. Execute Prolog Query
        raw_results = execute_prolog_query(prolog, prolog_query_str)

        print("\n--- Raw Prolog Results ---")
        if not raw_results:
            print("No results found or query was false.")
        elif "error" in raw_results[0]:
            print(f"Error: {raw_results[0]['error']}")
        else:
            for res in raw_results:
                print(res)
        
        # 5. Translate results to NL (via LLM)
        nl_explanation = translate_results_to_nl(user_question, prolog_query_str, raw_results)
        
        print("\n--- LLM Explanation ---")
        print(nl_explanation)

if __name__ == "__main__":
    main()