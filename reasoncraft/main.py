import os
import importlib.resources
from pyswip import Prolog
from pyswip.prolog import PrologError
from zhipuai import ZhipuAI
import re
import traceback

# --- Configuration ---
PACKAGE_NAME = "reasoncraft"
KB_FILENAME = "knowledge_base.pl"
ZHIPU_API_KEY = "b6e23abec9b746adb20cb0f3ae116f8e.LIlANJZWSUCbXLnX"
LLM_MODEL = "glm-4-flash"

# Initialize ZhipuAI client
client = None
try:
    if not ZHIPU_API_KEY or ZHIPU_API_KEY == "YOUR_ZHIPU_API_KEY_HERE":
        raise ValueError("ZHIPU_API_KEY is not set or is a placeholder.")
    client = ZhipuAI(api_key=ZHIPU_API_KEY)
    print("ZhipuAI client initialized successfully.")
except ValueError as ve:
    print(f"Configuration Error: {ve}")
    exit(1)
except Exception as e:
    print(f"ZhipuAI client initialization failed.")
    print(f"Error: {e}")
    print("Please ensure the ZHIPU_API_KEY is set correctly and is valid.")
    exit(1)

# --- Prolog Interface ---
class PrologInterface:
    def __init__(self):
        self.prolog = Prolog()
        actual_kb_path_str = "Unknown (Not loaded)"
        try:
            with importlib.resources.path(PACKAGE_NAME, KB_FILENAME) as actual_kb_path:
                actual_kb_path_str = str(actual_kb_path)
                self.prolog.consult(actual_kb_path_str)
            print(f"Prolog knowledge base '{KB_FILENAME}' from package '{PACKAGE_NAME}' loaded successfully from: {actual_kb_path_str}")
        except FileNotFoundError:
            print(f"CRITICAL ERROR: Prolog knowledge base file '{KB_FILENAME}' not found within the package '{PACKAGE_NAME}'.")
            raise
        except PrologError as e:
            print(f"Error loading Prolog knowledge base '{KB_FILENAME}': {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred while loading Prolog knowledge base: {e}")
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
                            decoded_dict[k] = v.decode('utf-8', errors='replace')
                        else:
                            decoded_dict[k] = v
                    decoded_results.append(decoded_dict)
                return {"success": True, "solutions": decoded_results}
        except PrologError as e:
            print(f"Error executing Prolog query '{prolog_query_string}': {e}")
            return {"success": False, "error": str(e), "solutions": []}
        except Exception as e:
            print(f"Unexpected error executing Prolog query: {e}")
            traceback.print_exc()
            return {"success": False, "error": str(e), "solutions": []}

# --- LLM Interface (ZhipuAI) ---
class LLMConverter:
    def __init__(self, model_name, prolog_interface):
        self.model_name = model_name
        self.prolog_interface = prolog_interface
        # 先定义mbti_examples
        self.mbti_examples = """
Examples of MBTI-related queries:
- "What is the dominant function of ISTJ?" -> dominant_function(istj, Dominant).
- "What are the cognitive functions for ENFP?" -> primary_stack(enfp, Stack).
- "Describe the Ni function" -> function_description(ni, Description).
- "Which types have Te as dominant function?" -> mbti_type(Type, te, _, _, _).
"""
        # 然后生成schema
        self.prolog_schema = self._generate_prolog_schema_from_kb()

    def _generate_prolog_schema_from_kb(self):
        print("Generating Prolog schema from knowledge base...")
        schema_parts = ["Available Prolog predicates:"]
        try:
            predicates_query = "current_predicate(Functor/Arity)"
            results = self.prolog_interface.query(predicates_query)

            if results["success"] and results["solutions"] and results["solutions"] != [{}]:
                valid_solutions = [s for s in results["solutions"] if s and 'Functor' in s and 'Arity' in s]
                sorted_solutions = sorted(valid_solutions, key=lambda x: (x.get('Functor', ''), x.get('Arity', 0)))

                for sol in sorted_solutions:
                    name = sol.get('Functor')
                    arity = sol.get('Arity')

                    if name and isinstance(arity, int):
                        if name.startswith('$') or name.startswith('swi_') or name in ['true', 'false', 'fail']:
                            continue
                        
                        arg_vars = []
                        if arity > 0:
                            base_vars = ['X', 'Y', 'Z', 'A', 'B', 'C']
                            arg_vars = base_vars[:arity] if arity <= len(base_vars) else base_vars + [f"V{i}" for i in range(len(base_vars) + 1, arity + 1)]
                        args_str = ", ".join(arg_vars)
                        
                        if arity == 0:
                            schema_parts.append(f"- {name}.")
                        else:
                            schema_parts.append(f"- {name}({args_str}).")
            else:
                schema_parts.append("- No user-defined predicates automatically detected or error querying predicates.")
                print("Warning: Could not automatically detect user-defined predicates from KB or query failed.")

        except Exception as e:
            print(f"Error generating schema from KB: {e}")
            traceback.print_exc()
            schema_parts.append("- Error occurred while generating predicate list.")

        schema_parts.append("\nGuidelines for Prolog queries:")
        schema_parts.append("- Constants should be lowercase atoms (e.g., istj, infj).")
        schema_parts.append("- Variables should start with an uppercase letter (e.g., Dominant, Stack).")
        schema_parts.append("- All queries MUST end with a period (.).")
        
        # 添加MBTI特定部分
        schema_parts.append("\nMBTI-specific predicates:")
        schema_parts.append("- mbti_type(Type, Dominant, Auxiliary, Tertiary, Inferior).")
        schema_parts.append("- dominant_function(Type, Dominant).")
        schema_parts.append("- primary_stack(Type, [Dominant, Auxiliary, Tertiary, Inferior]).")
        schema_parts.append("- function_description(FunctionAbbr, Description).")
        
        schema_parts.append(self.mbti_examples)
        
        return "\n".join(schema_parts)

    def _get_prolog_schema(self):
        return self.prolog_schema

    def nl_to_prolog(self, natural_language_question):
        schema = self._get_prolog_schema()
        system_prompt = f"""
You are a Prolog expert programmer specialized in MBTI and cognitive functions. 
Convert natural language questions about MBTI types and cognitive functions into valid Prolog queries.

Prolog Schema and Guidelines:
{schema}

Instructions:
1. For questions about a type's dominant function, use: dominant_function(Type, Dominant).
2. For questions about a type's full function stack, use: primary_stack(Type, Stack).
3. For function descriptions, use: function_description(FunctionAbbr, Description).
4. Only use predicates defined in the schema. 
5. Constants must be lowercase (e.g., istj, infj, se, ni).
6. Variables must start with uppercase (e.g., Dominant, Stack).
7. The query MUST end with a period.
8. Output ONLY the Prolog query. No explanations or additional text.
9. If unsure, output "CONVERSION_FAILED".
"""
        user_prompt = f"""
Natural Language Question:
"{natural_language_question}"

Prolog Query:
"""
        print(f"\n--- NL to Prolog Conversion ---")
        print(f"LLM Input (User Prompt for NL to Prolog):\n{user_prompt.strip()}")

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=150
            )
            prolog_query = response.choices[0].message.content.strip()
            print(f"LLM Raw Output (Prolog Query): {prolog_query}")

            if not prolog_query or prolog_query.upper() == "CONVERSION_FAILED":
                print("LLM indicated conversion failed or returned empty.")
                return None
            
            # Clean up the query
            prolog_query = re.sub(r"^(?:```(?:prolog)?\n)?(.*?)(?:\n```)?$", r"\1", prolog_query, flags=re.DOTALL | re.IGNORECASE).strip()
            
            if not prolog_query.endswith('.'):
                prolog_query += '.'
                
            # Validate the query format
            if not re.match(r'^[a-z0-9_]+\([A-Za-z0-9_, ]+\)\.$', prolog_query):
                print(f"Invalid query format: {prolog_query}")
                return None

            print(f"Cleaned Prolog Query: {prolog_query}")
            return prolog_query

        except Exception as e:
            print(f"Error during LLM call for NL to Prolog conversion: {e}")
            traceback.print_exc()
            return None

    def prolog_results_to_nl(self, original_question, prolog_query, prolog_results):
        print(f"\n--- Prolog Results to NL Conversion ---")
        print(f"Original Question: {original_question}")
        print(f"Executed Prolog Query: {prolog_query}")

        system_prompt = """
You are an AI assistant that explains Prolog query results about MBTI and cognitive functions.
Given the original question, the Prolog query, and its results, provide a clear and concise answer.

For dominant function questions:
- If successful: "The dominant function of [TYPE] is [FUNCTION]."
- If failed: "Could not determine the dominant function for [TYPE]."

For function stack questions:
- If successful: List the functions in order.
- If failed: "Could not retrieve the function stack for [TYPE]."

For function descriptions:
- If successful: Provide the description.
- If failed: "Could not find a description for [FUNCTION]."

Be natural and focus on directly answering the question.
"""
        results_summary = ""
        if prolog_results.get("error"):
            results_summary = f"Prolog execution resulted in an error: {prolog_results['error']}"
        elif prolog_results["success"]:
            if prolog_results["solutions"] == [{}]:
                results_summary = "The query was successful (true)."
            elif prolog_results["solutions"]:
                results_summary = "The query was successful with these solutions:\n"
                for sol_set in prolog_results["solutions"]:
                    if not sol_set:
                        continue
                    parts = [f"{var} = {val}" for var, val in sol_set.items()]
                    results_summary += "- " + ", ".join(parts) + "\n"
                results_summary = results_summary.strip()
            else:
                results_summary = "The query was successful but returned no specific results."
        else:
            results_summary = "The query failed or returned no solutions."

        user_prompt = f"""
Original Question:
"{original_question}"

Prolog Query:
`{prolog_query}`

Prolog Results:
{results_summary}

Natural Language Answer:
"""
        print(f"LLM Input (User Prompt for Results to NL):\n{user_prompt.strip()}")

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
                max_tokens=250
            )
            nl_answer = response.choices[0].message.content.strip()
            print(f"LLM Raw Output (NL Answer): {nl_answer}")
            return nl_answer
        except Exception as e:
            print(f"Error during LLM call for Prolog results to NL conversion: {e}")
            traceback.print_exc()
            return "I encountered an error while trying to interpret the results."

# --- Main Application Logic ---
def main():
    print("Initializing ReasonCraft...")
    if client is None:
        print("LLM client (ZhipuAI) is not initialized. Exiting.")
        return

    try:
        prolog_interface = PrologInterface()
    except Exception as e:
        print(f"Failed to initialize PrologInterface: {e}")
        return 

    llm_converter = LLMConverter(model_name=LLM_MODEL, prolog_interface=prolog_interface)

    print("\nWelcome to ReasonCraft!")
    print("Type your question about MBTI and cognitive functions, or 'exit' to quit.")

    while True:
        try:
            natural_language_question = input("\nAsk ReasonCraft> ")
            if natural_language_question.strip().lower() == 'exit':
                print("Exiting ReasonCraft. Goodbye!")
                break
            if not natural_language_question.strip():
                continue

            prolog_query_str = llm_converter.nl_to_prolog(natural_language_question)

            if prolog_query_str:
                prolog_results = prolog_interface.query(prolog_query_str)
                
                nl_answer = llm_converter.prolog_results_to_nl(natural_language_question, prolog_query_str, prolog_results)
                print(f"\nAnswer: {nl_answer}")
            else:
                print("Could not convert your question to a valid Prolog query. Please try rephrasing.")

        except KeyboardInterrupt:
            print("\nExiting ReasonCraft. Goodbye!")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            traceback.print_exc()
            print("Please try again or type 'exit' to quit.")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(script_dir) == PACKAGE_NAME:
        import sys
        parent_dir = os.path.dirname(script_dir)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
            print(f"Development: Added '{parent_dir}' to sys.path to find package '{PACKAGE_NAME}'.")
    elif not os.path.isdir(os.path.join(script_dir, PACKAGE_NAME)):
         print(f"Warning: The package directory '{PACKAGE_NAME}' was not found relative to the script location.")

    main()