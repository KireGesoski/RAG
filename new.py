# from LangMem import LangMem
#
# mem = LangMem()
#
# # Store facts
# mem.add_memory("User likes concise answers.")
# mem.add_memory("User is working on Magento 2.4.7 with PHP 8.2.")
# mem.add_memory("We discussed guardrails for LLM tuning.")
#
# # Ask a question
# query = "What Magento version am I using?"
# relevant_memories = mem(query)
#
# print("Relevant memories:")
# for m in relevant_memories:
#     print("-", m)




#-------------------------------------------------------------------------
# from LangMemHybrid import LangMemHybrid
# mem = LangMemHybrid()
#
# # --- Semantic memory ---
# mem.add_semantic("Paris is the capital of France.")
# mem.add_semantic("User uses Magento 2.4.7 with PHP 8.2.")
#
# # --- Episodic memory ---
# mem.add_episode("Chatted about guardrails for LLM tuning.", context="work session")
# mem.add_episode("User said their name is Alice.", context="intro conversation")
#
# # --- Procedural memory ---
# def greet(name: str):
#     return f"Hello, {name}!"
#
# mem.add_skill("greet", greet)
#
# # --- Queries ---
# print(mem("What is the capital of France?"))
# # -> {'semantic': ['Paris is the capital of France.'], 'episodes': [...], 'skills': ['greet']}
#
# print(mem.run_skill("greet", "Alice"))
# # -> "Hello, Alice!"




#--------------------------------------------------------------------------
# from LangMemHybridEmbedding import LangMemHybridEmbedding
# # Init
# mem = LangMemHybridEmbedding(provider="openai")
#
# # Episodic
# mem.add_episode("User: Hello, I am Kire", context="chat")
# mem.add_episode("Assistant: Nice to meet you!", context="chat")
#
# # Semantic
# mem.add_semantic("The user's name is Kire.")
# mem.add_semantic("The user works with Magento 2.4.7 and PHP 8.2.")
#
# # Procedural
# def greet_user(name): return f"Hello {name}, welcome back!"
# mem.add_skill("greet_user", greet_user)
#
# # Query memory
# print(mem("What do I know about the user?"))





#---------------------------------------------------------------------------
# from llama_cpp import Llama
# from LangMemHybridEmbedding import LangMemHybridEmbedding
# from SimpleGuard import GuardBlocked, SimpleGuard
# import os
# from db_connect import Neo4jSandbox
# from Agent import Agent
# from conf_file import openAi_key
# uri = "bolt://44.203.117.86:7687"
# user = "neo4j"
# password = "lasers-excesses-contribution"
#
# db = Neo4jSandbox(uri, user, password)
# llm = Llama(
#     model_path="models/qwen2.5-0.5b-instruct/qwen2.5-0.5b-instruct-q4_0.gguf",
#     n_ctx=32768,
#     verbose=False
# )
# SYS = (
#   "If you are not certain, respond exactly with: NO_ANSWER\n"
#   "If certain, respond with: ANSWER: <concise fact>\n"
#   "Do not add anything else."
# )
#
# def call_llm_direct(question: str) -> str:
#     print('in the model')
#     resp = llm.create_chat_completion(
#         messages=[
#             {"role": "system", "content": SYS},
#             {"role": "user", "content": question}
#         ],
#         max_tokens=120,
#         temperature=0.2,
#         top_p=0.9,
#         repeat_penalty=1.15,
#         stop=["</s>"]
#     )
#     # return resp["choices"][0]["text"].strip()
#     text = resp["choices"][0]["message"]["content"].strip()
#     print('Text', resp)
#
#     if text.startswith("NO_ANSWER"):
#         return 'No Answer Provided'
#     return text
#
#
# guards = SimpleGuard()
# mem = LangMemHybridEmbedding()
#
# try:
#     agent = Agent(openAi_key)
#     while True:   # infinite loop until you stop with Ctrl+C
#         q = input("👉 Enter statement or question: ").strip()
#         agent.classify_input(q)
#         if not q.endswith("?"):
#             try:
#                 safe_text = guards.run(
#                     q,
#                     lambda x: x,  # identity function (no LLM call, just validation)
#                     mask_input_pii=True,
#                     block_on_malware_input=True,
#                     block_on_malware_output=True,
#                     block_on_output_pii=True
#                 )
#                 db.add_info_to_person("Kire", safe_text)
#                 typ = mem.classify_memory(safe_text)
#                 method_name = f"add_{typ}"  # e.g., "add_skill"
#                 method = getattr(mem, method_name)  # resolves to mem.add_skill
#                 method(safe_text)
#                 print("💾 Stored as sensitive info:", safe_text, "as Type:", typ)
#             except GuardBlocked as e:
#                 print("❌ Blocked from storing sensitive info:", e)
#             continue
#
#         past = mem.search_semantic(q, top_k=1)
#         past_db = db.get_info("Kire")
#         if past:
#             print("📚 From Memory:", past[0])
#             print("📚 From DB:", past_db)
#             continue
#         # Otherwise it's a QUESTION → run through guard + LLM
#         try:
#             answer = guards.run(
#                 q,
#                 call_llm_direct,
#                 mask_input_pii=True,
#                 block_on_malware_input=True,
#                 block_on_malware_output=True,
#                 block_on_output_pii=False
#             )
#             print("✅ From LLM:", answer)
#
#             # Save Q & A into semantic memory
#             # mem.add_semantic(f"Q: {q}")
#             # mem.add_semantic(f"A: {answer}")
#
#
#         except GuardBlocked as e:
#             print("❌ Blocked by guard:", e)
#
# except KeyboardInterrupt:
#     print("\n👋 Exiting.")


# DB CONNECTION-----------------------------------------------------------------------
# from db_connect import Neo4jSandbox
# uri = "bolt://44.203.117.86:7687"
# user = "neo4j"
# password = "lasers-excesses-contribution"
# db = Neo4jSandbox(uri, user, password)
#db.set_property("Kire", "Info", "Loves Python and AI")
# print(db.get_info("Kire"))

# Parallel with agents----------------------------------------
from SimpleLLM import SimpleLLM
from LangMemHybridEmbedding import LangMemHybridEmbedding
from SimpleGuard import GuardBlocked, SimpleGuard
import os
from db_connect import Neo4jSandbox
from Agent import Agent
from conf_file import openAi_key
uri = "bolt://44.203.117.86:7687"
user = "neo4j"
password = "lasers-excesses-contribution"

db = Neo4jSandbox(uri, user, password)
guards = SimpleGuard()
mem = LangMemHybridEmbedding()

try:
    llm = SimpleLLM()
    agent = Agent(openAi_key)
    while True:   # infinite loop until you stop with Ctrl+C
        q = input("👉 Enter statement or question: ").strip()
        safe_text = ''
        try:
            safe_text = guards.run(
                q,
                lambda x: x,  # identity function (no LLM call, just validation)
                mask_input_pii=True,
                block_on_malware_input=True,
                block_on_malware_output=True,
                block_on_output_pii=True
            )
        except GuardBlocked as e:
            print("❌ Blocked from storing sensitive info:", e)
        agent.classify_input(safe_text)
        llm.call_llm_direct(safe_text)
        # db.add_info_to_person("Kire", safe_text)
        # typ = mem.classify_memory(safe_text)
        # method_name = f"add_{typ}"  # e.g., "add_skill"
        # method = getattr(mem, method_name)  # resolves to mem.add_skill
        # method(safe_text)
        # print("💾 Stored as sensitive info:", safe_text, "as Type:", typ)
        # past = mem.search_semantic(q, top_k=1)
        # past_db = db.get_info("Kire")
        # if past:
        #     print("📚 From Memory:", past[0])
        #     print("📚 From DB:", past_db)
        #     continue
        # # Otherwise it's a QUESTION → run through guard + LLM
        # try:
        #     answer = guards.run(
        #         q,
        #         call_llm_direct,
        #         mask_input_pii=True,
        #         block_on_malware_input=True,
        #         block_on_malware_output=True,
        #         block_on_output_pii=False
        #     )
        #     print("✅ From LLM:", answer)
        #
        #     # Save Q & A into semantic memory
        #     # mem.add_semantic(f"Q: {q}")
        #     # mem.add_semantic(f"A: {answer}")
        #
        #
        # except GuardBlocked as e:
        #     print("❌ Blocked by guard:", e)

except KeyboardInterrupt:
    print("\n👋 Exiting.")
