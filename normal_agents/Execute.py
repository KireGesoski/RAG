from SimpleAgent import SimpleAgent
from LangMemHybridEmbedding import LangMemHybridEmbedding
from SimpleGuard import GuardBlocked, SimpleGuard
from DataDB import DataDB
import os
from db_connect import Neo4jSandbox
from DeterminationAgent import DeterminationAgent
from QuestionDeterminatorAgent import QuestionDeterminationAgent
from StatementDeterminationAgent import ImportanceAgent
from conf_file import openAi_key
uri = "bolt://44.222.97.237"
user = "neo4j"
password = "wraps-injection-legs"

#db = Neo4jSandbox(uri, user, password)
guards = SimpleGuard()
mem = LangMemHybridEmbedding()

try:
    llm = SimpleAgent()
    initialDeterminationAgent = DeterminationAgent()
    questionAgent = QuestionDeterminationAgent()
    statementAgent = ImportanceAgent()
    db = DataDB(uri, user, password)
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
        determinate = initialDeterminationAgent.classify_text(safe_text)
        print("1", determinate)
        if determinate == "Statement":
            determinate = statementAgent.analyze_text(safe_text)
            print("3", determinate)
            db.set_user_property("U1001", determinate, determinate, safe_text)


        if determinate == "Question":
            determinate = questionAgent.analyze_text(safe_text)

        if determinate is None:
            print('Not authorised to answer')
            exit(0)
        else:
            llm.call_llm_direct(determinate)



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