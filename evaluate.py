import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from chat import query_rag

# Load API keys
load_dotenv()

# 1. Set up our "Judge"
# We use a strict temperature=0 so the judge is consistent
judge_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# 2. Define the Evaluation Prompt
# This tells the LLM how to act as a strict grader
eval_prompt = ChatPromptTemplate.from_template("""
You are a strict legal grading assistant. 
You will be given a QUESTION, a source CONTEXT, and a GENERATED ANSWER.
Your job is to determine if the GENERATED ANSWER is fully supported by the CONTEXT (Faithfulness) 
and if it actually answers the QUESTION (Relevance).

QUESTION: {question}
CONTEXT: {context}
GENERATED ANSWER: {answer}

Reply with exactly one word:
"PASS" if the answer is factually correct based on the context AND answers the question.
"FAIL" if the answer hallucinates outside information OR does not answer the question.
""")

judge_chain = eval_prompt | judge_llm

# 3. Define our test questions
test_questions = [
    "What are the rules regarding speed limits?",
    "What is the penalty for driving without a license?"
]

print("--- Starting Custom RAG Evaluation ---\n")

passed = 0
for q in test_questions:
    print(f"Testing Question: '{q}'")
    
    # Run the question through our RAG pipeline
    response = query_rag(q)
    bot_answer = response["answer"]
    contexts = "\n\n".join([doc.page_content for doc in response["context"]])
    
    # Pass the results to our Judge LLM
    grade = judge_chain.invoke({
        "question": q,
        "context": contexts,
        "answer": bot_answer
    }).content.strip()
    
    if grade == "PASS":
        print("PASS: The answer is faithful and relevant.")
        passed += 1
    else:
        print("FAIL: The answer hallucinated or was irrelevant.")
        print(f"Bot Answer was: {bot_answer}")
    print("-" * 40)

print(f"\nFinal Score: {passed}/{len(test_questions)} Passed")
