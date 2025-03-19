import streamlit as st
import json
import time
from validation_agent2 import run_workflow
from dataname_mapping import map_data_names, llm

def process_queries(file_path, user_queries):
    start_time = time.time()
    mapped_html_path = map_data_names(file_path)
    html_path = "File1 HTML.html"
    st.success("HTML file mapped successfully.")
    results = []
    for user_query in user_queries:
        result = run_workflow(user_query, html_path)
        json_result = json.loads(result["messages"][1].content)
        results.append(json_result)
    st.success("Validation results saved successfully.")
    output_file = "annotated_json.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    
    with open(output_file, "r", encoding="utf-8") as f:
        validation_results = json.load(f)
    
    llm_input = f"Summarize the following validation results:\n{json.dumps(validation_results, indent=4)}"
    response = llm(llm_input).content
    end_time = time.time()
    st.write(f"Total time taken: {end_time - start_time} seconds")
    st.success("LLM response received successfully.")
    return response

st.title("Validation Workflow Processor")

file_path = st.text_input("Enter File Path:")
user_queries = st.session_state.get("user_queries", [])

query = st.text_input("Enter your query:")
if st.button("Add Query"):
    if query:
        user_queries.append(query)
        st.session_state["user_queries"] = user_queries

st.write("**Queries:**")
for i, q in enumerate(user_queries):
    st.write(f"{i+1}. {q}")

if st.button("Submit"):
    if file_path and user_queries:
        response = process_queries(file_path, user_queries)
        st.text_area("LLM Response:", response, height=200)
    else:
        st.error("Please provide a file path and at least one query.")
