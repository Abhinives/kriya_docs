import os
import json
from typing import Dict, Literal
from typing_extensions import TypedDict
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
import nltk
from dotenv import load_dotenv
import re
import spacy
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langgraph.graph import StateGraph, MessagesState, END, START
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from elsai_core.prompts import PezzoPromptRenderer
from elsai_core.model import AzureOpenAIConnector
from elsai_core.model import AzureOpenAIConnector

nltk.download('punkt')  # Download the Punkt tokenizer models
nltk.download('punkt_tab')
model_name = "en_core_web_trf"

try:
    nlp = spacy.load(model_name)
    print(f"✔️ '{model_name}' is already installed.")
except OSError:
    print(f"⚠️ '{model_name}' not found. Downloading...")
    spacy.cli.download(model_name)
    nlp = spacy.load(model_name)
    print(f"✔️ '{model_name}' has been successfully installed.")
load_dotenv()
# 🛠️ Initialize Pezzo Client
prompt_renderer = PezzoPromptRenderer(
    api_key=os.getenv("PEZZO_API_KEY"),
    project_id=os.getenv("PEZZO_PROJECT_ID"),
    environment=os.getenv("PEZZO_ENVIRONMENT"),
    server_url=os.getenv("PEZZO_SERVER_URL"),
)

# 🔥 Initialize LLM
llm = AzureOpenAIConnector().connect_azure_open_ai("gpt-4o-mini")


# 📌 Define State
class State(MessagesState):
    next: str


# 📌 Define Routing Logic
members = ["title_validation", "word_count","abstract_validation", "tobacco_organization", "author_contribution"]
options = members + ["FINISH"]

system_prompt = (
    "You are a supervisor agent responsible for managing agents: {members}. "
    "Based on the user's request, route it to the appropriate agent. If no agent applies, return FINISH."
).format(members=members)


class Router(TypedDict):
    next: Literal["title_validation","word_count", "FINISH"]  # Determines the next agent


def get_unique_data_names(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    data_names = set()

    # Find all tags with a 'data-name' attribute
    for tag in soup.find_all(attrs={'data-name': True}):
        data_name = tag.get('data-name')
        if data_name:
            data_names.add(data_name)

    return list(data_names)


def validate_word_count(excluded_sections, file_path) -> str:
    """
    Validates if the word count (excluding specified sections) is within the allowed range (3000-4000 words),
    and saves the modified HTML after removing excluded sections.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, "html.parser")

    # Exclude content based on data-name attributes
    for tag in soup.find_all(attrs={"data-name": True}):
        data_name = tag.get("data-name", "").strip().lower()
        if any(excl.lower() in data_name for excl in excluded_sections):
            tag.extract()  # Remove excluded sections

    # Save the modified HTML
    cleaned_html = str(soup)

    # Extract remaining text and count words from cleaned HTML
    text = soup.get_text(separator=" ").strip()
    words = word_tokenize(text)
    word_count = len(words)

    # Define allowed range (3000 - 4000)
    min_limit, max_limit = 2700, 3300
    flag = False
    if word_count<max_limit:
        flag="pass"
        result = f"Word count is {word_count}. The word count is less than the maximum limit of 3300 words."
    else:
        flag = "fail"
        result = f"Word Count Exceeded: Current word count including metadata is {word_count} words, exceeding the limit." 

    return flag, word_count, result


def clean_response_from_db(response):
    """Removes triple backticks and 'json' from the database response."""
    cleaned_content = re.sub(r"^```json\s*", "", response, flags=re.MULTILINE)  # Remove ```html at the start
    cleaned_content = re.sub(r"```[\s]*$", "", cleaned_content, flags=re.MULTILINE)  # Remove ending ```
    return cleaned_content.strip()

def read_html_file(file_path: str) -> str:
    """Reads the HTML content from a file."""
    file_path = file_path.strip("'\"") 
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def supervisor_node(state: State) -> Command[Literal["title_validation", "__end__"]]:
    """Decides which agent should process the task."""
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
    next_agent = response["next"]
    return Command(goto=next_agent if next_agent != "FINISH" else END)


def extract_contents(html_content: str, data_name:str) -> str:
    """Extracts article title based on dataname`."""
    title_text = ""
    soup = BeautifulSoup(html_content, "html.parser")

    title_tags = soup.find_all(attrs={"data-name": data_name})
    for tag in title_tags:
        title_text=title_text+tag.text+" "
    return title_text

def extract_article_title(html_content: str) -> str:
    """Extracts article title based on `data-name="Article Title"`."""
    title_text = ""
    soup = BeautifulSoup(html_content, "html.parser")

    title_tags = soup.find_all(attrs={"data-name": "Article Title"})
    for tag in title_tags:
        title_text=title_text+tag.text+" "
    return title_text
def validate_presence_of_author_list(html_path):
    '''Validate if the html content has an author list and extract the author names.'''
    html_content = open(html_path).read()
    soup = BeautifulSoup(html_content, "html.parser")

    # Extract author paragraphs with data-name="Authors"
    author_paragraphs = soup.find_all("p", {"data-name": "Authors"})

    authors = []

    for para in author_paragraphs:
        # Extract all author spans within the paragraph
        author_spans = para.find_all("span", {"data-name": "Author"})
        for author_span in author_spans:
            # Get first and last name elements
            first_name = author_span.find("span", {"data-name": "First Name"})
            last_name = author_span.find("span", {"data-name": "Last Name"})
            
            # Combine names (handle missing last names)
            if first_name and last_name:
                full_name = f"{first_name.text.strip()} {last_name.text.strip()}"
            elif first_name:
                full_name = first_name.text.strip()
            else:
                continue  # Skip if no name found
            
            authors.append(full_name)
    return {f"Presence of Author list": True if authors else False,'names': authors}

def validate_presence_of_contribution_list(html_path):
    """
    Validate if the html content has a contribution list and extract the contribution section.
    """
    html_content = open(html_path).read()
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # List of potential contribution section headers
    contribution_keywords = [
        'author contribution', 'contributions', 'authors\' role',
        'author\'s role', 'contribution statement', 'role of authors'
    ]
    
    # Try to find by data attributes first (most reliable)
    contribution_head = soup.find(attrs={"data-name": lambda x: x and "contrib" in x.lower()})
    
    # If data attributes not found, try text matching
    if not contribution_head:
        for header in soup.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'div']):
            header_text = header.get_text().strip().lower()
            if any(keyword in header_text for keyword in contribution_keywords):
                contribution_head = header
                break
    
    if not contribution_head:
        return {'presence_of_contribution_list': False}
    
    # Collect all content until next section heading
    section_content = []
    current_element = contribution_head.find_next_sibling()
    
    while current_element:
        # Stop at next section heading (any heading tag or div/p with class indicating section)
        if current_element.name in ['h1', 'h2', 'h3', 'h4'] or \
           (current_element.get('class') and current_element.get('class')[0].startswith('docx_heading')):
            break
            
        # Get clean text while preserving some structure
        if current_element.name == 'p':
            text = ' '.join(current_element.stripped_strings)
            if text:
                section_content.append(text)
                
        current_element = current_element.find_next_sibling()
    
    # Fallback: Check for direct contribution statement data attribute
    if not section_content:
        contribution_statement = soup.find(attrs={"data-name": "Author Contribution Statement"})
        if contribution_statement:
            statement = ' '.join(contribution_statement.stripped_strings)
        else:
            statement = None
    else:
        statement = ' '.join(section_content)
    
    # Return appropriate dictionary format
    if statement and statement.strip():
        return {
            'presence_of_contribution_list': True,
            'statement': statement.strip()
        }
    
    return {'presence_of_contribution_list': False}


def validate_all_authors_presence_in_contributions(html_path:str, llm) -> dict:
    '''Validate if all authors are mentioned in the contribution statement.'''
    author_list_validation = validate_presence_of_author_list(html_path)
    contribution_list_validation = validate_presence_of_contribution_list(html_path)
    if all([author_list_validation['Presence of Author list'],contribution_list_validation['presence_of_contribution_list']]):
        prompt = f"""
You are a scientific publication quality checker. Your task is to analyze the following author list and contribution statement using the CRediT taxonomy roles. Note that author names in the contribution statement may sometimes appear as abbreviations of the full names in the author list.

Author List: {author_list_validation['names']}

Contribution Statement: { contribution_list_validation['statement']}

Tasks:
1. Cross-reference all authors in the author list with the contribution statement, keeping in mind that abbreviations in the contribution statement may represent the full author names.
2. Verify that each contribution mention includes at least one CRediT taxonomy role.
3. Identify and list:
   - Authors from the author list who are missing from the contribution statement.
   - Authors mentioned in the contribution statement who do not have any CRediT roles assigned.

Output Template:
{{
  "flag": "pass" if all authors are mentioned with atleast one CRediT roles, "fail" otherwise",
    "message": "Explanation of why the check failed (if applicable)",
    "missing_authors": ["List of authors missing from contributions if any"],
    "authors_without_credits": ["List of authors in the contribution statement missing CRediT roles if any"]
}}
Important:
Make sure to resturn only the JSON object without any additional explanation or markdown formatting(triple backtick).
"""

    
        # Send the prompt as a HumanMessage
        response = llm([HumanMessage(content=prompt)])
        response_text = response.content 
        
        # Attempt to parse the response as JSON
        try:
            result = json.loads(response_text)
            result['check'] = "Presence of all authors in contribution statement"
        except Exception as e:
            # In case of a JSON parse error, return the raw output for debugging.
            result = {"error": f"Failed to parse JSON: {e}", "raw_response": response_text}
            result['check'] = "Presence of all authors in contribution statement"
        
        return {key: result[key] for key in ['check', 'flag', 'message','missing_authors','authors_without_credits']}


    else:
        return {"check": "Presence of all authors in contribution statement","flag":"fail","message":"Author list or Contribution list not found"}
 
def validate_tobacco_linked_organization(html_path:str, llm, tobacco_organization_list=[], context_window=75): 
    '''Validate if any organization in the html content is a tobacco company or closely related to tobacco.'''

    # Load spaCy's English model
    nlp = spacy.load("en_core_web_trf")

    #load html and extract all text
    html_content = open(html_path).read()
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator=" ")
    tobacco_linked_orgs=[]
    #extract organizations from the text
    doc = nlp(text)
    organizations = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    
    if tobacco_organization_list:
        tobacco_linked_orgs = [org for org in organizations for tobacco_org in tobacco_organization_list
                               if org.lower() in tobacco_org.lower() or tobacco_org.lower() in org.lower()]
        other_orgs = [org for org in organizations if org not in tobacco_linked_orgs]
    else:
        other_orgs = organizations

    accumulated_context = ''
    for org in other_orgs:
        idx = text.find(org)
        if idx != -1:
            org_context = text[max(0, idx - context_window): idx + len(org) + context_window]
            accumulated_context += org_context + '\n'
    
    prompt = f"""
     Act as a forensic document analyst specializing in corporate affiliations. Analyze the following text sentences extracted from a document.
     
     Use your knowledge (and the internet if possible) to check if any organization is a tobacco company or closely related to tobacco like
     a subsidiary or child company of a tobacco corporation, or if their primary business is aligned with tobacco products.
       
     Return results in JSON format: 
     {{<company_name>": "<reason for tobacco association>"}} if any found else return {{}}.
     Sentences are separated by a newline character ('\\n'). 
     Sentences: {accumulated_context}. 
     Note : Make sure the result are accurate and should not have markdown formatting(triple backtick), line breaks, or explanatory text.
    """


    # passing prompt to LLM
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)

    try:
        result = json.loads(response.content)

        result = {"found_organization": result}
        result['check'] = "tobacco linked organization"
        if tobacco_linked_orgs:
            for org in tobacco_linked_orgs:
                result["found_organization"][org] = "Tobacco company or closely related to tobacco (matched in the list)"
        result['flag'] = "fail" if result else "pass"
        return  {key: result[key] for key in ['check', 'flag', 'found_organization']}

    
    except Exception as e:
        if '{}' in response.content:
            result = {}
        if tobacco_linked_orgs:
            for org in tobacco_linked_orgs:
                result["found_organization"][org] = "Tobacco company or closely related to tobacco (matched in the list)"
        result['flag'] = "fail" if result else "pass"
        return {key: result[key] for key in ['check', 'flag', 'found_organization']}
    
# 📌 Title Validation Logic
def validate_article_title(file_path: str) -> str:
    """Validates the article title formatting."""
    html_content = read_html_file(file_path)
    article_title = extract_article_title(html_content)
    article_title = re.sub(r"^\s*Title:\s*", "", article_title, flags=re.IGNORECASE)
    if not article_title:
        return json.dumps({"flag": False, "reason": "Article title not found in HTML."})

   # prompt = prompt_renderer.get_prompt("TitleCaseValidationPrompt")

    prompt="""
NO PREAMBLE.

You are given an article title. Your task is to validate whether it follows the specified casing rules.

---

Rules for Processing the Title:

1] First word capitalization:
   - The first word of the entire title must always start with an uppercase letter.
   - Example (✅ Correct): "Understanding deep learning models"
   - Example (❌ Incorrect): "understanding deep learning models"

2] True-cased format:
   - Each word's first letter should be lowercase, except for proper nouns and the first letter of the entire title.
   - Example (✅ Correct): "The impact of climate change on biodiversity"
   - Example (❌ Incorrect): "The Impact Of Climate Change On Biodiversity"

3] Colon rule:
   - If a colon (":") appears within the title, the first character after it should be lowercase, unless the preceding text represents data, in which case it should be uppercase.
   - Example (✅ Correct): "Machine learning for drug discovery: an innovative approach"
   - Example (❌ Incorrect): "Machine Learning for Drug Discovery: An Innovative Approach"

4] Proper names:
   - Proper names must have correct capitalization.
   - For place names consisting of multiple words, each word that forms the proper noun should be capitalized.
   - Example (✅ Correct): "Geological formations in North India"
   - Example (❌ Incorrect): "Geological formations in north india"

5] Special terms:
   - Standard abbreviations and region-based acronyms must be properly capitalized.
   - Example (✅ Correct): "Geological analysis of SW China"
   - Example (❌ Incorrect): "Geological analysis of Sw China"

6] Species names:
   - Scientific species names must be italicized.
   - Example (✅ Correct): "The behavior of *Homo sapiens* in social groups"
   - Example (❌ Incorrect): "The behavior of Homo Sapiens in social groups"

---

Important Notes:
- Always verify each rule separately.
- If multiple issues are found, list only the incorrect aspects in the output.
- Do not include correctly identified aspects in the reasons.

---

Output Format (JSON):
- The response must be in valid JSON format.
- If the title follows all rules, return:
{{
  "flag": "pass"
}}
- If the title fails validation, return:
{{
  "flag": "fail",
  "reasons": ["Reason 1", "Reason 2", ...],
  "corrected_title": "Suggested correct format of the title."
}}

---

Article Title (Input):
{article_title}
"""


    #prompt = prompt_renderer.get_prompt("TitleCaseValidationPrompt")
    prompt_template = ChatPromptTemplate.from_template(prompt)
    prompt_input = prompt_template.format_messages(article_title=article_title)
    
    # Call LLM with extracted title
    response = llm(prompt_input).content 

    return response

def validate_abstract(file_path:str):
    html_content = ""
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    abstract = extract_contents(html_content, "Abstract Paragraph")
    prompt = prompt_renderer.get_prompt("AbstractValidationPrompt")
    prompt_template = ChatPromptTemplate.from_template(prompt)
    prompt_input = prompt_template.format_messages(abstract_content=abstract)
    
    # Call LLM with extracted title
    response = llm(prompt_input).content 

    return response

def word_count(file_path:str):
    html_content = ""
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    unique_data_names = get_unique_data_names(html_content)
    prompt  = prompt_renderer.get_prompt("testing")

    prompt_template = ChatPromptTemplate.from_template(prompt)

    prompt_input = prompt_template.format_messages(data_names=unique_data_names)

    response=llm(prompt_input).content

    print(response)

    data_name_array = eval(clean_response_from_db(response))

    flag, word_count, result = validate_word_count(data_name_array, file_path)

    return flag, word_count, result

def extract_html_content(html: str, exclude_data_names: list) -> str:
    """
    Extracts text from an HTML document, excluding elements with specific data-name attributes.

    :param html: The input HTML content as a string.
    :param exclude_data_names: List of data-name attributes to exclude.
    :return: Extracted text content as a string.
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    # Find all elements with a data-name attribute
    for element in soup.find_all(attrs={"data-name": True}):
        data_name = element.get("data-name", "").lower()
        
        # Check if any exclude data-name is a substring of the element's data-name
        if any(exclude_name.lower() in data_name for exclude_name in exclude_data_names):
            element.decompose()  # Remove the element from the soup
    
    return soup.get_text(separator=' ', strip=True)

def title_validation_node(state: State) -> Command[Literal["supervisor"]]:
    """Calls the title validation function and returns the result."""
    input_data = state["messages"][-1].content  # Extracts input from the last message
    file_path = state["messages"][-1].additional_kwargs.get("html_input_path")

    result = validate_article_title(file_path)
    result = json.loads(result)
    if result["flag"] == "pass":
        final_result = {
            "check_name":"Title Casing Validation",
            "flag":result["flag"],
            "result":"The title follows the specified casing rules."
        }
    else:
        final_result = {
            "check_name":"Title Casing Validation",
            "flag":result["flag"],
            "result":{"reasons":result["reasons"],"corrected_title":result["corrected_title"]},


        }
    return Command(
        update={"messages": [HumanMessage(content=json.dumps(final_result), name="title_validator")]},
        goto="FINISH",
    )

def word_count_agent(state:State)-> Command[Literal["supervisor"]]:
    input_data = state["messages"][-1].content  # Extracts input from the last message
    file_path = state["messages"][-1].additional_kwargs.get("html_input_path")

    flag, count, result = word_count(file_path)
    final_result = {
        "check_name":"Word Count",
        "flag":flag,
        "result":count,
        "message": result
         

    }
    return Command(
        update={"messages": [HumanMessage(content=json.dumps(final_result), name="word_count")]},
        goto="FINISH",
    )

def validate_tobacco_organization_agent(state: State) -> Command[Literal["supervisor"]]:
    file_path = state["messages"][-1].additional_kwargs.get("html_input_path")
    result = validate_tobacco_linked_organization(file_path, llm)
    return Command(
        update={"messages": [HumanMessage(content=json.dumps(result, indent=2))]},
        goto="FINISH",
    )

def validate_prsence_of_author_in_contribution_agent(state: State) -> Command[Literal["supervisor"]]:
    file_path = state["messages"][-1].additional_kwargs.get("html_input_path")
    result = validate_all_authors_presence_in_contributions(file_path, llm)
    return Command(
        update={"messages": [HumanMessage(content=json.dumps(result, indent=2))]},
        goto="FINISH",
    )
def abstract_validation_agent(state:State)-> Command[Literal["supervisor"]]:
    input_data = state["messages"][-1].content  # Extracts input from the last message
    file_path = state["messages"][-1].additional_kwargs.get("html_input_path")

    result = validate_abstract(file_path)
    cleaned_result = clean_response_from_db(result)
    json_result = json.loads(cleaned_result)

    final_result = {
        "check_name":"Abstract Validation",
        "flag":"pass" if json_result["flag"]=="structured" else "fail",
        "result": f"The abstract content is {json_result['flag']}."
    }
    return Command(
        update={"messages": [HumanMessage(content=json.dumps(final_result), name="abstract_validation")]},
        goto="FINISH",
    )
# 📌 Build Graph
builder = StateGraph(State)
builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor_node)
builder.add_node("title_validation", title_validation_node)
builder.add_node("word_count", word_count_agent)
builder.add_node("abstract_validation", abstract_validation_agent)
builder.add_node("tobacco_organization", validate_tobacco_organization_agent)
builder.add_node("author_contribution", validate_prsence_of_author_in_contribution_agent)

graph = builder.compile()


# 📌 Execute Workflow
# user_query = "Count the number of words in the html file"
# user_query = "Check the title casing in the html file"
def run_workflow(user_query, html_input_path):
    inputs = HumanMessage(
        content=user_query,
        additional_kwargs={"html_input_path": html_input_path}
    )

    result = graph.invoke({"messages": [inputs]})
    return result
# user_query = "Validate the abstract content"
# html_input_path = "File5 HTML.html"

# inputs =HumanMessage(

#     content = user_query,
#     additional_kwargs={"html_input_path": html_input_path}
# )

# result = graph.invoke({"messages": [inputs]})
# print(result)
