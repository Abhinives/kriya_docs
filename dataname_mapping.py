from kriya_utilities import convert_docx_to_html, clean_html_from_db



from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import re
from collections import defaultdict
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import uuid
from elsai_core.prompts import PezzoPromptRenderer
from elsai_core.model import OpenAIConnector
import os
import json

load_dotenv()
llm = OpenAIConnector().connect_open_ai("gpt-4o")

prompt_renderer = PezzoPromptRenderer(
        api_key=os.getenv("PEZZO_API_KEY"),
        project_id=os.getenv("PEZZO_PROJECT_ID"),
        environment=os.getenv("PEZZO_ENVIRONMENT"),
        server_url=os.getenv("PEZZO_SERVER_URL"),
    )
def split_html_into_chunks(html_content, max_chunk_size=2000):
    """Splits HTML content into chunks based on headers and size."""
    html_content = re.sub(r'\s+', ' ', html_content)

    section_keywords = [
        'Abstract', 'Keywords', 'Introduction', 'Method', 'Results', 'Discussion', 'Conclusion',
        'References', 'Acknowledgments', 'Objectives', 'Strengths and Limitations', 'Trial Registration'
    ]

    section_pattern = r'(<h\d.*?>.*?</h\d>|<strong>.*?</strong>)'
    paragraphs = re.split(section_pattern, html_content)

    chunks = []
    current_chunk = ''
    current_section = ''
    image_references = {}
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        

        def replace_image(match):
            src_link = match.group(1)
            unique_id = str(uuid.uuid4())  # Generate a unique ID
            image_references[unique_id] = src_link
            return f'<img src="{unique_id}">'

        para = re.sub(r'<img\s+src="([^"]+)"\s*/?>', replace_image, para)
        header_match = re.match(r'<(strong|h\d).*?>(.*?)</\1>', para)
        if header_match:
            section_name = header_match.group(2).strip()
            if any(keyword.lower() in section_name.lower() for keyword in section_keywords):
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ''
                current_section = section_name
                current_chunk += para
                continue

        if len(current_chunk) + len(para) > max_chunk_size:
            chunks.append(current_chunk)
            current_chunk = para
        else:
            current_chunk += para

    if current_chunk:
        chunks.append(current_chunk)

    with open('image_references.json', 'w') as json_file:
        json.dump(image_references, json_file, indent=4)

    return chunks

def chunk_html_by_sections_1(html_content):
    """Splits HTML content into sections based on headings, keywords, and short bold text."""
    soup = BeautifulSoup(html_content, "html.parser")
    chunks = defaultdict(list)
    current_section = "Uncategorized"  

    section_keywords = {
        "abstract", "introduction", "methods", "results", "discussion",
        "conclusion", "references", "acknowledgment", "materials", "funding"
    }

    for tag in soup.find_all(["h1", "h2", "h3", "p", "div", "b", "strong"]):  
        text = tag.get_text(strip=True)

        # **Detect new section if it's a heading**
        if tag.name in ["h1", "h2", "h3"]:
            current_section = text  
        
        # **Detect new section if it matches a known keyword**
        elif text.lower() in section_keywords:
            current_section = text  

        # **Detect bold text (≤5 words) as potential section (only if unique)**
        elif tag.name in ["b", "strong"] and len(text.split()) <= 5:
            if text not in chunks:  # Avoid duplicate sections from multiple bold tags
                current_section = text  
        
        # **Append content to the section instead of overwriting**
        chunks[current_section].append(tag.prettify())

    return ["".join(content) for content in chunks.values()]
def chunk_html_by_sections(html_content):
    """Splits HTML content into sections based on headings."""
    soup = BeautifulSoup(html_content, "html.parser")
    chunks = defaultdict(list)
    current_section = None  # Default section if no heading at start

    for tag in soup.find_all(["h1", "h2", "h3", "p"]):
        if tag.name in ["h1", "h2", "h3"]:  
            current_section = tag.get_text(strip=True)  
        if current_section is None:  # Handle content before the first heading
            current_section = "Uncategorized"
        chunks[current_section].append(tag.prettify())


    return ["".join(chunks[section]) for section in chunks] 
def process_chunk(chunk):
    """Processes a chunk using LLM."""
    prompt = prompt_renderer.get_prompt("DataNameAttributes1")
    prompt_template = ChatPromptTemplate.from_template(prompt)
    prompt_input = prompt_template.format_messages(html_content=chunk)
    llm_response = llm(prompt_input).content  # LLM call (assumed synchronous)
    cleaned_response = clean_html_from_db(llm_response)
    return cleaned_response
async def process_chunks_async(chunks):
    """Executes LLM processing on HTML chunks in parallel using asyncio."""
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as executor:
        tasks = [loop.run_in_executor(executor, process_chunk, chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks)
    return results

async def map_data_name_attributes(html_content):
    """Splits HTML content and processes chunks in parallel."""
    chunks = split_html_into_chunks(html_content)
    processed_chunks = await process_chunks_async(chunks)
    return "".join(processed_chunks) 
# with_basic_structure = wrap_html_with_structure(html)
# print(with_basic_structure)
# html_with_attributes = map_data_name_attributes(html)
# cleaned_html = clean_html_from_db(html_with_attributes)
# print(cleaned_html)



# htmltext = html_to_text(html)
# count= count_words(htmltext)

def map_data_names(file_path:str):

    html_content = convert_docx_to_html(file_path)



    mapped = asyncio.run(map_data_name_attributes(html_content))
    cleaned = clean_html_from_db(mapped)

    output_file = "mapped_output.html"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(cleaned)

    print(f"Mapped HTML saved to {output_file}")
    return os.path.abspath(output_file)
