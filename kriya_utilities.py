import mammoth
import re
def convert_docx_to_html(docx_path):
    with open(docx_path, "rb") as docx_file:
        result = mammoth.convert(docx_file)
    return result.value  # Returns the HTML string

def clean_html_from_db(response):
    """Removes triple backticks and 'html' from the database response."""
    cleaned_content = re.sub(r"^```html\s*", "", response, flags=re.MULTILINE)  # Remove ```html at the start
    cleaned_content = re.sub(r"```[\s]*$", "", cleaned_content, flags=re.MULTILINE)  # Remove ending ```
    return cleaned_content.strip()