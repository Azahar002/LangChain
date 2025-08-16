from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

model = ChatOpenAI()

prompt1 = PromptTemplate(
    template = "Generate short and simple notes from the following text \m {text}",
    input_variables = ['text']
)

prompt2 = PromptTemplate(
    template = "Generate 5 short questions from the following text \n {text}",
    input_variables = ['text']
)

prompt3 = PromptTemplate(
    template = "merge the provide notes and quiz into a single document. \n notes ->{notes} and quiz -> {quiz}",
    input_variables = ['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt1 | model | parser,
    'quiz': prompt2 | model | parser
})


merge_chain = prompt3 | model | parser

chain = parallel_chain | merge_chain

text = """
Gotcha — if you’re looking at a Confluence **UI dashboard** and just want to make the dropdown edits, you’ll need to get into **Edit mode** for that specific page first.

Here’s the click-path on Confluence Cloud (and similar on Server/Data Center):

---

**1. Open the page you want to edit**

* Navigate to the Confluence page that has your dashboard content.

**2. Click the "Edit" button**

* In the top-right corner of the page, look for a **pencil icon ✏️** or a button labeled **Edit**.
* Click it — this switches you into the page editor.

**3. Place your cursor where you want the dropdown**

* Scroll to the section of the page where you want the new dropdown.

**4. Insert the Expand macro (Confluence Cloud)**

* Type `/expand` in the editor.
* Select **Expand** from the dropdown menu that appears.

**5. Fill out the Expand block**

* In **Title**, type the heading you want (e.g., “Deployment Details”).
* In the **body** area, add your content (text, images, code block, tables, etc.).

**6. Publish the changes**

* Click **Publish** in the top-right to save your changes.

---

If your dashboard is a **Confluence macro page** (e.g., a Page Properties Report or a Jira dashboard inside Confluence), the pencil button still opens the page editor, but you’ll see the macro placeholders. You can still insert expands around or inside those sections.

---

If you want, I can mark **exactly where to click on your screenshot** so you can visually follow it. Do you want me to do that next?

"""


result = chain.invoke({'text': text})

print(result)