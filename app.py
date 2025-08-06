
```python
import os
import gradio as gr
import tempfile
from langchain_milvus import Milvus
from langchain_community.llms import Replicate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from transformers import AutoTokenizer
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_community.embeddings import HuggingFaceEmbeddings

# Replicate token and model setup
os.environ['REPLICATE_API_TOKEN'] = "your_replicate_token"
model_path = "ibm-granite/granite-3.3-8b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = Replicate(model=model_path, replicate_api_token=os.environ['REPLICATE_API_TOKEN'])

# Temporary Milvus DB
db_file = tempfile.NamedTemporaryFile(prefix="milvus_", suffix=".db", delete=False).name
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_db = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": db_file},
    auto_id=True,
    index_params={"index_type": "AUTOINDEX"},
)

# Financial Literacy Content
filename = "finance_literacy.txt"
with open(filename, "w") as f:
    f.write("""
UPI: Unified Payments Interface (UPI) is a real-time payment system developed by NPCI. It allows users to link bank accounts and make instant money transfers via mobile.

Avoiding Online Scams: Never share your OTP, UPI PIN, or passwords. Beware of links promising free rewards. Use official banking apps only.

Interest Rates: Interest is the cost of borrowing money. Loans usually have an interest rate. Compare rates from different banks before applying.

Budgeting: Create a monthly budget by listing your income and expenses. Prioritize needs over wants. Track your spending to save money.

Digital Wallets: Apps like PhonePe, Google Pay, Paytm allow you to pay bills, recharge, and send money securely.

Safe Loan Practices: Always read terms before taking a loan. Choose loans with lower interest and no hidden fees.

How to use BHIM app: Download, register with your mobile number linked to your bank, set UPI PIN, and start sending or requesting money.

Fraud Protection: Use two-factor authentication. Don’t save card details on unknown websites. Report suspicious activity to your bank immediately.Explain about How to use Phone pe. Explain Google pay also
""")

# Load & Split Content
loader = TextLoader(filename)
documents = loader.load()
splitter = CharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=tokenizer,
    chunk_size=tokenizer.model_max_length // 2,
    chunk_overlap=0,
)
texts = splitter.split_documents(documents)
for i, doc in enumerate(texts):
    doc.metadata["doc_id"] = i + 1
vector_db.add_documents(texts)

# Prompt Template
template = """
You are a helpful Digital Financial Literacy Assistant.
User Question: {question}
Give a simple, culturally-inclusive explanation in easy language.
Include tips to stay safe, explain tools like UPI, wallets, loans, interest, scams, budgeting.
"""
prompt = PromptTemplate(template=template, input_variables=["question"])

# Chains
llm_chain = LLMChain(llm=model, prompt=prompt)
combine_chain = StuffDocumentsChain(llm_chain=llm_chain)

rag_chain = RetrievalQA(
    retriever=vector_db.as_retriever(),
    combine_documents_chain=combine_chain,
    return_source_documents=False
)

# Gradio Interface
def ask_finance_agent(query):
    try:
        response = rag_chain.run(query)
        return response
    except Exception as e:
        return f"❌ Error: {str(e)}"

iface = gr.Interface(
    fn=ask_finance_agent,
    inputs=gr.Textbox(label="Ask a question about digital finance", placeholder="e.g. How to use UPI?"),
    outputs=gr.Textbox(label="Answer"),
    title="AI Agent for Digital Financial Literacy",
    description="Ask about UPI, loans, scams, budgeting, and digital tools. This assistant helps you stay safe and informed.",
    theme="default"
)

iface.launch(share=True)
