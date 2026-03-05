from typing import List
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
import json
import re
from groq import Groq
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import chainlit as cl

# Agent and Tool Imports
from langchain_core.tools import tool, Tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_classic.agents import initialize_agent, AgentType
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# ============================================================================
# Configuration
# ============================================================================

try:
    # Ensure .files directory exists for Chainlit elements
    os.makedirs(".files", exist_ok=True)
except OSError:
    print("Warning: Could not create .files directory. Proceeding anyway (likely on a read-only filesystem like Vercel).")

PDF_FOLDER_PATH = "./fin_ed_docs"  # Your PDF documents folder
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing! Please add it to your .env file.")

# Model Configuration
GROQ_MODEL = "llama-3.3-70b-versatile"  # Latest Groq model (Nov 2025)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Open-source embeddings

# RAG Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TEMPERATURE = 0

# ============================================================================
# Document Processing Functions
# ============================================================================

def load_and_process_pdfs(pdf_folder_path: str) -> List[Document]:
    """
    Load and process PDF documents from a folder.
    
    Args:
        pdf_folder_path: Path to folder containing PDF files
        
    Returns:
        List of processed document chunks
    """
    documents = []
    
    # Check if folder exists
    if not os.path.exists(pdf_folder_path):
        raise FileNotFoundError(f"PDF folder not found: {pdf_folder_path}")
    
    # Load all PDF files
    pdf_files = [f for f in os.listdir(pdf_folder_path) if f.endswith('.pdf')]
    
    if not pdf_files:
        raise ValueError(f"No PDF files found in {pdf_folder_path}")
    
    print(f"Loading {len(pdf_files)} PDF files...")
    
    for file in pdf_files:
        pdf_path = os.path.join(pdf_folder_path, file)
        try:
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
            print(f"✓ Loaded: {file}")
        except Exception as e:
            print(f"✗ Error loading {file}: {str(e)}")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    splits = text_splitter.split_documents(documents)
    print(f"✓ Created {len(splits)} document chunks")
    
    return splits


def initialize_vectorstore(splits: List[Document], embeddings_model) -> FAISS:
    """
    Initialize FAISS vector store with document chunks.
    
    Args:
        splits: List of document chunks
        embeddings_model: Embedding model instance
        
    Returns:
        FAISS vector store
    """
    print("Creating vector store...")
    vectorstore = FAISS.from_documents(
        documents=splits, 
        embedding=embeddings_model
    )
    print("✓ Vector store created successfully")
    return vectorstore


# ============================================================================
# Initialize Models and Vector Store
# ============================================================================

print("=" * 60)
print("Initializing RAG Chatbot...")
print("=" * 60)

# Initialize embeddings model (Hugging Face - open source)
print("\n1. Loading embeddings model...")
embeddings_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'},  # Use 'cuda' if GPU available
    encode_kwargs={'normalize_embeddings': True}
)
print(f"✓ Loaded: {EMBEDDING_MODEL}")

# Load and process PDFs
print(f"\n2. Processing PDFs from: {PDF_FOLDER_PATH}")
splits = load_and_process_pdfs(PDF_FOLDER_PATH)

# Initialize vector store
print("\n3. Building vector database...")
vectorstore = initialize_vectorstore(splits, embeddings_model)

# Initialize Groq LLM (latest version)
print("\n4. Initializing Groq LLM...")
model = ChatGroq(
    model=GROQ_MODEL,
    temperature=TEMPERATURE,
    groq_api_key=GROQ_API_KEY,
    max_tokens=8192,  # Maximum context for response
)
print(f"✓ Model: {GROQ_MODEL}")

print("\n" + "=" * 60)
print("✓ Initialization Complete!")
print("=" * 60 + "\n")


# ============================================================================
# Chainlit Event Handlers
# ============================================================================

@cl.on_chat_start
async def on_chat_start():
    """
    Initialize the complete Agent session when user connects.
    """
    
    await cl.Message(
        content="🚀 **Welcome to Your Agentic AI Assistant!**\n\n"
                f"Powered by **Groq** ({GROQ_MODEL}) and **LangChain Agents**.\n\n"
                "I am an intelligent agent. I can autonomously decide to:\n"
                "1. 📚 Read your internal PDF documents.\n"
                "2. 🌐 Search the live internet for recent news or general facts.\n\n"
                "💬 Ask me anything!"
    ).send()
    
    # 1. Create CUSTOM PDF Retriever Tool (preserves metadata)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    def pdf_search_func(query: str) -> str:
        """Search internal PDFs and return results WITH source metadata."""
        docs = retriever.invoke(query)
        results = []
        for i, doc in enumerate(docs):
            source_file = os.path.basename(doc.metadata.get('source', 'Unknown'))
            page_num = doc.metadata.get('page', 'N/A')
            content = doc.page_content.strip()
            results.append({
                "chunk": i + 1,
                "file": source_file,
                "page": page_num,
                "content": content
            })
        return json.dumps(results, indent=2)
    
    retriever_tool = Tool(
        name="financial_education_pdf_search",
        func=pdf_search_func,
        description="Search and return information from the internal PDF database. Returns results with exact file names, page numbers, and text excerpts. Use this for financial education, investing, and regulation questions."
    )
    
    # 2. Create Web Search Tool (returns structured results with URLs)
    web_search_tool = DuckDuckGoSearchResults(
        name="web_search",
        description="Search the live internet. Returns results with URLs, titles, and snippets. Use this for current events, recent news, live data, or general knowledge."
    )
    
    # Compile tools list
    tools = [retriever_tool, web_search_tool]
    
    # 3. Create the Agent
    # For Langchain 0.1/0.2 we use initialize_agent with memory
    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True
    )
    
    agent_executor = initialize_agent(
        tools=tools,
        llm=model,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        agent_kwargs={
            "system_message": (
                "You are a helpful and intelligent AI assistant. You have access to internal financial PDFs and the live internet.\n"
                "Use the 'financial_education_pdf_search' tool for questions about finance concepts, regulations, and education.\n"
                "Use the 'web_search' tool for recent news, live data, or general knowledge.\n"
                "If you use a tool, make sure to synthesize the information clearly into your final answer."
            )
        }
    )
    
    cl.user_session.set("agent", agent_executor)
    print(f"✓ New Agent session started")


@cl.on_message
async def on_message(message: cl.Message):
    """
    Handle incoming messages from users and route through the Agent Executor.
    """
    agent = cl.user_session.get("agent")
    
    if not agent:
        await cl.Message(content="❌ Session missing. Please refresh.").send()
        return
    
    # UI Loading indicator
    thinking_msg = cl.Message(content="🤔 Agent is thinking and deciding which tool to use...")
    await thinking_msg.send()
    
    try:
        # Run the agent (no callback handler - incompatible with LangChain 1.0)
        res = await agent.ainvoke(
            {"input": message.content}
        )
        
        answer = res["output"]
        intermediate_steps = res.get("intermediate_steps", [])
        
        await thinking_msg.remove()
        
        # Build sidebar elements with REAL source proof (only visible on click)
        text_elements = []
        tools_used = []
        source_labels = []  # Short labels for inline display
        
        for step_idx, step in enumerate(intermediate_steps):
            action, observation = step
            tool_name = action.tool
            tools_used.append(tool_name)
            obs_text = str(observation)
            
            # === PDF SEARCH: detailed info in sidebar only ===
            if tool_name == "financial_education_pdf_search":
                try:
                    chunks = json.loads(obs_text)
                except Exception:
                    chunks = []
                
                if chunks:
                    sidebar_content = "📚 **PDF Sources Used**\n\n"
                    file_summary_parts = []
                    
                    for chunk in chunks:
                        fname = chunk.get('file', 'Unknown')
                        page = chunk.get('page', 'N/A')
                        content = chunk.get('content', '')
                        
                        sidebar_content += f"---\n"
                        sidebar_content += f"### 📄 {fname} — Page {page}\n\n"
                        sidebar_content += f"{content}\n\n"
                        
                        file_summary_parts.append(f"`{fname}` (p.{page})")
                    
                    # Short inline label
                    source_labels.append(f"📚 {', '.join(file_summary_parts)}")
                    
                    text_elements.append(
                        cl.Text(
                            content=sidebar_content,
                            name="📚 View PDF Sources",
                            display="side"
                        )
                    )
                else:
                    source_labels.append("📚 Internal PDFs")
            
            # === WEB SEARCH: detailed info in sidebar only ===
            elif tool_name == "web_search":
                sidebar_content = f"🌐 **Web Search Results**\n\n**Query:** {action.tool_input}\n\n"
                link_summary_parts = []
                
                try:
                    pattern = r'\[snippet:\s*(.*?),\s*title:\s*(.*?),\s*link:\s*(.*?)\]'
                    matches = re.findall(pattern, obs_text)
                    
                    if matches:
                        for i, (snippet, title, link) in enumerate(matches, 1):
                            sidebar_content += f"---\n"
                            sidebar_content += f"### {i}. [{title.strip()}]({link.strip()})\n\n"
                            sidebar_content += f"{snippet.strip()}\n\n"
                            link_summary_parts.append(title.strip())
                    else:
                        sidebar_content += obs_text
                
                except Exception:
                    sidebar_content += obs_text
                
                if link_summary_parts:
                    source_labels.append(f"🌐 {len(link_summary_parts)} web results")
                else:
                    source_labels.append("🌐 Web search")
                
                text_elements.append(
                    cl.Text(
                        content=sidebar_content,
                        name="🌐 View Web Sources",
                        display="side"
                    )
                )
            
            else:
                source_labels.append(f"🔧 {tool_name}")
            
        # --- Clean, compact footer with clickable source links ---
        if tools_used:
            unique_tools = list(set(tools_used))
            tool_icons = {"financial_education_pdf_search": "📚 PDF Search", "web_search": "🌐 Web Search"}
            tool_labels = [tool_icons.get(t, t) for t in unique_tools]
            
            answer += f"\n\n---\n🛠️ **Tools Used:** {', '.join(tool_labels)}"
            answer += f"\n📋 **Sources:** {' | '.join(source_labels)}"
            
            # Reference element names so Chainlit renders them as clickable links
            element_names = [el.name for el in text_elements]
            if element_names:
                answer += "\n\n**👇 Click to view full source details:**\n"
                for name in element_names:
                    answer += f"\n{name}"
        else:
            answer += "\n\n---\n🧠 **Agent answered directly from memory.**"
            
        await cl.Message(
            content=answer, 
            elements=text_elements
        ).send()
        
    except Exception as e:
        await thinking_msg.remove()
        await cl.Message(content=f"❌ **Error:** {str(e)}").send()
        print(f"Error: {str(e)}")


@cl.on_chat_end
async def on_chat_end():
    """
    Clean up when chat session ends.
    """
    print("✓ Chat session ended")


# ============================================================================
# Run Instructions
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("To run this application, use:")
    print("  chainlit run app.py -w")
    print("=" * 60 + "\n")