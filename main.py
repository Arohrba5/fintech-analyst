# This Streamlit app loads a pre-built vector index of fintech company 10-K filings and enables semantic search over the documents.
# The index is stored in the 'index_storage' folder and is generated with the build_index.py script.

import os
from dotenv import load_dotenv

import streamlit as st
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores.types import MetadataFilters, ExactMatchFilter

# === Load .env file to access OPENAI_API_KEY ===
load_dotenv()

# === Setup Streamlit front end ===
st.title("Competitive Intelligence: Fintech Payments")
st.subheader("Compare PayPal, Square, Toast, and Fiserv")

# === Load the vector index from disk ===
@st.cache_resource
def load_vector_index():
    storage_context = StorageContext.from_defaults(persist_dir="index_storage")
    return load_index_from_storage(storage_context)

index = load_vector_index()

# === Create the LLM (GPT-4) ===
llm = OpenAI(model="gpt-4")

# === Define companies ===
company_list = ["PayPal", "Square", "Toast", "Fiserv"]

# === Retrieval settings ===
N_CHUNKS_PER_COMPANY = 4  # Set at 4 so to prevent hitting OpenAI 10000 token limit

# === Streamlit input ===
query = st.text_input("What would you like to know?")

# === If user submits a question ===
if st.button("Ask"):
    if query:
        # Prompt to guide GPT-4 behavior
        system_prompt = (
            "You are a management consultant with expertise in fintech payments. "
            "You provide clear, strategic, and actionable insights grounded strictly in the provided 10-K filings. "
            "You will receive information from multiple companies' 10-K filings. "
            "Base your answer **only on relevant information from these filings**. "
            "If the question pertains to a specific company, focus only on that company's filings. "
            "If the question requires comparing multiple companies, use information from each relevant filing. "
            "Clearly state if information is not available in the filings. "
            "In the 'See sources' section, only show excerpts that directly support your answer."
        )

        # === Step 1: Retrieve the best N chunks per company ===
        all_nodes = []
        for company in company_list:
            filters = MetadataFilters(filters=[ExactMatchFilter(key="company", value=company)])

            retriever = VectorIndexRetriever(
                index=index,
                filters=filters,
                similarity_top_k=N_CHUNKS_PER_COMPANY
            )

            retrieved_nodes = retriever.retrieve(query)
            all_nodes.extend(retrieved_nodes)

        # === Step 2: Combine the content of all nodes ===
        context_texts = []
        for node in all_nodes:
            # Get content and metadata
            content = node.node.get_content()
            company = node.node.metadata.get("company", "Unknown Company")

            # Prepend metadata to content
            labeled_content = (
                f"[Company: {company}]\n\n"
                f"{content}"
            )
            context_texts.append(labeled_content)

        combined_context = "\n\n---\n\n".join(context_texts)

        # === Step 3: Build the full prompt ===
        full_prompt = f"{system_prompt}\n\nContext:\n{combined_context}\n\nQuestion: {query}"

        # === Step 4: Send the full prompt to GPT-4 ===
        response = llm.complete(full_prompt)

        # === Step 5: Display the answer ===
        st.subheader("Answer:")
        st.write(response.text)

        # === Step 6: Show source nodes with metadata ===
        with st.expander("See sources"):
            for node in all_nodes:
                company = node.node.metadata.get("company", "Unknown Company")

                st.markdown(f"**Company:** {company}")
                st.write(node.node.get_content()[:500])  # Preview first 500 characters

    else:
        st.warning("Please enter a question!")
