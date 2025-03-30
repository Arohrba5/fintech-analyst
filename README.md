# Fintech Payments Competitive Intelligence Prototype

## 1. Product Overview

This tool helps fintech analysts quickly review and compare 2024 10-K filings from key competitors (specifically: PayPal, Square, Toast, and Fiserv). It enables users to ask strategic questions and compare solutions. Built with Retrieval-Augmented Generation, it leverages vector search and GPT-4 to deliver concise, cited answers based on the filings.

Example questions it can answer include:
- 1. What are the sources of revenue for Toast?
- 2. What are business risks facing Fiserve
- 3. Compare PayPal's strategic initiatives with Square's strategic initiatives.

---

## 2. Target Users

- **Primary**: Fintech strategy analysts and decision-makers.  
- **Secondary**: Sales teams.

---

## 3. Key Technical Decisions / Tradeoffs

- **LLM**: I chose OpenAI GPT-4 for reliable outputs / language reasoning and because I had $15 of credit from my previous project.   
- **RAG Stack**: I used LlamaIndex for vector indexing and retrieval because its lightweight, simple, and free. 
- **Chunking Strategy**: I used UnstructuredReader to read each file and SentenceSplitter to create 512 token chunks (with 50 token overlap). This approach caused issues throughout the project, in particular not being able to extract metadata.
- **Metadata**: I tagged each chunk with its respective 10-K company name for query filtering. I needed to do this, so I would not lose company information tied to each chunk.

---

## 4. Architecture Diagram

PDF Filings → build_index.py → LlamaIndex Vector Store (index_storage)
                                     ↓
                    main.py Streamlit App → GPT-4 Completion
                                     ↓
                          Answers + Source Chunks

---

## 5. How to run the app

After you clone the repo and set your OPENAI key in a .env file:
1. Run the build_index.py script first (python build_index.py) to set up the vector index
2. Run main.py as a Streamlit app (streamlit run main.py) to launch the frontend app

You can access the app as localhost.

---

## 6. Known Bugs / Limitations / Improvements
- **No Section Metadata**: I couldn't figure out how to extract section-level metadata (e.g., key risks section of 10-K), which would better inform the LLM. This should be fixed in a subsequent version
- **UI Limitations**: I should set up the chat to enable follow-up questions 
- **Limited Chunk Retrieval**: Retrieval pulls a fixed number of chunks per company (currently set to 4). Worth testing more chunks per company.
- **Token Limits**: GPT-4 context window and OpenAI rate limits may restrict large queries. I was hitting a 10000 token limit from OpenAI.

