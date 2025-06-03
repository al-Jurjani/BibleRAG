# BibleRAG: Retrieval-Augmented Generation for Bible Question Answering

## Project Overview

This project implements and evaluates a Retrieval-Augmented Generation (RAG) system for answering questions based on the King James Version (KJV) of the Bible. The study benchmarks various configurations, including different language models (Qwen2.5 1.5B and 3B), document chunking strategies, embedding models, and retrieval techniques, to assess their impact on answer quality. Evaluation focuses on faithfulness, relevance, and similarity to manually curated ground truth answers.

The complete source code, corpus, and generated results are detailed in this repository and the accompanying project report.

## Dataset

*   **Corpus:** The King James Version (KJV) of the Bible, formatted as a single .txt file.
*   **Source:** Obtained from the OpenBible website.
*   **Domain:** Religious literature, specifically Christian theological texts.
*   **Preprocessing:** The entire text file is read as a single document, which is then segmented into chunks of varying sizes.

## Methodology

The RAG pipeline involves the following key steps:

1.  **Data Loading & Chunking:**
    *   The Bible text is loaded using `TextLoader`.
    *   The text is split into smaller chunks using `RecursiveCharacterTextSplitter` with an overlap of 50 characters. Chunk sizes experimented with include 256, 512, and 1024 characters (the provided notebook focuses on 1024).

2.  **Embedding Models:**
    *   Text chunks are converted into vector embeddings using `HuggingFaceEmbeddings`.
    *   Models evaluated include:
        *   `BAAI/bge-small-en`
        *   `sentence-transformers/paraphrase-MiniLM-L6-v2`
        *   `sentence-transformers/all-MiniLM-L6-v2`

3.  **Vector Store:**
    *   `FAISS` (Facebook AI Similarity Search) is used to create an indexed vector store from the document embeddings, enabling efficient similarity searches.

4.  **Retrieval Techniques:**
    *   Several methods are used to retrieve relevant document chunks for a given question:
        *   **BM25:** A traditional keyword-based retrieval algorithm implemented with `BM25Retriever`.
        *   **Semantic Search:** Cosine similarity search on the FAISS vector store.
        *   **MMR (Maximal Marginal Relevance):** A method to balance relevance and diversity in retrieved documents, using the FAISS vector store.
        *   **Hybrid RRF (Reciprocal Rank Fusion):** Combines the results from BM25, Semantic, and MMR searches using RRF to re-rank and select the top documents.

5.  **Language Models (LLMs) for Generation:**
    *   Open-access language models from Hugging Face are used to generate answers based on the retrieved context and the input question.
    *   Models used in the experiments (as per the report):
        *   `Qwen/Qwen2.5-1.5B-Instruct`
        *   `Qwen/Qwen2.5-3B-Instruct` (used in the provided notebook)
    *   The prompt instructs the LLM to act as a Bible expert and use strictly the provided context.

6.  **Ground Truth Generation:**
    *   Ground truth answers for evaluation were generated using a custom-built ChatGPT model titled "Bible," created by Dan An (as mentioned in the project report).

7.  **Evaluation Metrics:**
    *   A custom `evaluate_response` function computes cosine similarity scores between the generated answer and:
        *   **Faithfulness:** The concatenated content of retrieved documents.
        *   **Relevance:** The original input question.
        *   **Similarity:** The manually curated ground truth answer.
    *   The embedding model used for retrieval is also used for encoding texts for evaluation.

## Key Technologies & Libraries

*   Python 3
*   Jupyter Notebook
*   **Core ML/NLP:**
    *   `transformers`
    *   `sentence-transformers`
    *   `torch`
*   **RAG & Vector Search:**
    *   `langchain` (including `TextLoader`, `RecursiveCharacterTextSplitter`, `HuggingFaceEmbeddings`, `FAISS`, `BM25Retriever`)
    *   `faiss-cpu`
    *   `rank_bm25`
*   **Data Handling:**
    *   `pandas`
    *   `numpy`
*   **Utilities:**
    *   `scikit-learn` (for `cosine_similarity`)
    *   `time`
    *   `warnings`

## Setup & Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```
2.  Ensure you have Python 3 installed.
3.  Install the required packages. You can typically do this by running:
    ```bash
    pip install numpy pandas scikit-learn sentence-transformers faiss-cpu rank_bm25 langchain-community transformers torch
    ```
    (Refer to the second code cell in `BibleRAG.ipynb` for specific installations if needed).
4.  Place the `bible.txt` file (KJV Bible) in the specified path (e.g., `/kaggle/input/the-bible/bible.txt` as in the notebook, or adjust the path in the `TextLoader` accordingly).

## Running the Code

The main logic for the RAG pipeline and experimentation is contained within the `BibleRAG.ipynb` Jupyter Notebook.

1.  Open the notebook using Jupyter Lab or Jupyter Notebook.
2.  Ensure the path to the Bible corpus (`bible.txt`) is correctly set in Cell 3.
3.  Execute the cells sequentially. The notebook will:
    *   Load and process the data.
    *   Set up configurations for chunking, embedding, retrieval, and LLMs.
    *   Run the main experiment loop, iterating through questions, retrieving documents, generating answers, and evaluating them.
    *   Save the results to a CSV file (e.g., `qwen3B_chunk1024_Results.csv` for the specific run in the notebook).

## Results Summary (from Project Report)

The project report details extensive experiments. Key findings include:

*   **Best Overall Configuration:** The **Qwen2.5 3B model** paired with the **MMR retriever**, using a **chunk size of 256**, and **BAAI/bge-small-en embeddings** consistently demonstrated strong performance in terms of faithfulness and similarity, while maintaining good relevance.
*   **Embedding Model Performance:** The **BAAI/bge-small-en** embedding model consistently outperformed other tested embeddings (like MiniLM variants), especially for nuanced theological queries.
*   **Chunk Size Impact:** Smaller chunk sizes ( **256 or 512 characters**) were generally more effective. However, for questions about topics dispersed across various scriptures (e.g., the concept of "grace"), larger chunks (e.g., 1024) occasionally proved beneficial by providing broader context.
*   **Retrieval Method Effectiveness:** **Semantic Search** and **MMR** retrieval consistently outperformed **BM25** in faithfulness and relevance. The **Hybrid RRF** method offered robust and stable performance, often serving as a strong fallback when individual methods varied.
*   **Specific Question Insights:**
    *   **Forgiveness (Q1):** Qwen2.5 3B with MMR (chunk 256) achieved faithfulness > 0.80 and relevance > 0.82.
    *   **Greatest Commandment (Q2):** Chunk size 256 was ideal. BM25 showed surprising strength in similarity scores, likely due to lexical matching.
    *   **Wealth and Poverty (Q3):** Qwen2.5 3B with Semantic retrieval (chunk 256) achieved faithfulness > 0.86. MMR and Semantic methods excelled.
    *   **Concept of Grace (Q4):** Qwen2.5 1.5B with Semantic retrieval (chunk 512) achieved top faithfulness (0.88). Larger chunks were beneficial for this broader topic.

The detailed quantitative results for each configuration and question can be found in the project report and the CSV files generated by the notebook.

## File Structure

*   `BibleRAG.ipynb`: The main Jupyter Notebook containing the RAG implementation and experimentation logic.
*   `bible.txt`: (Not included in this repo, user must provide) The King James Version of the Bible text file.
*   `*.csv`: Output files containing the results of the experiments (e.g., `qwen3B_chunk1024_Results.csv`).
*   `Project_Report.pdf`: (If you include the PDF in the repo) The detailed project report.

## Authors

*   Zuhair Farhan (27100)
*   Sahil Kumar (27149)

## Acknowledgements

*   Ground truth answers were generated using a custom ChatGPT model, "Bible," by Dan An.
*   The project utilized Google Colab and Kaggle for GPU resources.