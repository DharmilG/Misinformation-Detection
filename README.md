# Misinformation-Detection
This project detects potentially misleading image-caption pairs using pretrained CLIP and a classification head, then retrieves relevant evidence from trusted news and fact-checking sources using RAG to generate an evidence-grounded explanation.


# Folder strcuture

```text
multimodal-misinformation-rag/
    app.py -> Main Streamlit application entry point
    requirements.txt -> Python dependencies
    
    src/ -> Application source code
        /input_pipeline/ -> Image upload, validation and text preprocessing
        /clip_model/ -> CLIP features, classifier inference and model loading
        /retrieval/ -> Evidence ingestion, chunking, embeddings and search
        /rag/ -> LLM prompts, evidence-grounded assessment and report generation
        /reporting/ -> Final assessment formatting and source citations
        /utils/ -> Shared configuration, logging and helper functions
    
    tests/ -> Unit, integration and end-to-end tests
    
    data/ -> Dataset manifests, curated evidence and local data
    
    models/ -> Saved classifier weights and other model artifacts
    
    notebooks/ -> Exploratory data analysis and experiments
    
    scripts/ ->Dataset preparation, indexing and training commands
```