Retrieval-Augmented Generation (RAG) Chatbot

This repository contains a generic Retrieval-Augmented Generation (RAG) chatbot framework built around document-based question answering.

The system is designed to work with any PDF-based knowledge source.
By updating the documents in the data/ directory and rerunning the ingestion pipeline, the same system can be reused for different domains without changing application logic.

An Electric Vehicle (EV) document set is used as a sample use case, but the architecture itself is domain-agnostic.
