## Retrieval-Augmented Generation Unified (RAGU)

Retrieval-Augmented Generation Unified (RAGU) is an advanced AI architecture that enhances Large Language Models (LLMs) by systematically integrating them with diverse retrieval mechanisms 
to provide more accurate, reliable, and contextually grounded responses.
Retrieval-Augmented Generation (RAG) is an approach where an LLM is augmented with external data retrieved dynamically from various authoritative knowledge sources such as internal databases,
documents, APIs, or the web. Instead of solely relying on the static knowledge embedded during training, RAG allows the model to query and incorporate up to date and domain-specific information 
at inference time for improved accuracy.

# Features
	- Dynamic Retrieval: Uses BM25 to retrieve the top-k relevant documents for each query.
	- Fine-Tuning: Fine-tunes a google/flan-t5-base model on legal query-document-answer triples.
	- Evaluation: Computes ROUGE-1, ROUGE-L, and BLEU scores for generated answers.
	- Robustness: Includes error handling, text preprocessing, and mixed precision training.
	- Checkpointing: Saves model checkpoints when validation loss improves.
	- Logging: Logs training and evaluation results to a file (training.log).

# Input File Format
The pipeline requires three text files:
	queries.txt: One legal query per line.
	documents.txt: One legal document per line.
	answers.txt: One ground-truth answer per line, corresponding to each query.

# Error Handling and Preprocessing:
Added a load_file function to check for file existence and use UTF-8 encoding.
Preprocessed text by converting to lowercase and removing punctuation using regex.
Validated that the number of queries and answers match in the dataset.

# Dynamic Document Retrieval:
Integrated BM25 for retrieving the top-k relevant documents per query, replacing the one-to-one query-document assumption.
Combined multiple documents into a single context for RAG input.

# Proper Labels:
Added support for a separate answers.txt file containing ground-truth answers for fine-tuning.
Tokenized answers as labels instead of using input IDs, aligning with standard seq2seq training.

# Model and Tokenizer:
Switched to google/flan-t5-base, which is better suited for instruction-based tasks and likely performs better for legal text than t5-base.
Added padding (padding="max_length") to ensure consistent input sizes for batching.

# Training Enhancements:
Split the dataset into training and validation sets (80-20 split) for better generalization.
Added a learning rate scheduler (get_linear_schedule_with_warmup) to improve convergence.
Implemented mixed precision training with torch.cuda.amp to reduce memory usage and speed up training.
Added gradient clipping to stabilize training.
Included early stopping by saving checkpoints when validation loss improves.


# Evaluation Metrics:

Added ROUGE-1, ROUGE-L, and BLEU scores to evaluate generated text against ground truth answers.
Used beam search (num_beams=5) in model.generate for better output quality.


# Model Saving and Checkpointing:

Saved the model and tokenizer to a specified directory when validation loss improves.
Created a directory for checkpoints to avoid overwriting.


# Logging and Configurability:

Logging to save training and evaluation results to training.log.
Used argparse to make file paths and epochs configurable via command-line arguments.



# How to Use the Code
Prepare Input Files:

Create queries.txt, documents.txt, and answers.txt with corresponding queries, documents, and ground-truth answers.
Example format:

queries.txt: One legal query per line (e.g., "What is the statute of limitations for breach of contract?").
documents.txt: One legal document per line (e.g., "The statute of limitations for breach of contract in California is four years...").
answers.txt: One answer per line (e.g., "The statute of limitations for breach of contract is four years in California.").

Install Dependencies:

		pip install -r requirements.txt


Run the pipeline with:

		python main_ragu.py --queries_file queries.txt --documents_file documents.txt --answers_file answers.txt --model_dir ./model_checkpoints --epochs 5





Output:

Training and validation losses will be printed and logged to training.log.
Checkpoints will be saved in ./model_checkpoints/checkpoint_epoch_X.
Evaluation results (ROUGE and BLEU scores) will be printed for each validation query.


# How does RAGU work?
	1. User Query: A user inputs a question or prompt.
	2. Encoding & Retrieval: The system encodes the query into embeddings and simultaneously applies unified retrieval methods to gather relevant information from various knowledge bases.
	3. Context Enrichment: Retrieved data is consolidated and transformed into enriched context.
	4. Generation: The LLM receives the original query augmented with retrieved information and generates a precise, informative response.
	5. Post-processing: Optional refinement such as summarization or validation may be applied before delivering the response.

# Benefits of RAGU
	1. Enhances the knowledge scope of LLMs with fresh, specific data.
	2. Bridges knowledge gaps without costly model retraining.
	3. Supports varied retrieval styles for more complete understanding.
	4. Improves response relevance, accuracy, and trustworthiness in real time applications.

# License
This project is licensed under the MIT License

