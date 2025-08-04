import torch
import numpy as np
import os
import argparse
import logging
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from rank_bm25 import BM25Okapi
import re

# Set up logging to track training progress
logging.basicConfig(filename="training.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Function to load and preprocess text files
def load_file(file_path):
    """Load and preprocess text from a file, ensuring it exists and is non-empty."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
    with open(file_path, 'r', encoding='utf-8') as f:
        # Remove empty lines and preprocess text (lowercase, remove punctuation)
        lines = [re.sub(r'[^\w\s]', '', line.strip().lower()) for line in f.readlines() if line.strip()]
    return lines

# Custom dataset class for legal queries, documents, and answers
class LegalDataset(Dataset):
    def __init__(self, queries, documents, answers, tokenizer, top_k=5):
        """Initialize dataset with queries, documents, answers, and BM25 for retrieval."""
        self.queries = queries
        self.documents = documents
        self.answers = answers
        if len(queries) != len(answers):
            raise ValueError("Number of queries and answers must match.")
        self.tokenizer = tokenizer
        self.top_k = top_k
        # Initialize BM25 for document retrieval
        tokenized_docs = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        """Retrieve query, top-k documents, and answer, and tokenize them."""
        query = self.queries[idx]
        answer = self.answers[idx]
        # Retrieve top-k relevant documents using BM25
        tokenized_query = query.split()
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_doc_indices = np.argsort(doc_scores)[::-1][:self.top_k]
        top_docs = [self.documents[i] for i in top_doc_indices]
        document = " ".join(top_docs)  # Combine top documents into a single context
        input_text = f"<context>{document}</context><query>{query}</query>"
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length"
        )
        labels = self.tokenizer(
            answer,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding="max_length"
        )["input_ids"]
        return {
            "input_ids": inputs["input_ids"].flatten(),
            "attention_mask": inputs["attention_mask"].flatten(),
            "labels": labels.flatten()
        }

# Parse command-line arguments for file paths
def parse_args():
    """Parse command-line arguments for file paths."""
    parser = argparse.ArgumentParser(description="RAG pipeline for legal text processing")
    parser.add_argument("--queries_file", default="queries.txt", help="Path to queries file")
    parser.add_argument("--documents_file", default="documents.txt", help="Path to documents file")
    parser.add_argument("--answers_file", default="answers.txt", help="Path to answers file")
    parser.add_argument("--model_dir", default="./model_checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    return parser.parse_args()

# Main function to run the pipeline
def main():
    args = parse_args()

    # Load and preprocess data
    queries = load_file(args.queries_file)
    documents = load_file(args.documents_file)
    answers = load_file(args.answers_file)

    # Split data into train and validation sets
    train_queries, val_queries, train_docs, val_docs, train_answers, val_answers = train_test_split(
        queries, documents, answers, test_size=0.2, random_state=42
    )

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create datasets and data loaders
    train_dataset = LegalDataset(train_queries, train_docs, train_answers, tokenizer)
    val_dataset = LegalDataset(val_queries, val_docs, val_answers, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    # Define optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=500,
        num_training_steps=len(train_loader) * args.epochs
    )
    scaler = GradScaler()

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_train_loss += loss.item()

        # Validation step
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                with autocast():
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    total_val_loss += outputs.loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        logging.info(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save checkpoint if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(args.model_dir, exist_ok=True)
            model.save_pretrained(f"{args.model_dir}/checkpoint_epoch_{epoch+1}")
            tokenizer.save_pretrained(f"{args.model_dir}/checkpoint_epoch_{epoch+1}")
            logging.info(f"Saved checkpoint for epoch {epoch+1}")

    # Evaluation with metrics
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    model.eval()
    with torch.no_grad():
        for query, document, answer in zip(val_queries, val_docs, val_answers):
            tokenized_query = query.split()
            doc_scores = train_dataset.bm25.get_scores(tokenized_query)
            top_doc_indices = np.argsort(doc_scores)[::-1][:train_dataset.top_k]
            top_docs = [train_docs[i] for i in top_doc_indices]
            document = " ".join(top_docs)
            input_text = f"<context>{document}</context><query>{query}</query>"
            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            generated_text = model.generate(
                **inputs,
                max_length=128,
                num_beams=5,
                early_stopping=True
            )
            generated_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)
            rouge_scores = scorer.score(answer, generated_text)
            bleu_score = sentence_bleu([answer.split()], generated_text.split())
            print(f"Query: {query}\nGenerated: {generated_text}\nROUGE-1: {rouge_scores['rouge1'].fmeasure:.4f}, BLEU: {bleu_score:.4f}")
            logging.info(f"Query: {query}\nGenerated: {generated_text}\nROUGE-1: {rouge_scores['rouge1'].fmeasure:.4f}, BLEU: {bleu_score:.4f}")

if __name__ == "__main__":
    main()