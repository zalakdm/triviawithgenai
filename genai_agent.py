import openai
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import time
from collections import Counter

with open("openaikey.txt") as f:
    openai.api_key = f.read().strip()

class OpenaiAgent:
    def __init__(self, debug_mode=False):
        # Debug mode flag to control verbose output
        self.debug_mode = debug_mode
        # Define few-shot examples based on the biology questions in the dataset
        self.few_shot_examples = [
            {
                "question": "GMOs are created by ________",
                "choices": [
                    "generating genomic DNA fragments with restriction endonucleases",
                    "introducing recombinant DNA into an organism by any means", 
                    "overexpressing proteins in E. coli",
                    "all of the above"
                ],
                "correct_answer": "introducing recombinant DNA into an organism by any means",
                "reasoning": "Let me think about this step by step:\n1. GMOs (Genetically Modified Organisms) are created by introducing foreign DNA into an organism\n2. This involves recombinant DNA technology\n3. The key is that any method of introducing recombinant DNA creates a GMO\n4. The other options are specific techniques, but the broad definition is 'introducing recombinant DNA by any means'\n5. Therefore, the answer is 'introducing recombinant DNA into an organism by any means'"
            },
            {
                "question": "What is a biomarker?",
                "choices": [
                    "the color coding of different genes",
                    "a protein that is uniquely produced in a diseased state",
                    "a molecule in the genome or proteome", 
                    "a marker that is genetically inherited"
                ],
                "correct_answer": "a protein that is uniquely produced in a diseased state",
                "reasoning": "Let me think about this step by step:\n1. A biomarker is a measurable indicator of a biological state or condition\n2. It's specifically used to detect diseases or medical conditions\n3. The key characteristic is that it's 'uniquely produced' in a diseased state\n4. While biomarkers can be molecules in genome/proteome, the most specific definition is a protein uniquely produced in disease\n5. Therefore, the answer is 'a protein that is uniquely produced in a diseased state'"
            },
            {
                "question": "Which scientific concept did Charles Darwin and Alfred Wallace independently discover?",
                "choices": [
                    "mutation",
                    "natural selection",
                    "overbreeding", 
                    "sexual reproduction"
                ],
                "correct_answer": "natural selection",
                "reasoning": "Let me think about this step by step:\n1. Charles Darwin and Alfred Wallace are both credited with the theory of evolution\n2. They independently arrived at the same conclusion about how species evolve\n3. The key concept they discovered was the mechanism of evolution\n4. This mechanism is called 'natural selection' - the process where organisms better adapted to their environment survive and reproduce\n5. Therefore, the answer is 'natural selection'"
            }
        ]
        
        # Initialize textbook chunks and embeddings
        self.textbook_chunks = []
        self.textbook_embeddings = []
        self._load_textbook_chunks()
    
    def _load_textbook_chunks(self):
        """
        Load and chunk the textbook, then create embeddings for RAG.
        """
        try:
            # Check if embeddings cache exists
            cache_file = "textbook_embeddings.json"
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    self.textbook_chunks = data['chunks']
                    self.textbook_embeddings = data['embeddings']
                print(f"Loaded {len(self.textbook_chunks)} textbook chunks from cache")
                print("Precomputing the bm25 stats from the textbook chunks")
                self._precompute_bm25_stats()
                return
            
            # Read and chunk the textbook
            print("Processing textbook for RAG...")
            with open("textbook.txt", 'r', encoding='utf-8') as f:
                textbook_content = f.read()
            
            print("Splitting textbook into chunks... ")
            # Split into chunks (paragraphs or sections)
            chunks = self._chunk_text(textbook_content)
            self.textbook_chunks = chunks
            
            print("Creating embeddings for textbook chunks...")
            # Create embeddings for each chunk
            self.textbook_embeddings = []
            
            # Process chunks in batches of size/8 for faster processing
            batch_size = len(chunks) // 8
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                print(f"Processing batch {i//batch_size}/8 ({len(batch)} chunks)...")
                
                start = time.time()
                batch_embeddings = self._get_embedding(batch)
                end = time.time()
                print(f"Batch {i//batch_size} embedding time: {end - start:.2f} seconds")
                
                self.textbook_embeddings.extend(batch_embeddings)
            print(f"Total embeddings created: {len(self.textbook_embeddings)}")   
            print("Caching embeddings for next run...")
            # Cache the embeddings
            with open(cache_file, 'w') as f:
                json.dump({
                    'chunks': self.textbook_chunks,
                    'embeddings': self.textbook_embeddings
                }, f)
            
            print(f"Created embeddings for {len(chunks)} textbook chunks")
            # Precompute BM25 statistics for all textbook chunks
            print("Precomputing the bm25 stats from the textbook chunks")
            self._precompute_bm25_stats()
        except Exception as e:
            print(f"Warning: Could not load textbook for RAG: {e}")
            self.textbook_chunks = []
            self.textbook_embeddings = []

    def _precompute_bm25_stats(self):
        """
        Precompute BM25 statistics for all textbook chunks for fast scoring.
        """
        self.doc_term_freqs = []  # List of Counter objects for each document
        self.doc_lengths = []     # List of document lengths
        self.term_doc_freq = Counter()  # term -> number of docs containing it
        for chunk in self.textbook_chunks:
            terms = chunk.lower().split()
            doc_tf = Counter(terms)
            self.doc_term_freqs.append(doc_tf)
            self.doc_lengths.append(len(terms))
            for term in set(terms):
                self.term_doc_freq[term] += 1
        self.avg_doc_length = np.mean(self.doc_lengths) if self.doc_lengths else 100
        self.total_docs = len(self.textbook_chunks)
        # Precompute IDF for all terms
        self.idf_cache = {}
        for term, df in self.term_doc_freq.items():
            self.idf_cache[term] = np.log((self.total_docs + 1) / (df + 1))

    def _chunk_text(self, text, chunk_size=1000, overlap=200):
        """
        Split text into overlapping chunks for better retrieval.
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks
    
    def _get_embedding(self, texts):
        """
        Get embedding for a text using OpenAI's embedding API.
        """
        try:
            if isinstance(texts, str):
                texts = [texts]

            response = openai.Embedding.create(
                model="text-embedding-3-small",
                input=texts
            )

            # Extract embeddings from response
            embeddings = []
            for item in response['data']:  # type: ignore
                if self._is_valid(item['embedding']):
                    embeddings.append(item['embedding'])
            
            return embeddings
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return []
    
    def _is_valid(self, vec):
        """
        Check if a vector is valid (not None, no NaN values, non-zero magnitude).
        Works with both lists and numpy arrays.
        """
        if vec is None:
            return False
        
        # Convert to numpy array if it's a list
        if isinstance(vec, list):
            vec = np.array(vec)
        
        # Check for NaN values
        if np.isnan(vec).any():
            return False
        
        # Check for zero magnitude
        if np.linalg.norm(vec) <= 0:
            return False
        
        return True

    #  using the hybrid approach:
    #  BM25 - keyword matching weightage : 30%
    #  sematic matching weighttage : 70%
    def _retrieve_relevant_chunks(self, question, top_k=3):
        """
        Retrieve the most relevant textbook chunks using hybrid search combining BM25 and embeddings.
        """

        if not self.textbook_chunks or not self.textbook_embeddings:
            return []
        
        # Get embedding for the question
        question_embeddings = self._get_embedding(question)
        if not question_embeddings:
            return []
        
        question_embedding = question_embeddings[0]
        
        # Calculate hybrid scores combining BM25 and embedding similarity
        hybrid_scores = []
        
        for i, chunk in enumerate(self.textbook_chunks):
            if i < len(self.textbook_embeddings) and self.textbook_embeddings[i]:
                # BM25 score (keyword matching)
                # startbm25_score = time.time()
                bm25_score = self._calculate_bm25_score(question, chunk)
                # endbm25_score = time.time()
                # print(f"bm25_score: {endbm25_score - startbm25_score:.2f} seconds")
                
                # Embedding similarity score (avoiding cosine similarity issues)
                # startembeddingscore = time.time()
                embedding_score = self._calculate_embedding_similarity(question_embedding, self.textbook_embeddings[i])
                # Combine scores with configurable weights
                bm25_weight = 0.3
                embedding_weight = 0.7
                hybrid_score = (bm25_weight * bm25_score) + (embedding_weight * embedding_score)
                hybrid_scores.append(hybrid_score)
            else:
                hybrid_scores.append(0)
        # Get top-k most relevant chunks
        top_indices = np.argsort(hybrid_scores)[-top_k:][::-1]
        relevant_chunks = []
        
        # Debug: Print retrieval results only in debug mode
        if self.debug_mode:
            print(f"\n=== RAG DEBUG ===")
            print(f"Question: {question}")
            print(f"Top {top_k} retrieved chunks:")
        
        for idx in top_indices:
            if self.debug_mode:
                score = hybrid_scores[idx]
                chunk_preview = self.textbook_chunks[idx][:100] + "..." if len(self.textbook_chunks[idx]) > 100 else self.textbook_chunks[idx]
                print(f"Score: {score:.4f} | Chunk: {chunk_preview}")
            
            relevant_chunks.append(self.textbook_chunks[idx])
        
        if self.debug_mode:
            print(f"Selected {len(relevant_chunks)} chunks")
            print(f"================\n")
        
        return relevant_chunks

    def _calculate_bm25_score(self, query, document):
        """
        Fast BM25 score calculation using precomputed statistics.
        
        BM25 is a ranking function that measures how well a document matches a query.
        It combines term frequency (TF) and inverse document frequency (IDF) with
        length normalization to provide accurate relevance scoring.
        
        Formula: BM25 = IDF × (TF × (k1 + 1)) / (TF + k1 × (1 - b + b × (doc_length / avg_doc_length)))
        
        Args:
            query (str): The search query/question
            document (str): The document/chunk to score
            
        Returns:
            float: BM25 relevance score (higher = more relevant)
        """
        # Check if precomputed statistics are available for fast calculation
        if not hasattr(self, 'idf_cache') or not self.idf_cache:
            # Fallback to original implementation if precomputed stats not available
            return 0
        
        # Preprocess query: convert to lowercase and split into individual terms
        query_terms = query.lower().split()
        
        # Try to find the document in our precomputed statistics for fast lookup
        try:
            doc_idx = self.textbook_chunks.index(document)
            # Use precomputed term frequencies and document length for speed
            doc_term_freq = self.doc_term_freqs[doc_idx]
            doc_length = self.doc_lengths[doc_idx]
        except ValueError:
            # If document not found in precomputed stats, calculate on-the-fly (slower)
            doc_terms = document.lower().split()
            doc_term_freq = Counter(doc_terms)  # Count frequency of each term
            doc_length = len(doc_terms)         # Total number of terms in document
        
        # BM25 parameters (standard values used in information retrieval)
        k1 = 1.2  # Term frequency saturation parameter - controls how much repeated terms matter
        b = 0.75  # Length normalization parameter - controls document length normalization
        
        # Initialize total BM25 score for this document
        score = 0
        
        # Calculate BM25 score for each term in the query
        for term in query_terms:
            # Only process terms that actually appear in the document
            if term in doc_term_freq:
                # Get Inverse Document Frequency (IDF) - measures how rare/important the term is
                idf = self.idf_cache.get(term, 0)
                
                # Get Term Frequency (TF) - how many times this term appears in the document
                tf = doc_term_freq[term]
                
                # Calculate BM25 numerator: TF × (k1 + 1)
                # This rewards documents with more occurrences of the term
                numerator = tf * (k1 + 1)
                
                # Calculate BM25 denominator: TF + k1 × (1 - b + b × (doc_length / avg_doc_length))
                # This normalizes for document length and applies term frequency saturation
                denominator = tf + k1 * (1 - b + b * (doc_length / self.avg_doc_length))
                
                # Add this term's contribution to the total BM25 score
                # Formula: score += IDF × (numerator / denominator)
                score += idf * (numerator / denominator)
        
        return score

    def _calculate_embedding_similarity(self, query_embedding, doc_embedding):
        """
        Calculate embedding similarity using dot product and normalization to avoid cosine similarity issues.
        
        This function computes the semantic similarity between two text embeddings using cosine similarity.
        It measures how close two pieces of text are in meaning, not just word overlap.
        
        Formula: Cosine Similarity = (A · B) / (||A|| × ||B||)
        Where A and B are the embedding vectors, · is dot product, and |||| is magnitude.
        
        Args:
            query_embedding (list): Numerical representation of the query text
            doc_embedding (list): Numerical representation of the document text
            
        Returns:
            float: Similarity score between -1.0 and 1.0 (higher = more similar)
        """
        try:
            # Convert embedding lists to numpy arrays for efficient computation
            # Using float32 for memory efficiency and sufficient precision
            query_vec = np.array(query_embedding, dtype=np.float32)
            doc_vec = np.array(doc_embedding, dtype=np.float32)
            
            # Validate that both vectors are meaningful (not empty, no NaN values, non-zero magnitude)
            # This prevents errors in subsequent calculations
            if not self._is_valid(query_vec) or not self._is_valid(doc_vec):
                return 0.0
            
            # Calculate dot product between the two vectors
            # Dot product measures how aligned the vectors are in high-dimensional space
            # Higher values indicate more similar semantic meaning
            dot_product = np.dot(query_vec, doc_vec)
            
            # Calculate the magnitude (length) of each vector using L2 norm
            # Magnitude is used for normalization in cosine similarity calculation
            query_magnitude = np.linalg.norm(query_vec)
            doc_magnitude = np.linalg.norm(doc_vec)
            
            # Prevent division by zero - if either vector has zero magnitude, return 0 similarity
            # This can happen with empty or invalid embeddings
            if query_magnitude == 0 or doc_magnitude == 0:
                return 0.0
            
            # Calculate cosine similarity: dot product divided by product of magnitudes
            # This normalizes the similarity score to be between -1 and 1
            # 1.0 = perfect semantic match, 0.0 = no relationship, -1.0 = opposite meaning
            similarity = dot_product / (query_magnitude * doc_magnitude)
            
            # Ensure the similarity score stays within the valid range [-1.0, 1.0]
            # This handles any numerical precision issues that might occur
            similarity = max(-1.0, min(1.0, similarity))
            
            return similarity
            
        except Exception as e:
            # Handle any numerical errors or other exceptions gracefully
            # Log the error for debugging but don't crash the system
            print(f"Warning: Error in embedding similarity calculation: {e}")
            return 0.0


    
    def _create_rag_prompt(self, question, answer_choices, relevant_chunks):
        """
        Creates a prompt with RAG (textbook context) and chain of thought reasoning.
        """
        # Start with system instruction
        system_prompt = """You are an expert biology tutor. Your task is to answer multiple choice questions by using the provided textbook information and thinking through the problem step-by-step.

Important instructions:
1. Use the textbook information provided to help answer the question
2. Read the question carefully and understand what is being asked
3. Think through the problem step-by-step, showing your reasoning
4. Consider each answer choice and eliminate obviously incorrect ones
5. Choose the answer that best matches the question based on the textbook information
6. End your response with 'ANSWER: [exact text of chosen answer]'
7. Make sure your final answer matches one of the given choices exactly"""

        # Build the few-shot examples with reasoning
        examples = ""
        for i, example in enumerate(self.few_shot_examples, 1):
            choices_text = "\n".join([f"{j+1}. {choice}" for j, choice in enumerate(example['choices'])])
            examples += f"""
Example {i}:
Question: {example['question']}
Choices:
{choices_text}

{example['reasoning']}

ANSWER: {example['correct_answer']}

"""

        # Format the current question with textbook context
        current_choices_text = "\n".join([f"{i+1}. {choice}" for i, choice in enumerate(answer_choices)])
        
        # Add textbook context if available
        textbook_context = ""
        if relevant_chunks:
            textbook_context = "\n\nRelevant textbook information:\n" + "\n\n".join(relevant_chunks[:2])  # Limit to 2 chunks
        
        # Put question first, then context, then examples for better focus
        user_prompt = f"""Question: {question}
Choices:
{current_choices_text}{textbook_context}

{examples}
Think through this step by step:"""

        return system_prompt, user_prompt

    def _create_chain_of_thought_prompt(self, question, answer_choices):
        """
        Creates a prompt with chain of thought reasoning using few-shot examples.
        """
        # Start with system instruction
        system_prompt = """You are an expert biology tutor. Your task is to answer multiple choice questions by thinking through the problem step-by-step and then selecting the most accurate answer.

Important instructions:
1. Read the question carefully and understand what is being asked
2. Think through the problem step-by-step, showing your reasoning
3. Consider each answer choice and eliminate obviously incorrect ones
4. Choose the answer that best matches the question
5. End your response with 'ANSWER: [exact text of chosen answer]'
6. Make sure your final answer matches one of the given choices exactly"""

        # Build the few-shot examples with reasoning
        examples = ""
        for i, example in enumerate(self.few_shot_examples, 1):
            choices_text = "\n".join([f"{j+1}. {choice}" for j, choice in enumerate(example['choices'])])
            examples += f"""
Example {i}:
Question: {example['question']}
Choices:
{choices_text}

{example['reasoning']}

ANSWER: {example['correct_answer']}

"""

        # Format the current question
        current_choices_text = "\n".join([f"{i+1}. {choice}" for i, choice in enumerate(answer_choices)])
        
        # Combine everything into the final prompt
        user_prompt = f"""{examples}
Now answer this question:
Question: {question}
Choices:
{current_choices_text}

Think through this step by step:"""

        return system_prompt, user_prompt

    def _create_simple_rag_prompt(self, question, answer_choices, relevant_chunks):
        """
        Creates a simple RAG prompt that puts the question first for better focus.
        """
        system_prompt = """You are an expert biology tutor. Answer the question using the provided textbook information if it's relevant, otherwise use your general knowledge. End your response with 'ANSWER: [exact text of chosen answer]'."""

        current_choices_text = "\n".join([f"{chr(65 + i)}. {choice}" for i, choice in enumerate(answer_choices)])
        
        # Use only 1 chunk and keep it short
        context = ""
        if relevant_chunks:
            context = f"\n\nTextbook information:\n{relevant_chunks[0][:500]}..." if len(relevant_chunks[0]) > 500 else relevant_chunks[0]
        
        user_prompt = f"""Question: {question}

Answer choices:
{current_choices_text}{context}

Answer with just the letter (A, B, C, or D):"""

        return system_prompt, user_prompt

    def _extract_answer_from_response(self, response_text, answer_choices):
        """
        Extracts the final answer from a chain of thought response.
        # Looks for 'ANSWER:' pattern and matches it to the choices.
        """
        # First, look for single letter answers (A, B, C, D)
        letter_pattern = r'\b([A-D])\b'
        letter_match = re.search(letter_pattern, response_text, re.IGNORECASE)
        if letter_match:
            letter = letter_match.group(1).upper()
            return ord(letter) - ord('A')  # Convert A=0, B=1, C=2, D=3
        # Look for the ANSWER: pattern
        answer_pattern = r'ANSWER:\s*(.+)'
        match = re.search(answer_pattern, response_text, re.IGNORECASE)
        
        if match:
            extracted_answer = match.group(1).strip()
            # Try to match the extracted answer to the choices
            for i, answer_choice in enumerate(answer_choices):
                if extracted_answer.lower() == answer_choice.lower():
                    return i
        
        # If no ANSWER: pattern found, try to match the last sentence or phrase
        # Split by lines and look for the last non-empty line
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]
        if lines:
            last_line = lines[-1]
            for i, answer_choice in enumerate(answer_choices):
                if last_line.lower() == answer_choice.lower():
                    return i
        
        # Fallback: try partial matching on the entire response
        for i, answer_choice in enumerate(answer_choices):
            if (answer_choice.lower() in response_text.lower() or 
                response_text.lower() in answer_choice.lower()):
                return i
        
        return -1

    def get_response(self, question, answer_choices):
        """
        Calls the OpenAI 3.5 API to generate a response using RAG and chain of thought reasoning.
        The response is then parsed to extract the final answer and matched to one of the answer choices.
        If the response does not match any answer choice, -1 is returned.

        Args:
            question: The question to be asked.
            answer_choices: A list of answer choices.

        Returns:
            The index of the answer choice that matches the response, or -1 if the response
            does not match any answer choice.
        """
        print("Fetching relevant textbook chunks...")
        # Try to retrieve relevant textbook chunks for RAG
        relevant_chunks = self._retrieve_relevant_chunks(question)
        
        if relevant_chunks:
            # Use simple RAG prompt for better performance
            print("Performing RAG...")
            system_prompt, user_prompt = self._create_simple_rag_prompt(question, answer_choices, relevant_chunks)
        else:
            # Fall back to regular chain of thought
            print("Performing shot + CoT")
            system_prompt, user_prompt = self._create_chain_of_thought_prompt(question, answer_choices)

        # Call the OpenAI 3.5 API with RAG and chain of thought prompting
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,  # Slightly higher temperature for more creative reasoning
            max_tokens=400,   # Allow more tokens for reasoning with textbook context
        )
        
        # Extract response text
        response_text = response['choices'][0]['message']['content'].strip()  # type: ignore

        # Extract the final answer from the chain of thought response
        return self._extract_answer_from_response(response_text, answer_choices)

    def debug_retrieval_strategies(self, question):
        """
        Debug different retrieval strategies to see which works best.
        """
        print(f"\n=== RETRIEVAL STRATEGY DEBUG ===")
        print(f"Question: {question}")
        
        # Test BM25 only
        bm25_scores = []
        for i, chunk in enumerate(self.textbook_chunks):
            score = self._calculate_bm25_score(question, chunk)
            bm25_scores.append(score)
        
        top_bm25 = np.argsort(bm25_scores)[-3:][::-1]
        print(f"\nTop 3 BM25 chunks:")
        for i, idx in enumerate(top_bm25):
            score = bm25_scores[idx]
            chunk_preview = self.textbook_chunks[idx][:100] + "..." if len(self.textbook_chunks[idx]) > 100 else self.textbook_chunks[idx]
            print(f"{i+1}. Score: {score:.4f} | {chunk_preview}")
        
        # Test embedding only
        question_embeddings = self._get_embedding(question)
        if question_embeddings:
            embedding_scores = []
            for i, embedding in enumerate(self.textbook_embeddings):
                if embedding:
                    score = self._calculate_embedding_similarity(question_embeddings[0], embedding)
                    embedding_scores.append(score)
                else:
                    embedding_scores.append(0)
            
            top_embedding = np.argsort(embedding_scores)[-3:][::-1]
            print(f"\nTop 3 Embedding chunks:")
            for i, idx in enumerate(top_embedding):
                score = embedding_scores[idx]
                chunk_preview = self.textbook_chunks[idx][:100] + "..." if len(self.textbook_chunks[idx]) > 100 else self.textbook_chunks[idx]
                print(f"{i+1}. Score: {score:.4f} | {chunk_preview}")
        
        print(f"================\n")
    def test_rag_vs_cot(self, question, answer_choices):
        """
        Test RAG vs CoT on the same question to see the difference.
        """
        print(f"\n=== RAG vs CoT TEST ===")
        print(f"Question: {question}")
        
        # Test RAG
        relevant_chunks = self._retrieve_relevant_chunks(question)
        if relevant_chunks:
            system_prompt, user_prompt = self._create_simple_rag_prompt(question, answer_choices, relevant_chunks)
        else:
            system_prompt, user_prompt = self._create_chain_of_thought_prompt(question, answer_choices)
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=200,
        )
        
        rag_response = response['choices'][0]['message']['content'].strip()  # type: ignore
        rag_answer = self._extract_answer_from_response(rag_response, answer_choices)
        
        print(f"RAG Response: {rag_response}")
        print(f"RAG Answer: {rag_answer}")
        
        # Test CoT only
        system_prompt, user_prompt = self._create_chain_of_thought_prompt(question, answer_choices)
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=200,
        )
        
        cot_response = response['choices'][0]['message']['content'].strip()  # type: ignore
        cot_answer = self._extract_answer_from_response(cot_response, answer_choices)
        
        print(f"CoT Response: {cot_response}")
        print(f"CoT Answer: {cot_answer}")
        print(f"================\n")
        
        return rag_answer, cot_answer

 