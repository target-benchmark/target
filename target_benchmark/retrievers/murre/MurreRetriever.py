import json
import os
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Any, Optional
from scipy.spatial.distance import cosine
from dataclasses import dataclass
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from sentence_transformers import SentenceTransformer

from target_benchmark.retrievers import AbsCustomEmbeddingRetriever

cur_dir_path = Path(__file__).parent.resolve()
data_path = cur_dir_path / "data/"

@dataclass
class BeamState:
    # single beam state
    retrieved_tables: List[int]
    current_query: str
    score: float
    hop: int
    
    def __post_init__(self):
        self.retrieved_tables = list(self.retrieved_tables) if self.retrieved_tables else []


class MurreRetriever(AbsCustomEmbeddingRetriever):
    def __init__(self, 
                 model: str = "text-embedding-3-small",
                 embedding_type: str = "openai",
                 use_rewriting: bool = False,
                 beam_size: int = 3,
                 max_hops: int = 2,
                 rewriter_model: str = "gpt-4o-mini"):
        """
        Initialize MurreRetriever

        Args:
            model: Embedding model name
            embedding_type: "openai", "custom", or "sentence-transformers"
            use_rewriting: Use query rewriting
            beam_size: Number of beams
            max_hops: Maximum number of hops
            rewriter_model: LLM for query rewriting
        """
        super().__init__(expected_corpus_format="nested array")
        
        self.model = model
        self.embedding_type = embedding_type
        self.use_rewriting = use_rewriting
        self.beam_size = beam_size
        self.max_hops = max_hops
        self.rewriter_model = rewriter_model
        self.embedding_client = None
        self.rewriter_client = None
        self.tokenizer = None
        self.embedding_model = None
        self.corpus_embeddings = {}

    def retrieve(self, query: str, dataset_name: str, top_k: int, **kwargs) -> List[Tuple[str, str]]:
        """
        Retrieve relevant tables using multi-hop beam search
        
        Args:
            query: User query
            dataset_name: Name of the dataset
            top_k: Number of tables
            
        Returns:
            List of (database_id, table_id) tuples
        """
        if dataset_name not in self.corpus_embeddings:
            dataset_persistence_path = data_path / dataset_name
            if not dataset_persistence_path.exists():
                raise ValueError(f"Need to embed corpus first!")
            self._load_embeddings(dataset_name)
        
        try:
            if self.beam_size > 1:
                return self._beam_search_retrieval(query, dataset_name, top_k)
            else:
                return self._single_hop_retrieval(query, dataset_name, top_k)
        except Exception as e:
            print(f"Error: {e}")
            return []

    def embed_corpus(self, dataset_name: str, corpus: Iterable[Dict[str, Any]]) -> None:
        self._ensure_embedding_client()
        
        dataset_persistence_path = data_path / dataset_name
        dataset_persistence_path.mkdir(parents=True, exist_ok=True)
        
        all_table_texts = []
        all_table_ids = []
        
        for batch_dict in corpus:
            batch_db_ids = batch_dict.get("database_id", [])
            batch_table_names = batch_dict.get("table_id", [])
            batch_table_contents = batch_dict.get("table", [])

            if not (len(batch_db_ids) == len(batch_table_names) == len(batch_table_contents)):
                print(f"Warning: Mismatch in lengths of batch lists for dataset {dataset_name}. Skipping batch.")
                continue

            for i in range(len(batch_db_ids)):
                db_id = str(batch_db_ids[i])
                table_name = str(batch_table_names[i])
                table_content = batch_table_contents[i]

                if table_content is None:
                    print(f"Warning: Missing 'table' content for {db_id}.{table_name} in dataset {dataset_name}. Skipping.")
                    continue
                
                table_str = self._convert_table_to_string(table_content)
                all_table_texts.append(table_str)
                all_table_ids.append((db_id, table_name))

        if not all_table_texts:
            print(f"Warning: No tables found or processed for dataset {dataset_name}.")
            return

        all_embeddings = self._get_embeddings_batch(all_table_texts, is_query=False)
        
        embedding_data = {
            "ids": all_table_ids,
            "embeddings": all_embeddings,
            "table_strings": all_table_texts,
            "config": {
                "model": self.model,
                "embedding_type": self.embedding_type
            }
        }
        
        embeddings_path = dataset_persistence_path / "embeddings.json"
        with open(embeddings_path, 'w') as f:
            json.dump(embedding_data, f)
        
        self.corpus_embeddings[dataset_name] = embedding_data
        

    def _load_embeddings(self, dataset_name: str) -> None:
        dataset_persistence_path = data_path / dataset_name
        embeddings_path = dataset_persistence_path / "embeddings.json"
        
        if not embeddings_path.exists():
            raise ValueError(f"No embeddings file found for dataset '{dataset_name}'")
        
        with open(embeddings_path, 'r') as f:
            self.corpus_embeddings[dataset_name] = json.load(f)

    def _ensure_embedding_client(self) -> None:
        if self.embedding_client is not None:
            return
            
        # TODO: if possible, add support for other embedding types
        if self.embedding_type == "openai":
            self._init_openai_embeddings()
        elif self.embedding_type == "sentence-transformers":
            self._init_sentence_transformer_embeddings()
        else:
            self._init_custom_embeddings()

    def _ensure_rewriter_client(self) -> None:
        if not self.use_rewriting or self.rewriter_client is not None:
            return
            
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found! Please set OPENAI_API_KEY environment variable.")
        
        self.rewriter_client = ChatOpenAI(
            model=self.rewriter_model,
            openai_api_key=api_key,
            temperature=0.1,
            max_tokens=150
        )

    def _init_openai_embeddings(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found! Please set OPENAI_API_KEY environment variable.")
        
        self.embedding_client = OpenAIEmbeddings(
            model=self.model,
            openai_api_key=api_key,
            chunk_size=1
        )

    def _init_sentence_transformer_embeddings(self) -> None:
        self.embedding_client = SentenceTransformer(self.model)

    def _init_custom_embeddings(self) -> None:
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model)
            self.embedding_model = AutoModel.from_pretrained(self.model)
            self.embedding_model.eval()
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.embedding_model.to(self.device)
            
            # handle SGPT models
            self.use_specb = "sgpt" in self.model.lower()
            if self.use_specb:
                self.SPECB_QUE_BOS = self.tokenizer.encode("[", add_special_tokens=False)[0]
                self.SPECB_QUE_EOS = self.tokenizer.encode("]", add_special_tokens=False)[0]
                self.SPECB_DOC_BOS = self.tokenizer.encode("{", add_special_tokens=False)[0]
                self.SPECB_DOC_EOS = self.tokenizer.encode("}", add_special_tokens=False)[0]
                
        except Exception as e:
            raise ValueError(f"Failed to load custom embedding model: {e}")

    def _get_embeddings_batch(self, texts: List[str], is_query: bool = True) -> List[List[float]]:
        if self.embedding_type == "openai":
            all_embeddings = []
            # TODO: add support for other batch sizes in __init__
            batch_size = 256  
            for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
                batch_texts = texts[i:i + batch_size]
            if is_query:
                    all_embeddings.extend([self.embedding_client.embed_query(text) for text in batch_texts])
            else:
                    all_embeddings.extend(self.embedding_client.embed_documents(batch_texts))
            return all_embeddings
        elif self.embedding_type == "sentence-transformers":
            embeddings = self.embedding_client.encode(texts, show_progress_bar=True)
            return embeddings.tolist()
        else:
            return self._get_custom_embeddings_batch(texts, is_query)

    def _get_custom_embeddings_batch(self, texts: List[str], is_query: bool) -> List[List[float]]:
        try:
            if self.use_specb:
                tokens = self._tokenize_with_specb(texts, is_query)
            else:
                tokens = self.tokenizer(texts, padding=True, truncation=True, 
                                      return_tensors="pt", max_length=512).to(self.device)
            
            with torch.no_grad():
                outputs = self.embedding_model(**tokens)
                embeddings = outputs.last_hidden_state.mean(dim=1)
            
            return [emb.cpu().numpy().tolist() for emb in embeddings]
        except Exception as e:
            print(f"Error getting custom embeddings: {e}")
            return [[] for _ in texts]

    def _tokenize_with_specb(self, texts: List[str], is_query: bool) -> Dict[str, torch.Tensor]:
        """Tokenize with SPECB tokens for SGPT models"""
        batch_tokens = self.tokenizer(texts, padding=False, truncation=True, max_length=512)
        
        for seq, att in zip(batch_tokens["input_ids"], batch_tokens["attention_mask"]):
            if is_query:
                seq.insert(0, self.SPECB_QUE_BOS)
                seq.append(self.SPECB_QUE_EOS)
            else:
                seq.insert(0, self.SPECB_DOC_BOS)
                seq.append(self.SPECB_DOC_EOS)
            att.insert(0, 1)
            att.append(1)
        
        return self.tokenizer.pad(batch_tokens, padding=True, return_tensors="pt").to(self.device)

    def _convert_table_to_string(self, table_data: Any) -> str:
        """Convert table data to string representation"""
        if not table_data or not isinstance(table_data, list):
            return str(table_data)

        string_rows = []
        for i, row_items in enumerate(table_data):
            if i == 0:
                string_rows.append("Columns: " + " | ".join(map(str, row_items)))
            else:
                string_rows.append("Data: " + " | ".join(map(str, row_items)))
        
        return " \\n ".join(string_rows)

    def _beam_search_retrieval(self, query: str, dataset_name: str, top_k: int) -> List[Tuple[str, str]]:
        """Multi-hop beam search retrieval"""
        stored_corpus = self.corpus_embeddings[dataset_name]
        table_ids = stored_corpus["ids"]
        corpus_embeddings = stored_corpus["embeddings"]
        table_strings = stored_corpus["table_strings"]

        if not table_ids or not corpus_embeddings:
            return []

        # init beam search
        beams = [BeamState([], query, 0.0, 0)]
        
        for hop in range(self.max_hops):
            new_beams = []
            
            for beam in beams:
                if not beam.current_query.strip():
                    new_beams.append(beam)
                    continue
                
                query_embeddings = self._get_embeddings_batch([beam.current_query], is_query=True)
                if not query_embeddings or not query_embeddings[0]:
                    new_beams.append(beam)
                    continue
                
                query_embedding = query_embeddings[0]
                
                similarities = []
                for idx, doc_embedding in enumerate(corpus_embeddings):
                    if idx not in beam.retrieved_tables and doc_embedding:
                        sim = 1 - cosine(query_embedding, doc_embedding)
                        similarities.append((idx, sim))
                
                if not similarities:
                    new_beams.append(beam)
                    continue
                
                similarities.sort(key=lambda x: x[1], reverse=True)
                num_candidates = min(3, len(similarities))
                
                for table_idx, sim_score in similarities[:num_candidates]:
                    new_retrieved = beam.retrieved_tables + [table_idx]
                    beam_score = beam.score + sim_score
                    
                    rewritten_query = beam.current_query
                    if self.use_rewriting:
                        self._ensure_rewriter_client()
                        rewritten_query = self._rewrite_query(
                            beam.current_query, 
                            [table_strings[table_idx]]
                        )
                    
                    new_beams.append(BeamState(
                        new_retrieved, rewritten_query, beam_score, hop + 1
                    ))
            
            # prune beams
            if new_beams:
                new_beams.sort(key=lambda x: x.score, reverse=True)
                beams = new_beams[:self.beam_size]
            else:
                break
        
        all_retrieved = set()
        for beam in beams:
            all_retrieved.update(beam.retrieved_tables)
        
        # return top_k 
        table_scores = {}
        for beam in beams:
            for idx in beam.retrieved_tables:
                if idx not in table_scores or beam.score > table_scores[idx]:
                    table_scores[idx] = beam.score
        
        sorted_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for idx, _ in sorted_tables[:top_k]:
            if idx < len(table_ids):
                results.append(table_ids[idx])
        
        return results

    def _single_hop_retrieval(self, query: str, dataset_name: str, top_k: int) -> List[Tuple[str, str]]:
        """Single-hop retrieval fallback"""
        stored_corpus = self.corpus_embeddings[dataset_name]
        table_ids = stored_corpus["ids"]
        corpus_embeddings = stored_corpus["embeddings"]

        query_embeddings = self._get_embeddings_batch([query], is_query=True)
        if not query_embeddings or not query_embeddings[0]:
            return []
        
        query_embedding = query_embeddings[0]
        
        similarities = []
        for doc_embedding in corpus_embeddings:
            if doc_embedding:
                sim = 1 - cosine(query_embedding, doc_embedding)
                similarities.append(sim)
            else:
                similarities.append(0.0)
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [table_ids[i] for i in top_indices]

    def _rewrite_query(self, query: str, retrieved_table_strings: List[str]) -> str:
        """Rewrite query by removing information about retrieved tables"""
        try:
            retrieved_info = "\n".join(retrieved_table_strings)
            
            # TODO: perhaps extract this to config file
            prompt = f"""Given a user question and previously retrieved tables, rewrite the question by REMOVING information that is already covered by the retrieved tables.

            Previously Retrieved Tables:
            {retrieved_info}

            Original Question: {query}

            Task: Rewrite the question to focus on information NOT covered by the retrieved tables. If all necessary information is already covered, respond with "None".

            Rewritten Question (or "None"):"""
            
            messages = [
                ("system", "You are a helpful assistant for multi-hop table retrieval."),
                ("user", prompt)
            ]
            
            response = self.rewriter_client.invoke(messages)
            rewritten_query = response.content.strip()
            
            # check for early stopping
            if any(signal in rewritten_query.lower() for signal in ["none", "no additional"]):
                return ""
            
            return rewritten_query
            
        except Exception as e:
            print(f"Warning: Query rewriting failed: {e}")
            return query 