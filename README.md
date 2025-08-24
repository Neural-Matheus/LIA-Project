# STJ Súmulas — RAG (Embeddings + FAISS + LLM)

Pipeline para identificar qual Súmula do STJ corresponde a um trecho de texto, usando:

- Chunking "structure-first"
- Embeddings Sentence-Transformers (MPNet/MiniLM)
- FAISS Flat (cosine) com ranking por súmula (per-number)
- Geração com Groq (Llama 3.1) ou local (Transformers)

## Estrutura do projeto

```
LIA-Project-/
├─ ADRs/
│  ├─ 001.md … 008.md                               # decisões técnicas (ADR)
├─ src/
│  └─ data/
│     ├─ sumulas_stj_clean.json                     # dataset limpo (extraído do PDF)
│     ├─ sumulas_chunks_mpnet.jsonl                 # chunks gerados (MPNet tokenizer/384/64)
│     ├─ sumulas_chunks.jsonl                       # (variante opcional)
│     └─ faiss_per_number/                          # OUT_DIR (índices finais)
│        ├─ chunks.faiss
│        ├─ docstore.json
│        ├─ numbers.faiss                           # (opcional, centróides)
│        ├─ numbers_meta.json                       # (opcional)
│        ├─ X.npy                                   # (opcional, cache embeddings)
│        └─ manifest.json
├─ notebooks/
│  ├─ extract_sumu.ipynb                            # PDF → JSON
│  ├─ chunk_and_analysis_generate.ipynb             # chunking + análises (t-SNE, stats)
│  ├─ faiss_db.ipynb                                # embeddings + FAISS + avaliação
│  ├─ rag_model_local.ipynb                         # RAG com Transformers local
│  └─ rag_model_groq.ipynb                          # RAG com Groq (LLM remoto)
├─ README.md
├─ requirements.txt
└─ LICENSE
```

## Modelos usados

- **Embeddings (default)**: sentence-transformers/paraphrase-multilingual-mpnet-base-v2 (dim=768)
- **Alternativa rápida**: paraphrase-multilingual-MiniLM-L12-v2
- **LLM (Groq)**: llama-3.1-8b-instant (latência/$$) ou llama-3.1-70b (qualidade)

## Hiperparâmetros principais

### Chunking (ADR.002 v2)

- **Estratégia**: structure-first
- **Seções**: header, enunciado (inteiro), referencias_legislativas, orgao_data_fonte, excertos_precedentes
- **target_size**: 384 tokens | **overlap**: 64 | **max_len**: 512
- **max_windows_per_section**: 4
- **enunciado_mini** (≤128 tokens): opcional p/ boost de recall

### Embeddings

- **Encoder** = MPNet | **max_seq_length**=384 | **batch_size**=64 (GPU)
- **normalize_embeddings**=True (cosine via IP)

### FAISS + Ranking por Súmula

- **IndexFlatIP** com L2 normalize
- Busca chunks top_M (150–200) → agrega por número com pool="sum_sqrt"
- **Pesos por seção** (leve viés):
  - enunciado:1.40, enunciado_mini:1.30, referencias:1.05, excertos:1.00, orgao_data_fonte:0.90, header:0.85
- **top_numbers** (5–10) para a shortlist final

### Prompt (LLM)

- Contexto em blocos por súmula (enunciado + 1–3 evidências)
- **Caps recomendados**: enunciado_char_cap ≈ 800, evidence_char_cap ≈ 600

## Sequência de replicação (passo a passo)

### 0) Ambiente

```bash
python -m venv .venv && source .venv/bin/activate  # (opcional)
pip install -r requirements.txt
```

### 1) PDF → JSON limpo

- Abra `notebooks/extract_sumu.ipynb`
- **Saída esperada**: `src/data/sumulas_stj_clean.json`

### 2) Chunking + análises

- Abra `notebooks/chunk_and_analysis_generate.ipynb`
- Use tokenizer do encoder escolhido; política 384/64
- **Saída esperada**: `src/data/sumulas_chunks_mpnet.jsonl` (ou `sumulas_chunks.jsonl`)

### 3) Embeddings + FAISS + avaliação

- Abra `notebooks/faiss_db.ipynb`
- Gere o OUT_DIR: `src/data/faiss_per_number/` com:
  - `chunks.faiss`, `docstore.json`
  - `numbers.faiss`, `numbers_meta.json` (opcional, centróides p/ 2 estágios)
  - `manifest.json` (+ `X.npy` opcional)
- Cheque métricas de recall@k e t-SNE

### 4) RAG

- **Local**: `notebooks/rag_model_local.ipynb`
- **Groq**: `notebooks/rag_model_groq.ipynb`
  - Set `GROQ_API_KEY`
  - Aponte `OUT_DIR = "src/data/faiss_per_number"`


## Resultados (dev, principais)

- **Self-match** (enunciado_mini → corpus completo): Recall@1/3/5/10 = 1.00/1.00/1.00/1.00
- **Per-number** (enunciado_mini → corpus sem enunciado): ≈ 0.244 / 0.382 / 0.439 / 0.558
- **Per-number** (enunciado_mini → header/refs/excertos/orgao): ≈ 0.327 / 0.441 / 0.490 / 0.538

## Como rodar um teste rápido (Groq) usando OUT_DIR do projeto

Dentro de um notebook (ou script) no repo:

```python
import os, json
from rag_core import load_retriever, gather_hits_for_query, generate_json_answer

os.environ["GROQ_API_KEY"] = "SEU_TOKEN_AQUI"  # necessário p/ Groq

OUT_DIR = "src/data/faiss_per_number"
retriever = load_retriever(
    out_dir=OUT_DIR,
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    max_seq_length=384, batch_size=64, device=None
)

query = "Conversão de prisão em flagrante em preventiva de ofício após a Lei 13.964/2019 é possível?"

hits = gather_hits_for_query(
    query, retriever,
    top_numbers=8, top_chunks_per_query=160,
    max_evidence_per_number=3, enunciado_char_cap=800, evidence_char_cap=600
)

ans = generate_json_answer(
    query, hits,
    model="llama-3.1-8b-instant",
    temperature=0.0, max_completion_tokens=400
)

print(json.dumps(ans, ensure_ascii=False, indent=2))
```
## Observabilidade & controles

- **Caps** (enunciado_char_cap, evidence_char_cap) para não estourar janelas/TPM da API
- **top_numbers**, **top_chunks_per_query** e pesos por seção calibráveis
- **manifest.json** guarda o encoder usado no índice — mantenha queries com o mesmo modelo
