import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
INDEX = DATA / "index"
SOURCES = DATA / "sources.json"
ALIASES = DATA / "aliases.json"
GAPS = DATA / "gaps.jsonl"
FEEDBACK = DATA / "feedback.jsonl"
PROMPTS = ROOT / "prompts"
EVAL_DIR = ROOT / "eval"

MODEL = os.getenv("MHA_MODEL", "claude-sonnet-5-5")
# bge-m3: 100+ хэл дэмждэг тул Монгол, англи, орос, хятад асуулт нэг орон зайд хайгдана.
EMBED_MODEL = os.getenv("MHA_EMBED_MODEL", "BAAI/bge-m3")
# Cross-encoder reranker. Хоосон string ("") бол унтраана.
RERANK_MODEL = os.getenv("MHA_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
PROMPT = os.getenv("MHA_PROMPT", "v2")
