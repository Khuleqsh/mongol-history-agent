# 📜 mongol-history-agent

**Монголын түүхийн архивт олон хэлээр хайж, ишлэлтэйгээр, цаг хугацааны дарааллаар ойлгомжтой хариулдаг agentic AI.**
**A multilingual, citation-first AI research agent for Mongolian history.**

![License: MIT](https://img.shields.io/badge/license-MIT-blue)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![Tests](https://img.shields.io/badge/tests-52%20passing-brightgreen)

[🇲🇳 Монгол](#-монгол) · [🇬🇧 English](#-english)

---

## 🇲🇳 Монгол

### Энэ юу вэ?

Та өөрийн цуглуулсан түүхийн эх сурвалжуудыг (шастир, судалгаа, нийтлэл) оруулж өгнө. Агент тэдгээрээс хайж, **зөвхөн олдсон текстэд тулгуурлан** хариулж, өгүүлбэр бүрийг ишлэлээр баталгаажуулна. Олдохгүй бол "мэдэхгүй" гэж хэлнэ.

### Онцлог

- 🔎 **Олон хэлээр хайлт.** Монгол (Кирилл ба Латин), англи, орос, хятад асуулт нэг орон зайд хайгдана (`bge-m3`). Нэрийн хувилбарыг автоматаар өргөжүүлнэ: Тэмүжин ↔ Temüjin ↔ 铁木真.
- 📌 **Ишлэл заавал.** Хариултын өгүүлбэр бүр `[S1]` гэх мэт эх сурвалжид холбогдоно. Олдоогүй ишлэлийг илрүүлнэ.
- ⚖️ **Эх сурвалжийн зөрүүг зэрэгцүүлнэ.** "А сурвалж ингэж, Б сурвалж ингэж бичсэн" гэж харуулна.
- 📅 **Цаг хугацааны дараалал.** Хариулт нь товч хариулт, дарааллаар тайлбар, зөрүү, итгэлийн түвшин, дараагийн асуултуудтай. Оноор шүүж хайх боломжтой.
- 🎓 **Тайлбарын түвшин.** Товч, энгийн, сурагчид зориулсан, судлаачид зориулсан.
- ✅ **Давхар шалгалт (заавал биш).** Тусдаа дуудлагаар батлагдаагүй мэдэгдлийг олж засна.
- 🔁 **Өөрийгөө сайжруулах мөчлөг.** Архивт дутсан сэдвийг бүртгэнэ, 👎 санал нь шалгах асуулт болно, eval оноо буурвал илрүүлнэ, prompt-ийн хувилбаруудыг харьцуулна.
- 💸 **Үнэгүй эхэлж болно.** Анхдагчаар Google Gemini API (AI Studio-ийн free tier). Embedding ба хайлт нь өөрийн машинд ажиллана. `MHA_PROVIDER=anthropic`-оор Claude руу шилжиж болно.
- 🖥️ **Вэб апп ба CLI.** Эх сурвалжаа апп дотроос файлаар нэмж болно.

### Хэрхэн ажилладаг

```
Асуулт ─► Агент (LLM + tool use) ─► Хайлт ─► Ишлэлтэй хариулт ─► (давхар шалгалт)
                                      │
   нэрийн хувилбар → vector + BM25 → RRF → оны шүүлт → (reranker) → эх сурвалжаар тэнцвэржүүлэлт
```

Дэлгэрэнгүйг [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)-аас үзнэ үү.

### Хурдан эхлэх (компьютергүй, зөвхөн GitHub-аар)

1. **Түлхүүр авна.** [aistudio.google.com](https://aistudio.google.com) → **Get API key** → **Create API key**. Карт шаардахгүй.
2. **Түлхүүрээ хадгална.** GitHub → Settings → **Codespaces** → **New secret**. Нэр: `GEMINI_API_KEY`. Утга: түлхүүр. Repository access дээр энэ repo-г сонгоно.
3. **Codespace нээнэ.** Repo дээр **Code → Codespaces → Create codespace on main**. Эхний удаа хэдэн минут суулгана, дараа нь вэб апп автоматаар нээгдэнэ.
4. **Эх сурвалж нэмнэ.** Апп-ын хажуугийн самбарын **➕ Эх сурвалж нэмэх** хэсгээр `.md`, `.txt`, `.pdf` файл оруулаад гарчиг, хэл, лиценз сонгоно.
5. **Асууна.** Жишээ нь: `Чингис хаан хэзээ хаан өргөмжлөгдсөн бэ?`, `When did Genghis Khan die?`, `chinggis khaan khezee khaan bolson be?`

Апп нээгдэхгүй бол **Ports** таб → `8501` → **Open in Browser**. Эсвэл терминалд: `streamlit run app/streamlit_app.py`.

### Терминалаар

```bash
pip install -e ".[dev,ui]"
export GEMINI_API_KEY="..."        # эсвэл .env файлд (git-д орохгүй)

mha check-sources                   # sources.json-ийг шалгах
mha ingest                          # индекс бүтээх
mha ask "Чингис хаан хэзээ хаан өргөмжлөгдсөн бэ?" --level student
mha chat --verify                   # ярилцах, ишлэлийг давхар шалгах
mha gaps                            # архивт юу дутаж байна вэ
mha eval --update-baseline          # анхны оноог тогтоох
mha eval                            # өөрчлөлт бүрийн дараа; буурвал exit 1
```

### Эх сурвалж ба лиценз

`data/sources.json` дотор эх сурвалж бүрийг бүртгэнэ (`id`, `file`, `title`, `lang`, `license` заавал). `.md`, `.txt`, `.pdf` дэмжинэ. Markdown гарчиг (`#`) байвал ишлэл бүлэг хүртэл нарийснэ.

- Репод зөвхөн **нийтийн эзэмшлийн** болон **нээлттэй лицензтэй** эх сурвалж оруулна (`public-domain`, `cc0`, `cc-by`, `cc-by-sa`, `open`, `own`, `permission`).
- Зохиогчийн эрхтэй номыг `license: personal-use` болгож `data/raw/personal/` дотор байлгана. Энэ нь git-д орохгүй.
- Эх сурвалж бүрийн лицензийг өөрөө шалгаж баталгаажуулна уу.

Нэг хүн, газар, үйл явдлын бүх бичлэгийг `data/aliases.json`-д нэг жагсаалтаар нэмбэл хайлт автоматаар өргөжинө.

### Тохиргоо

| Хувьсагч | Анхдагч | Утга |
|---|---|---|
| `GEMINI_API_KEY` | (шаардлагатай) | Google AI Studio түлхүүр |
| `MHA_PROVIDER` | `gemini` | `gemini` эсвэл `anthropic` |
| `MHA_MODEL` | `gemini-3.5-flash-lite` | Түлхүүрээрээ ашиглаж болох загвар |
| `MHA_EMBED_MODEL` | `BAAI/bge-m3` | Олон хэлний embedding (өөрийн машинд) |
| `MHA_RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | `""` бол унтарна (Codespace-д анхдагчаар унтраасан) |
| `MHA_PROMPT` | `v2` | `prompts/` доторх prompt-ийн хувилбар |
| `APP_PASSWORD` | (байхгүй) | Тохируулбал вэб апп нууц үг асууна |
| `MHA_MAX_QUESTIONS` | `40` | Нэг сессийн асуултын дээд хязгаар |

### Анхаарах зүйл

- **Квот ба нууцлал.** Gemini-ийн free tier-ийн хязгаар жижиг бөгөөд өөрчлөгдөж болдог. Хязгаараа AI Studio дээрээс шалгаарай. Free tier дээр илгээсэн агуулгыг Google бүтээгдэхүүнээ сайжруулахад ашиглаж болдог гэж мэдээлэгддэг тул нийтийн эзэмшлийн эх сурвалж, нийтэд нээлттэй асуулт ашиглаарай.
- **Хариултын чанар эх сурвалжаас хамаарна.** Цэвэр текст хамгийн сайн. Скан PDF-ийн OCR алдаа хайлтад нөлөөлнө.
- **AI алдаж болно.** Чухал баримтыг үргэлж ишлэлийн эх сурвалжаас нь шалгаарай.
- Латин бичигтэй Монгол асуултыг Кирилл болгох ажлыг загвар өөрөө хийдэг, тусгай хөрвүүлэгч алга.
- Хуучин монгол бичиг (ᠮᠣᠩᠭᠣᠯ бичиг) шууд дэмжигдээгүй.
- Eval нь түлхүүр үг болон ишлэлийн бүтцийг шалгадаг, утга зүйн зөвийг бүрэн хэмждэггүй.
- Төсөл эхний шатандаа байна. Жинхэнэ эх сурвалжтай томоохон туршилт дөнгөж эхэлж байна.

### Хөгжүүлэлт

```bash
pip install -e ".[dev,ui]"
pytest -q        # 52 тест; хүнд загвар хэрэггүй, хуурамч embedder болон AppTest ашиглана
```

Зорилтот чиглэлүүд: entity граф, timeline ба газрын зургийн дүрслэл, Монгол бичгийн OCR, LLM шүүгчтэй eval, Латин-Кирилл тусгай хөрвүүлэгч. Санал, алдааны мэдээллийг Issues-ээр хүлээн авна.

---

## 🇬🇧 English

### What is it?

You provide your own collection of historical sources (chronicles, scholarship, articles). The agent searches them, answers **only from the retrieved text**, and backs every sentence with a citation. If the archive doesn't cover a question, it says so instead of guessing.

### Features

- 🔎 **Multilingual retrieval.** Mongolian (Cyrillic and Latin script), English, Russian and Chinese queries share one embedding space (`bge-m3`). Name variants are expanded automatically (Тэмүжин ↔ Temüjin ↔ 铁木真).
- 📌 **Citations required.** Every factual sentence links to a source label like `[S1]`; unresolved citations are flagged.
- ⚖️ **Conflicting accounts side by side.** "Source A says X; source B says Y."
- 📅 **Chronological answers.** Short answer, ordered explanation, disagreements, confidence level and follow-up questions. Search can be filtered by year.
- 🎓 **Explanation levels.** Brief, standard, student, researcher.
- ✅ **Optional verification pass.** A separate call checks each cited claim and rewrites unsupported ones.
- 🔁 **Self-improvement loop.** Logs coverage gaps, turns 👎 feedback into eval candidates, detects score regressions, compares prompt versions.
- 💸 **Free to start.** Google Gemini API (AI Studio free tier) by default; embeddings and search run locally. Switch to Claude with `MHA_PROVIDER=anthropic`.
- 🖥️ **Web app and CLI.** Add sources by uploading files in the app.

### How it works

```
Question ─► Agent (LLM + tool use) ─► Search ─► Cited answer ─► (verification)
                                        │
   alias expansion → vector + BM25 → RRF → year filter → (reranker) → per-source balancing
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

### Quick start (no computer needed, GitHub only)

1. **Get a key** at [aistudio.google.com](https://aistudio.google.com) → **Get API key** → **Create API key**. No credit card required.
2. **Store it as a secret.** GitHub → Settings → **Codespaces** → **New secret**. Name: `GEMINI_API_KEY`. Value: your key. Grant access to this repository.
3. **Open a Codespace.** On the repo page: **Code → Codespaces → Create codespace on main**. The first start takes a few minutes, then the web app opens automatically.
4. **Add sources.** In the sidebar, open **➕ Add source** (shown in Mongolian as **Эх сурвалж нэмэх**), upload a `.md`, `.txt` or `.pdf` file, and choose a title, language and license.
5. **Ask.** For example: `When did Genghis Khan die?` or `Чингис хаан хэзээ хаан өргөмжлөгдсөн бэ?`

If the app doesn't open, use the **Ports** tab → `8501` → **Open in Browser**, or run `streamlit run app/streamlit_app.py` in the terminal.

### Command line

```bash
pip install -e ".[dev,ui]"
export GEMINI_API_KEY="..."        # or put it in .env (git-ignored)

mha check-sources                   # validate sources.json
mha ingest                          # build the index
mha ask "When was Genghis Khan proclaimed ruler?" --level researcher
mha chat --verify                   # conversation with the verification pass
mha gaps                            # what is missing from the archive?
mha eval --update-baseline          # record a baseline score
mha eval                            # after every change; exits 1 on regression
```

### Sources and licensing

Register each source in `data/sources.json` (`id`, `file`, `title`, `lang`, `license` are required). `.md`, `.txt` and `.pdf` are supported; markdown headings (`#`) make citations point to the section.

- Commit only **public-domain** or **openly licensed** material (`public-domain`, `cc0`, `cc-by`, `cc-by-sa`, `open`, `own`, `permission`).
- Copyrighted books: use `license: personal-use` and keep them in `data/raw/personal/` (git-ignored).
- Verify each source's license yourself.

Add all spellings of a person, place or event as one list in `data/aliases.json` and search expands automatically.

### Configuration

| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | (required) | Google AI Studio key |
| `MHA_PROVIDER` | `gemini` | `gemini` or `anthropic` |
| `MHA_MODEL` | `gemini-3.5-flash-lite` | Any model your key can use |
| `MHA_EMBED_MODEL` | `BAAI/bge-m3` | Multilingual embeddings (run locally) |
| `MHA_RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | `""` disables it (off by default in Codespaces) |
| `MHA_PROMPT` | `v2` | Prompt version in `prompts/` |
| `APP_PASSWORD` | (unset) | If set, the web app asks for a password |
| `MHA_MAX_QUESTIONS` | `40` | Per-session question cap |

### Good to know

- **Quota and privacy.** Gemini's free tier has small limits that change over time; check yours in AI Studio. Content sent on the free tier may be used by Google to improve its products, so use public-domain sources and public questions.
- **Answer quality depends on your sources.** Clean text works best; OCR errors in scanned PDFs hurt retrieval.
- **AI can be wrong.** Always verify important facts against the cited source.
- Latin-script Mongolian is handled by the model itself; there is no dedicated transliterator.
- Traditional Mongolian script (ᠮᠣᠩᠭᠣᠯ ᠪᠢᠴᠢᠭ) is not directly supported.
- The eval checks keywords and citation structure, not full semantic correctness.
- This is an early-stage project; large-scale tests with real corpora have only just begun.

### Development

```bash
pip install -e ".[dev,ui]"
pytest -q        # 52 tests; no heavy models needed (fake embedder + Streamlit AppTest)
```

Roadmap: entity graph, timeline and map views, Mongolian-script OCR, LLM-judged eval, a dedicated Latin→Cyrillic converter. Issues and ideas are welcome.

---

MIT License
