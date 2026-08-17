# 🖼️ Image Embedding Workflow — The Standard

*How images get added to topic notes in this vault — for retrofitting existing notes AND for every new note going forward. Companion to `_TEMPLATE — Topic Note Format.md`. The CNN note (`Machine Learning/Computer Vision/Convolutional Neural Networks for Vision.md`) is the reference exemplar; the 6 other Computer Vision notes are the first batch produced with this process.*

Goal: make notes **faster to understand** by adding real figures (architecture diagrams, mechanism illustrations, annotated plots) — not prettier ASCII. **At least 1–3 images per note, with NO upper cap** — add as many as genuinely aid understanding (a dense, multi-concept note may warrant 5–8+); each image must earn its place, and skip a note only if no image helps at all. Don't drop a useful figure just to stay under a number.

---

## Progress tracker (update as you go)

| Topic | Source notebooks | Status |
|---|---|---|
| Computer Vision — CNN (pilot) | Intro_to_CV_and_CNN_Fundamentals | ✅ done (3 imgs) |
| Computer Vision — 6 other notes | see mapping below | ✅ done (14 imgs), reviewed — **user has pending un-specified change requests to CV** |
| NLP (6 notes) | see mapping | ✅ done (22 imgs) 2026-07-24 — Text Preprocessing 3 (self-made), Word Embeddings 4 (Alammar), NER 2 (self-made), Sentiment 2 (notebook NRC + self-made LDA), BERT 4 (Alammar), RNN·LSTM·Transformers 7 (colah×3 + Alammar×3 + notebook attn-heatmap). Notebook-first yielded only 2 usable concept figures (attn heatmap, NRC emotion) — rest code-heavy plots; backbone = canonical (colah/Alammar) + self-made. **Committed `5a6ccad`, pushed 2026-07-24** (commit also fixed a blank-line-less table in BERT §6). |
| Unsupervised ML (4) | see mapping | ✅ done (19 imgs) 2026-07-25 — GMM 4 (self-made: 1-D mixture, covariance-types, EM iterations, GMM-vs-KMeans), Anomaly Detection 4 (self-made iForest isolation-depth + canonical sklearn OneClassSVM-frontier/LOF/methods-comparison), Clustering 6 (self-made Lloyd-iterations/elbow+silhouette/DBSCAN-core-border-noise + canonical kmeans-assumptions/cluster-comparison + notebook Ward dendrogram-with-cut), PCA & t-SNE 5 (self-made projection/scree/Gaussian-vs-Student-t + canonical perplexity + notebook PCA-vs-tSNE composite). Notebook-first again mostly code-heavy results plots → backbone = self-made matplotlib (real EM/K-Means/DBSCAN/silhouette in numpy) + canonical sklearn (all attributed) + 2 notebook figs. **Uncommitted, pending user review.** |
| Supervised ML (7) | see mapping | ✅ done (27 imgs) 2026-07-26 — Naive Bayes 3 (self-made: Bayes-flip mechanism, zero-frequency/Laplace bars, Gaussian-NB boundary), KNN 4 (self-made: decision-boundary-vs-K, error-vs-K U-curve, curse-of-dimensionality distance-concentration, plain-vs-weighted vote), Classification Metrics 3 (self-made: confusion-matrix precision-column/recall-row, threshold-tradeoff overlapping distributions, ROC-vs-PR under imbalance), Logistic Regression 3 (self-made: sigmoid+linear-boundary proba field, why-not-MSE convexity via chord test, log-loss penalty curves), Decision Trees 5 (**1 notebook** graphviz trained tree [black-box artifact cleaned via PIL] + self-made: partition↔flowchart, Gini-vs-entropy, depth-boundary staircase, train/test-vs-depth), Linear Regression 5 (self-made: best-fit residuals, GD on MSE bowl+loss curve, residual diagnostics 3-panel, Ridge/Lasso geometry [**true** computed tangent pts: lasso→corner [1.7,0], ridge→off-axis], poly underfit/good/overfit), Ensemble 4 (self-made: bagging-vs-boosting concept diagram, variance-vs-ρ floor, boosting-fits-residuals staged, single-tree/RF/GBM boundaries). Notebook-first again mostly code-heavy results plots → backbone = self-made matplotlib (real sklearn models: GaussianNB/KNN/LogReg/DecisionTree/RF/GBM decision regions, real GD & GBM staged fits) + 1 notebook fig. **Committed `6036ad6`, pushed 2026-07-26.** |
| Neural Networks (5) | see mapping | ✅ done (24 imgs) 2026-07-28 — Neural Network Fundamentals 5 (self-made: LogReg-vs-2-hidden-layer-MLP on a spiral 66%→94% [real sklearn], single-neuron anatomy + fully-connected MLP, backprop computational-graph with exact §6 grads dZ²=A²−Y/dW₂=A¹ᵀdZ²/dZ¹=dA¹⊙f'(Z¹)/dW₁=XᵀdZ¹, sigmoid/tanh/ReLU/leaky + derivatives; **1 notebook** touch-decision neuron NN_1), Batch Normalization & Dropout 4 (**canonical** Srivastava JMLR-2014 standard-vs-thinned [attributed] + self-made: BN 3-stage mean4/var6→0/1→rescale γ=1.5, Dense→BN→Act→Dropout ordering, inverted-dropout train ×1/(1−p) vs inference), Weight Init & Optimizers 4 (**canonical** optimizer-trajectory GIF by Jaewan Yun after Radford [attributed, 30MB→5.1MB] + self-made: activation-variance across 30-layer ReLU He-vs-bad-init, EMA β=0.9/0.99 smoothing, LR-too-high/low/good + step/exp/cosine/warmup schedules), Autoencoders 6 (self-made: encoder-bottleneck-decoder hourglass, undercomplete-vs-overcomplete, AE-vs-PCA line-vs-curve; **3 notebook** MNIST denoising, 32-D reconstruction, t-SNE latent baseline — the notebook's own degenerate AE 2-D scatter was REJECTED and replaced by the honestly-labeled t-SNE), Text Classification 5 (self-made: text→embedding→pool→sigmoid pipeline, 10000×128 embedding lookup, Flatten 1.64M vs GlobalAvgPool 8.2K params, Model B stack with shapes; **1 notebook** param-comparison bar [failed-accuracy panel cropped off]). Pattern held: notebook figs mostly code-heavy → backbone = self-made matplotlib (real sklearn LogReg/MLP, simulated forward-pass variance, real GD) + 2 canonical (both attributed) + 5 notebook figs. Orchestrator-reviewed all 24 (read every image, verified links/no-`![[`/no-bracket-in-alt/no-orphans; fixed one self-made pipeline figure whose toy padded to L=5 but labeled the embed 200×128 → corrected to 5×128). Gen scripts nnA/nnB in scratchpad. **Committed `662d0a8`, pushed 2026-07-28.** |
| Time Series (2) | see mapping | ✅ done (10 imgs) 2026-07-28 — Foundations & Smoothing 5 (**1 notebook** seasonal_decompose 4-panel + self-made: additive-vs-multiplicative constant-vs-growing swing, baselines mean/naive/seasonal-naive over held-out block, exp-decay weights α(1−α)ᵏ for α=0.3/0.6, SES-flat/Holt-sloped/HW-seasonal comparison via **real numpy smoothing recursions** — Holt trend re-init deseasonalized+small α,β so it recovers the true upward slope through the season), SARIMAX 5 (**all notebook**: differencing before→after composite [raw trend → diff(1).diff(12) stationary], raw-series ACF seasonal spikes @12/24, PACF→p / ACF→q order-reading composite on stationary series with lag-12 seasonal flag, plot_diagnostics white-noise 4-panel, forecast + 95% prediction-interval band). Time-series notebooks yielded MORE usable canonical figures than any prior topic (seasonal_decompose, ACF/PACF, plot_diagnostics, conf_int all clean) → backbone = 6 notebook + 4 self-made; 2 PIL composites (stacked panels + labeled header bands). All 10 orchestrator-reviewed (read every image; fixed Holt negative-trend bug + additive-panel title/annotation collision). Gen scripts gen_ts_figs/build_notebook_figs in scratchpad. **Committed `695b1d3`, pushed 2026-07-28.** |
| Recommendation Systems (2) | see mapping | ✅ done (10 imgs) 2026-07-28 — Recommendation Systems 6 (all self-made: sparse user×item matrix [missing≠dislike, 24/80 filled], item-item vs user-user CF side-by-side [cosine over columns vs rows], cold start [empty new-user row + new-item column], matrix factorization A≈U·Vᵀ box decomposition, learned latent-factor space scatter [Manish's taste vector aligned with cerebral movies, orthogonal to action → dot-product = rating], truncated-SVD rank-1 theme layers + singular-value scree with top-k cut), Association Rules 4 (all self-made: market-basket transactions→two frequent itemsets [Bread+Milk, Beer+Diapers], **Apriori downward-closure lattice** [4-item Hasse diagram, {D} infrequent → all D-supersets struck-through/pruned green-vs-red], support/confidence/lift Venn [X∩Y overlap in N-transaction universe], confidence-vs-lift trap dual-bar [conf 0.80≈0.71 looks same, lift 8.0 vs 0.89 across the =1 line]). The `Recommendation_systems_1` notebook covers BOTH notes (cells 1-55 assoc-rules, 56-99 recsys) but is entirely markdown-concept + DataFrame outputs → **0 usable figures**, so 100% self-made matplotlib/PIL. Two bugs caught+fixed in self-review (sparse-matrix rotated-label overlap → vertical labels; Apriori single ABCD node stuck at far-left → centered). All 10 orchestrator-reviewed (read every image, verified links/no-`![[`/no-bracket-in-alt/no-orphans). Gen scripts gen_rs_figs/gen_ar_figs in scratchpad. **Committed `446ffc4`, pushed 2026-07-28. 🎉 ROLLOUT COMPLETE — all topics done, 148 images total across the vault.** |
| AI Engineering (6 notes) | self-made matplotlib + canonical (SBERT/CLIP) | ✅ done (19 imgs) 2026-07-22 — Embeddings 3, Prompt Eng 3, Evaluating LLMs 3, RAG 3, LLM 4, Multimodal RAG 3 |
| AI Engineering — Advanced AI Agents track (new notes, standing rule) | **lecture-PDF figures** | ✅ AI Agents — Foundations 4 (self-made) 2026-08-03 · **Agentic Workflows 8** + **Model Context Protocol 8** (2026-08-09) · **Fine-Tuning LLMs 10** + **Distributed Training for LLMs 3** (2026-08-15 — 6 extracted from the handwritten lecture PDFs via the `pdfimages` + smask-composite route, 7 self-made matplotlib) |

> **New source route proven 2026-08-09 — extract figures straight out of the lecture PDFs.** The Advanced-AI-Agents lectures ship as (a) handwritten-annotation PDFs with *clean raster figures underneath the ink*, and (b) vector "pattern reference" PDFs. Both beat self-made diagrams and cost minutes:
> - **Raster figures under handwriting:** `pdfimages -png -p lecture.pdf out/im` pulls the embedded images *without* the instructor's ink overlay (the ink is vector, drawn on top). Filter to real figures with PIL: keep `w ≥ 500 and h ≥ 200`, then `Read` only those. This yielded all 8 MCP figures (architecture infographic, three-primitives, client bridge, Anthropic sequence diagrams, M×N / M+N).
> - **Vector-only PDFs** (`pdfimages` returns 0): render the page with `pdftoppm -r 150 -png -f N -l N`, then auto-crop the schematic — mask rows containing *dark* ink (`min(RGB) < 140`), merge runs with a ≤22 px bridge, keep the first run ≥90 px, and trim columns at a *lighter* threshold (`< 215`) so pale box borders aren't clipped. Pad ~14 px. Verify with a stacked contact sheet (one `Read` for all crops) rather than reading each page.
> - Attribution: credit the original where identifiable (`*Source: Anthropic, "Building Effective Agents" (Dec 2024), as reproduced in the lecture.*`), else `*Source: Scaler lecture deck.*`

Rollout order: one **agent per topic folder**, in the order above. Pilot-then-review cadence: complete a topic, orchestrator reviews every image, user reviews, then commit + next topic.

---

## Prerequisite — Google Drive access (macOS Full Disk Access)

Lecture notebooks live at:
`/Users/manishdiddi/Library/CloudStorage/GoogleDrive-manishdiddi03@gmail.com/My Drive/Colab Notebooks/Machine Learning/<Area>/`
The **Drive MCP download route is unusable** (returns whole notebook as base64 into context). The only cheap route is the **local mount + `jq`/Python + `base64 -d`** (streams from disk).
If `ls "<that path>"` returns `Operation not permitted`, the mount is blocked → grant **System Settings → Privacy & Security → Full Disk Access → (Terminal/iTerm/VS Code) → quit & relaunch**. (This was granted on 2026-07-21; macOS can reset it on app updates.)

---

## Sourcing order (best-image-wins)
Pick the image that teaches the concept best. **Licensing is NOT a gate** — this is a personal, non-commercial educational study vault, so source the clearest figure regardless of where it lives (blogs, papers, course sites, canonical diagrams). Prefer sources in this order only because *quality/fit* tends to fall this way, not because of licensing:
1. **First** extract from the topic's Scaler notebook (real plot outputs + architecture diagrams in markdown cells) — best-matched to the note, and the author's own course.
2. **Then** a **famous canonical diagram** (Wikimedia Commons, papers on arXiv, well-known blog/course figures, GitHub repos like vdumoulin/conv_arithmetic). `curl -sL` the direct URL into `attachments/`. Scraping popular sites for the right figure is explicitly allowed.
3. **Else** build a **self-made diagram** (matplotlib/PIL) or skip if nothing genuinely helps.

**Attribution stays mandatory for any third-party image** — add a one-line italic source credit under it (e.g. `*Source: Wikimedia Commons* ` / `*Source: <site/author>*`). Reasons it stays: it's good scholarship, it makes the image swappable later, and for an educational vault it keeps you on solid **fair-use** footing (non-commercial, transformative, teaching context). Notebook-derived images (author's own course) need NO credit line.

**Residual risk (know the posture):** this is a *public* GitHub repo. Embedding a third-party image with attribution under educational fair use is a reasonable stance; the only real exposure is that if a rights-holder ever objects, the fix is to **swap that one image**. That's the whole risk — don't let it block you, just keep images swappable (attribution + kept in `attachments/`).

Prefer CONCEPT / mechanism / architecture images over mere results plots. A results plot earns a slot only if it teaches something the note emphasizes.

## Source-image quality gate (REJECT even if from the notebook or a scraped site)
Licensing isn't a gate, but **quality still is** — and this matters *more* now that scraping popular sites is allowed, since stock/blog results are where junk comes from. Reject any image with: a **watermark / stock-photo overlay** (alamy, Getty, Shutterstock…) — grab a clean, unwatermarked version or a different figure instead; **weapons, violence, gore, or off-topic/inappropriate content**; illegible text / heavy artifacts / clutter. If the chosen cell fails → cleaner cell, self-made figure, or **drop it** (fewer clean images beat one bad one). *(This rule exists because an early CV pass grabbed a watermarked stock photo of a man aiming a gun for an Object-Detection localization slot — dropped.)*

## Embedding convention (match the CNN exemplar exactly)
- Standard markdown only: `![full descriptive caption sentence](attachments/kebab-name.ext)`. **Never** `![[...]]` (dead on GitHub).
- **Alt text IS the caption** — a full, specific sentence naming what's shown and the lesson. No separate caption line, EXCEPT the one-line italic source credit for third-party images.
- **No `[` or `]` inside the alt text** (breaks markdown) — reword things like `[-1,0,1]`.
- Files in an `attachments/` folder **next to the note** (`mkdir -p` if missing). Kebab-case names.
- Place image right after the paragraph/heading it illustrates. **Keep existing ASCII diagrams as companions by default** — but you **may replace an ASCII diagram with the image when the image is clearly clearer** (the ASCII was a stand-in for a figure). Keep the ASCII when it carries detail the image doesn't (exact numbers, extra annotations); replace it when the image simply says the same thing better.
- Animated **GIFs allowed where motion genuinely helps** a process/mechanism (e.g. convolution, gradient descent, attention); static PNG otherwise. Both render on GitHub + Obsidian.

## Scope & safety
- Only CREATE images in the note's `attachments/` and EDIT that note's `.md`. DELETE only files you created this task. Never delete/overwrite pre-existing content or folders you didn't author. Don't touch `.DS_Store` / `.obsidian/`. Don't run `git` inside a topic agent — the orchestrator commits.

---

## Extraction workflow (cheap — never dump base64 into context)

Recreate this extractor into the session scratchpad, then run `python3 extract_nb_images.py "<notebook.ipynb>" "<outdir>"`. It writes every image to disk + a `manifest.txt` (image → source-cell context). **Triage via the manifest first**, then `Read` only ~5–15 promising candidates (big notebooks have hundreds).

```python
#!/usr/bin/env python3
"""Extract embedded images from a Jupyter notebook WITHOUT dumping base64 into context.
Usage: python3 extract_nb_images.py <notebook.ipynb> <outdir>
Writes <outdir>/<NN>_<kind>_cell<idx>.<ext> + manifest.txt (context only; base64 never printed)."""
import base64, json, os, sys, re
def join_src(s): return "".join(s) if isinstance(s, list) else (s or "")
def clean(txt, n=280): return re.sub(r"\s+", " ", txt).strip()[:n]
def decode_to(path, b64):
    if isinstance(b64, list): b64 = "".join(b64)
    open(path, "wb").write(base64.b64decode(b64))
def main():
    nb_path, outdir = sys.argv[1], sys.argv[2]; os.makedirs(outdir, exist_ok=True)
    nb = json.load(open(nb_path)); manifest = []; seq = 0; last_md = ""
    for i, cell in enumerate(nb.get("cells", [])):
        ctype, src = cell.get("cell_type"), join_src(cell.get("source"))
        if ctype == "markdown":
            last_md = clean(src)
            for name, mimed in (cell.get("attachments") or {}).items():
                for mime, data in mimed.items():
                    if mime.startswith("image/"):
                        seq += 1; ext = mime.split("/")[-1].replace("jpeg","jpg")
                        fn = f"{seq:02d}_mdimg_cell{i}.{ext}"; decode_to(os.path.join(outdir,fn), data)
                        manifest.append((fn,"MD-ATTACHMENT",i,clean(src)))
        elif ctype == "code":
            code_ctx = clean(src,160)
            for out in cell.get("outputs",[]) or []:
                for mime,val in (out.get("data") or {}).items():
                    if mime.startswith("image/"):
                        seq += 1; ext = mime.split("/")[-1].replace("jpeg","jpg").replace("svg+xml","svg")
                        fn = f"{seq:02d}_output_cell{i}.{ext}"; decode_to(os.path.join(outdir,fn), val)
                        manifest.append((fn,"CODE-OUTPUT",i,clean(f"[md] {last_md} || [code] {code_ctx}",320)))
    open(os.path.join(outdir,"manifest.txt"),"w").writelines(f"{fn}\t{k}\tcell#{i}\t{c}\n" for fn,k,i,c in manifest)
    print(f"{os.path.basename(nb_path)}: {len(manifest)} images")
    for fn,k,i,c in manifest: print(f"{fn} [{k} {i}] {c}")
if __name__ == "__main__": main()
```

**Image tooling:** PIL/Pillow available (no ImageMagick). Trim a Keras `plot_model` PNG with `ImageChops.difference` vs white + `getbbox` + small pad. Compose before/after by resizing to equal height and pasting on a white canvas with `ImageDraw` labels (`/System/Library/Fonts/Supplemental/Arial.ttf`). Fetch canonical GIF/PNG with `curl -sL <url> -o attachments/name.ext`.

**Self-made diagrams** (when notebook lacks a concept figure): build clean matplotlib/PIL box-and-arrow diagrams (see the CV self-made set — GAN loop, U-Net, semantic-vs-instance, IoU+anchors, triplet, transfer freeze/fine-tune). Verify every label is correct before embedding.

---

## Orchestration & review protocol
- Dispatch **one general-purpose agent per topic folder**. Give it this doc + the note↔notebook mapping + per-note hints.
- **Crash-resilience:** the agent completes and WRITES each note fully before starting the next (smallest notebooks first), so an interruption leaves finished notes intact. (Agents can die on account **session limits**; their transcripts vanish, so on-disk incrementalism matters — resume is unreliable, re-dispatch for the remainder.)
- **Orchestrator review (mandatory):** `Read` every produced image, check placement + caption accuracy + the quality gate, verify links resolve and no `![[`. Scrutinize self-made diagrams hardest (wrong labels hide there). Send targeted rework for any miss; if the agent's transcript is gone, fix it yourself. Fold every systemic mistake back into this doc so it isn't repeated.
- Verify per note: `grep -n '!\[' <note>` (paths resolve), `grep -n '!\[\[' <note>` (empty), `Read` each final image.

## Note ↔ notebook mapping (Drive `…/Colab Notebooks/Machine Learning/<Area>/`)
**Computer Vision** (done): CNN←Intro_to_CV_and_CNN_Fundamentals · GANs←L11_Introductory_Lecture_on_GANs · Image Segmentation←L9_Object_Segmentation · Object Detection←MobileNet And Object detection 2 stage + ObjectDetection_with_Single_Stage_Methods · Siamese Networks & Image Similarity←L10_Updated_Siamese_Network + Image_Similarity_using_CNN · Tackling Overfitting in CNNs←Tackling_Overfitting_in_CNN · Transfer Learning←Transfer_learning_1.
**NLP:** RNN · LSTM · Transformers←Vanilla_RNN_V2_Instructor + LSTM_V1 + Intro_to_Attention_Mechanisms_V2_Instructors + L10_Transformers_Instructor · BERT←BERT_Instructors · NER←L08_NER_Instructors · Word Embeddings←WordEmbeddings_(Word2Vec)_V1 · Sentiment Analysis & Topic Modeling←V2_of_Sentiment_analysis + Copy_of_Lexicon_Sentiment_Analysis_and_Topic_Modeling · Text Preprocessing←L1_Text_Pre_Processing_(NLTK_spaCy).
**Unsupervised ML:** PCA & t-SNE←PCA___TSNE · Clustering←Intro_to_Clustering__k_Means + K_means++ and Hierarchical Clustering + DBSCAN · GMM←GMM · Anomaly Detection←Anamoly Detection.
**Supervised ML:** Linear Regression←Linear_regression_1..4 · Logistic Regression←Logistic_Regression · Classification Metrics←Classification_Metrics · KNN←KNN · Naive Bayes←Naive Bayes · Decision Trees←Descision Tree · Ensemble Methods…←Random_Forest + Boosting.
**Neural Networks:** Neural Network Fundamentals←NN_1 + NN_2 · Weight Initialization & Optimizers←Weight_initialization_and_Optimisers · Text Classification with Neural Networks←ai_human_text_detection_complete_tutorial · Autoencoders←Autoencoders · Batch Normalization & Dropout←(within NN_1/NN_2, else canonical).
**Time Series:** Time Series Foundations & Smoothing←Time_series_1 + Time_series_2 · Seasonal ARIMA…←Timeseries_3.
**Recommendation Systems:** Recommendation Systems←Recommendation_systems_1 · Association Rules & Market Basket Analysis←(canonical if no notebook).
**AI Engineering:** no Drive notebooks — canonical open-licensed only; already has 3 images.
