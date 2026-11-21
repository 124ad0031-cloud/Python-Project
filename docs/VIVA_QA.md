# Viva Preparation: 41 questions

Each has a **short answer to say aloud** and **more detail for follow-ups**.


## Problem and usefulness

### 1. What problem does your project solve?
- **Say this:** It helps recruiters screen many resumes quickly by ranking them on meaning, not just exact keywords, and it shows the evidence.
- **If asked more:** Keyword filters reject people who describe a skill in other words and reward keyword stuffing. ResumeLens compares meaning with a pretrained sentence Transformer and explains each score.

### 2. Who would use it?
- **Say this:** Recruiters and HR teams, and also candidates who want to check how well their resume covers a job.
- **If asked more:** It is decision support: the recruiter sees the evidence sentence per requirement, skill gaps and suggested interview questions.

### 3. How is it different from a normal keyword resume filter?
- **Say this:** It understands that 'built web services' and 'REST APIs' mean similar things.
- **If asked more:** A keyword filter needs the same words. A sentence Transformer maps both sentences to nearby vectors. The 'Transformer vs keywords' tab shows both scores side by side.

### 4. What is your standout feature?
- **Say this:** Evidence-based matching, a live keyword-versus-Transformer comparison, and blind screening.
- **If asked more:** Each requirement is linked to the exact resume sentence that supports it, so a human can verify the score. Blind mode masks names and contact details.

### 5. Is the system meant to replace recruiters?
- **Say this:** No. It supports human decisions.
- **If asked more:** Automatic rejection raises fairness and legal concerns. The scores are explained and adjustable, and the limitations are stated.


## NLP concepts

### 6. What is NLP?
- **Say this:** Making computers understand and process human language.
- **If asked more:** Here: parsing text, converting it to vectors, measuring similarity, and extracting skills.

### 7. What is a sentence embedding?
- **Say this:** A list of numbers (384 here) that represents the meaning of a whole sentence.
- **If asked more:** A Transformer produces a vector per token; mean pooling averages them into one sentence vector. Similar meaning gives nearby vectors.

### 8. What is semantic similarity?
- **Say this:** How close two texts are in meaning, not in spelling.
- **If asked more:** Measured as the cosine similarity of their embeddings: near 1 = very similar, near 0 = unrelated.

### 9. What is cosine similarity and why not Euclidean distance?
- **Say this:** The cosine of the angle between two vectors; it measures direction, not length.
- **If asked more:** With normalised vectors it equals the dot product. It is the standard for embeddings because the length of a vector is not meaningful for similarity.

### 10. What is TF-IDF and why did you keep it?
- **Say this:** A word-counting method; I keep it as a baseline to show what the Transformer adds.
- **If asked more:** TF = how often a word appears; IDF = down-weights common words. It cannot see synonyms.

### 11. What is mean pooling?
- **Say this:** Averaging all token vectors of a sentence into one vector.
- **If asked more:** The sentence-transformers model uses it as the pooling layer after the Transformer encoder (attention-masked so padding is ignored).

### 12. What is tokenization?
- **Say this:** Splitting text into pieces (word-pieces) and giving each an ID.
- **If asked more:** Done inside the model's tokenizer (WordPiece, as in BERT). I also use regular expressions to split resumes into sentences.

### 13. What is skill extraction and how did you do it?
- **Say this:** Finding skill names in text using a dictionary and regular expressions.
- **If asked more:** skills.py maps each skill to aliases (e.g. 'sklearn' = scikit-learn). Whole-word regex matching avoids partial matches.

### 14. Is this Named Entity Recognition?
- **Say this:** Not exactly: it is dictionary-based skill matching, a simpler cousin of NER.
- **If asked more:** A future version could use an NER model to extract skills, companies, degrees and dates.


## Transformer and model

### 15. Why did you use a Transformer?
- **Say this:** Because meaning matching needs context-aware embeddings, which Transformers provide.
- **If asked more:** Older word vectors give one fixed vector per word. Transformers use self-attention so each word's vector depends on its sentence, giving far better sentence-level comparison.

### 16. Which model did you use and why?
- **Say this:** all-MiniLM-L6-v2: small, fast, and trained specifically for sentence similarity.
- **If asked more:** 6 layers, hidden size 384, about 22 M parameters, about 90 MB, runs on CPU, no API. Trained contrastively on over a billion sentence pairs.

### 17. Is it an encoder or decoder model?
- **Say this:** Encoder-only.
- **If asked more:** It reads the whole sentence and produces representations. Decoder models like GPT generate text.

### 18. Why not use BERT directly?
- **Say this:** Plain BERT is not trained to make whole-sentence vectors that compare well.
- **If asked more:** Averaging BERT vectors gives poor sentence similarity. Sentence-BERT-style models are fine-tuned with contrastive learning so similar sentences are close.

### 19. What is contrastive learning?
- **Say this:** Training so matching pairs get close vectors and non-matching pairs get far vectors.
- **If asked more:** Other sentences in the same batch act as negatives. The loss rewards high similarity for true pairs and low similarity for the rest.

### 20. Bi-encoder vs cross-encoder?
- **Say this:** Bi-encoder encodes texts separately and compares vectors; cross-encoder reads the pair together.
- **If asked more:** Bi-encoders are much faster for many comparisons. Cross-encoders are more accurate and could re-rank the top candidates in a future version.

### 21. Where is attention used in your project?
- **Say this:** Inside the model: every layer uses self-attention to build contextual token vectors before pooling.
- **If asked more:** I do not visualise attention here because for a recruiter the evidence sentence is a clearer explanation. (My earlier prototype AttentionLens did visualise it.)

### 22. Did you train or fine-tune the model?
- **Say this:** No. I used the pretrained model as it is.
- **If asked more:** Fine-tuning needs labelled resume-job pairs; that would be a future improvement.

### 23. Can the model understand any language?
- **Say this:** No, this model is for English.
- **If asked more:** Multilingual models such as paraphrase-multilingual-MiniLM could be swapped in by changing MODEL_NAME.


## Scoring and design

### 24. How is the final score calculated?
- **Say this:** A weighted mix: 60% requirement coverage by meaning and 40% skill-keyword coverage, by default.
- **If asked more:** Requirement coverage: Strong = 1, Partial = 0.5, Missing = 0, averaged over requirements. Weight is adjustable in the sidebar.

### 25. How do you decide Strong, Partial and Missing?
- **Say this:** By similarity thresholds, 0.50 and 0.35 by default.
- **If asked more:** They are heuristics, not learned values, adjustable in the sidebar. I state this limitation openly; calibrating them needs labelled data.

### 26. Why combine semantic matching with a skill list?
- **Say this:** Two independent signals are more reliable, and the skill list is transparent.
- **If asked more:** Semantic matching handles different wording; the list gives an easily checkable view of explicit skills. A resume that expresses a skill in other words can still be credited through the semantic part.

### 27. How do you find the job requirements?
- **Say this:** I take the lines after a heading like 'Requirements', each with 3 or more words.
- **If asked more:** If no heading is found, all sentences are used. The extracted list is shown in the app so the user can check it.

### 28. What is a 'Transformer-only match'?
- **Say this:** A requirement judged matched although the best resume sentence shares almost no words with it.
- **If asked more:** Defined as status not Missing and TF-IDF similarity below 0.10. It counts cases a pure keyword method would miss.

### 29. What does blind screening do?
- **Say this:** It masks names, e-mails, phones and links before scoring.
- **If asked more:** The goal is to reduce bias. It is rule-based and cannot remove every identifying detail (e.g. a university name).

### 30. What happens if the model cannot be loaded?
- **Say this:** The app shows a banner and falls back to the keyword method instead of crashing.
- **If asked more:** load_model returns (None, error) and semantic_similarity calls keyword_similarity when the model is None.


## Implementation

### 31. What libraries did you use and why?
- **Say this:** Streamlit (UI), sentence-transformers and PyTorch (model), scikit-learn (TF-IDF, cosine), pandas/numpy (tables, arrays), matplotlib (heatmap), pypdf (PDF text).
- **If asked more:** Each solves one small task, so the code stays short.

### 32. What happens when I upload a PDF?
- **Say this:** pypdf extracts its text; the text is split into sentences and analysed like any resume.
- **If asked more:** Scanned image PDFs have no text layer and give a warning.

### 33. What does model.encode do?
- **Say this:** It converts a list of texts into a matrix of embeddings.
- **If asked more:** With normalize_embeddings=True each row has length 1, so a dot product between matrices gives the cosine similarity matrix.

### 34. Explain the line similarity = vectors_a @ vectors_b.T
- **Say this:** It computes every requirement-sentence similarity at once.
- **If asked more:** Shape (requirements x 384) times (384 x sentences) = (requirements x sentences). I clip negatives to 0.

### 35. Why separate matcher.py and app.py?
- **Say this:** NLP logic is separated from the interface, so it is easy to test and explain.
- **If asked more:** test_core.py tests matcher.py without Streamlit.

### 36. How is the app kept fast?
- **Say this:** The model is loaded once, and analysis results are cached.
- **If asked more:** st.cache_resource for the model, st.cache_data for results. The model is small enough for a CPU.


## Evaluation and limitations

### 37. How did you evaluate it?
- **Say this:** I tested the code paths and behaviours, but I did not run a formal accuracy benchmark.
- **If asked more:** I verified the scoring logic with a keyword-only run, tested the sentence-Transformer loading and all UI paths with a randomly initialised stand-in model, and tested PDF reading. Real-model results are what I observe on my laptop. A proper evaluation would compare rankings with human recruiters' judgements, using precision@k or Spearman correlation.

### 38. What are the limitations?
- **Say this:** Heuristic thresholds, English only, small skill dictionary, no scanned PDFs, and the model can carry bias.
- **If asked more:** It treats sentences independently so it misses information spread over several sentences, and it does not check years of experience.

### 39. Could the system be biased?
- **Say this:** Yes, models can inherit bias from their training data.
- **If asked more:** Blind mode removes some signals, scores are explained, and humans make the final decision. Auditing across demographic groups would be needed before real use.

### 40. What future improvements would you make?
- **Say this:** Fine-tune on real resume-job pairs, add NER for skills, years of experience and education, a cross-encoder re-ranker, multilingual support.
- **If asked more:** Also a database of candidates and semantic search across all stored resumes.

### 41. What did you verify and what did you not?
- **Say this:** I verified logic and software behaviour; I did not verify real-model similarity values before the demo.
- **If asked more:** That is why test_core.py prints the real ranking and the thresholds are adjustable.
