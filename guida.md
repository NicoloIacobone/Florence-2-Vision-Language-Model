# Guida a Nico_test.ipynb

Questa guida riassume le funzionalità presenti nel notebook `Nico_test.ipynb` e fornisce istruzioni pratiche per eseguire ogni task. Tutti gli esempi assumono che il notebook e il relativo ambiente (virtualenv) siano configurati correttamente.

**Prerequisiti**
- Python 3.11 o compatibile e librerie del progetto installate: `pip install -r requirements.txt`.
- Attivare l'ambiente virtuale se presente: `source myenv/bin/activate`.
- File immagine usato nel notebook: modificare il path di `Image.open(...)` se necessario.
- GPU consigliata: molte chiamate usano `.cuda()`.

**Caricamento modello e processor**
- Esempio (dal notebook):

```
model_id = 'microsoft/Florence-2-base-ft'
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval().cuda()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
```

Definisce `model` e `processor` usati da tutte le funzioni di inferenza.

**Funzione di utilità principale: `run_example(task_prompt, text_input=None)`**
- Uso generale:
  - `task_prompt`: stringa di task speciale (es. `'<CAPTION>'`, `'<OD>'`, `'<REFERRING_EXPRESSION_SEGMENTATION>'`, ecc.).
  - `text_input`: (opzionale) testo addizionale richiesto per alcuni task (es. descrizioni, coordinate quantizzate, ecc.).

- Esempio di chiamata:

```
results = run_example('<CAPTION>')
results = run_example('<CAPTION_TO_PHRASE_GROUNDING>', text_input='A woman holding a cat.')
```

- Output: la funzione restituisce `parsed_answer`, un dizionario il cui contenuto dipende dal task richiesto. Per molti task il risultato è un dizionario con chiavi come `'<OD>'`, `'<DENSE_REGION_CAPTION>'`, ecc.

**Inizializzazione immagine**
- Nel notebook l'immagine è caricata e ridimensionata con `resize_longest_side(image, 512)`.
- Assicurarsi che la variabile `image` sia definita prima di chiamare i task.

----
Sezione per task specifici (uso, input, output e visualizzazione)

1) **Caption / Detailed Caption / More Detailed Caption**
- Token: `'<CAPTION>'`, `'<DETAILED_CAPTION>'`, `'<MORE_DETAILED_CAPTION>'`.
- Esecuzione: `run_example(task_prompt)` senza `text_input`.
- Output: testo (stringa) associato alla chiave del task. Es.: `results['<CAPTION>']`.
- Uso tipico: catturare il testo e visualizzarlo o passarlo ad altri task.

2) **Object Detection (OD)**
- Token: `'<OD>'`.
- Esecuzione: `results = run_example('<OD>')`.
- Output: `results['<OD>']` è un dizionario:
  - `'bboxes'`: lista di bounding box `[x1, y1, x2, y2]` in pixel
  - `'labels'`: lista di label corrispondenti
- Visualizzazione: usare la funzione `plot_bbox(image, results['<OD>'])` (definita nel notebook) per disegnare rettangoli e label.

3) **Dense Region Caption**
- Token: `'<DENSE_REGION_CAPTION>'`.
- Output formato simile a OD: `{'bboxes': [...], 'labels': [...]}` dove `labels` sono caption per ogni regione.
- Visualizzazione: `plot_bbox(image, results['<DENSE_REGION_CAPTION>'])`.

4) **Region Proposal**
- Token: `'<REGION_PROPOSAL>'`.
- Output: dizionario con `'bboxes'` (proposte di regione) e `'labels'` (spesso vuote o placeholder).
- Visualizzazione: `plot_bbox(image, results['<REGION_PROPOSAL>'])`.

5) **Phrase Grounding (Caption -> Boxes)**
- Token: `'<CAPTION_TO_PHRASE_GROUNDING>'`.
- Input: richiede `text_input` con la frase da localizzare, oppure si possono passare come `text_input` i risultati del caption precedente.
- Esempio: 
  - Generare caption: `c = run_example('<CAPTION>')` e poi `text = c['<CAPTION>']`.
  - Grounding: `g = run_example('<CAPTION_TO_PHRASE_GROUNDING>', text_input=text)`.
- Output: `g['<CAPTION_TO_PHRASE_GROUNDING>']` con `'bboxes'` e `'labels'`.
- Visualizzazione: `plot_bbox(image, g['<CAPTION_TO_PHRASE_GROUNDING>'])`.

6) **Referring Expression Segmentation**
- Token: `'<REFERRING_EXPRESSION_SEGMENTATION>'`.
- Input: `text_input` con la espressione di riferimento (es. `"a woman."`).
- Output: `{'<REFERRING_EXPRESSION_SEGMENTATION>': {'polygons': [...], 'labels': [...]}}`.
- Visualizzazione: usare `draw_polygons(output_image, results['<REFERRING_EXPRESSION_SEGMENTATION>'], fill_mask=True)` per disegnare maschere/poligoni sull'immagine.

7) **Region To Segmentation**
- Token: `'<REGION_TO_SEGMENTATION>'`.
- Input: coordinate quantizzate nel formato `"<loc_x1><loc_y1><loc_x2><loc_y2>"` dove ogni valore è intero tra 0 e 999 (coordinata quantizzata usata nel notebook). Esempio: `"<loc_702><loc_575><loc_866><loc_772>"`.
- Output: struttura simile a segmentation con poligoni e labels.

8) **Open Vocabulary Detection**
- Token: `'<OPEN_VOCABULARY_DETECTION>'`.
- Input: `text_input` con la query semantica (es. `"a necklace"`).
- Output: dizionario che può contenere sia `'bboxes'` e `'bboxes_labels'` sia `'polygons'` e `'polygons_labels'`.
- Conversione per `plot_bbox`: usare la funzione `convert_to_od_format(data)` presente nel notebook per ottenere formato `{'bboxes': ..., 'labels': ...}` poi `plot_bbox(image, bbox_results)`.

9) **Region To Category / Region To Description**
- Token: `'<REGION_TO_CATEGORY>'` e `'<REGION_TO_DESCRIPTION>'`.
- Input: coordinate nella stessa forma quantizzata `"<loc_x1><loc_y1><loc_x2><loc_y2>"`.
- Output: testo descrittivo o categorie per la regione fornita.

10) **OCR / OCR_WITH_REGION**
- Token: `'<OCR>'` e `'<OCR_WITH_REGION>'`.
- Esecuzione: `run_example('<OCR>')` o `run_example('<OCR_WITH_REGION>')`.
- Output (`'<OCR_WITH_REGION>'`): `{'quad_boxes': [[x1,y1,...,x4,y4], ...], 'labels': ['text1', ...]}`.
- Visualizzazione: usare `draw_ocr_bboxes(output_image, results['<OCR_WITH_REGION>'])` (funzione definita nel notebook) che disegna i poligoni dei quadrilateri e scrive il testo.

11) **Task a cascata (es. Caption + Phrase Grounding)**
- Metodo: eseguire il primo task (es. `'<CAPTION>'`), prendere il testo generato e passarlo come `text_input` al task successivo (es. `'<CAPTION_TO_PHRASE_GROUNDING>'`).
- Esempio nel notebook:

```
results = run_example('<CAPTION>')
text_input = results['<CAPTION>']
results = run_example('<CAPTION_TO_PHRASE_GROUNDING>', text_input)
results['<CAPTION>'] = text_input
```

----
**Funzioni di visualizzazione presenti nel notebook**
- `plot_bbox(image, data)`: disegna rettangoli e label su `image` usando `data['bboxes']` e `data['labels']`.
- `draw_polygons(image, prediction, fill_mask=False)`: disegna poligoni e (opzionalmente) li riempie.
- `draw_ocr_bboxes(image, prediction)`: disegna box OCR in formato quadrilateri e stampa le etichette.

**Formato degli input speciali**
- `text_input` comune: stringhe libere (es. descrizioni) o stringhe contenenti tag di coordinate quantizzate del tipo `"<loc_702><loc_575><loc_866><loc_772>"` per task che richiedono regioni.

**Suggerimenti e troubleshooting**
- Se si verifica un errore CUDA, provare a rimuovere `.cuda()` o impostare `model.to('cpu')` e rimuovere i `.cuda()` nelle chiamate.
- Assicurarsi che `image.width` e `image.height` siano corretti quando `processor.post_process_generation` richiede `image_size`.
- Aumentare `num_beams` o `do_sample` nelle impostazioni di `model.generate()` per variare qualità e diversità.

----
Se vuoi, posso:
- Eseguire brevi esempi per ogni task (se hai GPU e accesso al modello),
- Aggiungere esempi di visualizzazione salvando le immagini risultanti su disco,
- Oppure estendere la guida con comandi di setup dettagliati.

File creato: [guida.md](guida.md)
