# Files in Google Drive/bg

-   00_dbourke_introduction_to_nlp_original.ipynb
-   [Wip_] Exp01_Split_sentences_wip.ipynb - load 1000 sentences, split using nlp and using space, which is faster

-   [????] 01 Experiment DB.ipynb
-   [????] 01_experiment_dbourke_nlp.ipynb
-   [Rev1] 03_Prepare-Data_Vocabulary_process.ipynb - read raw csv file with words, clean it, process it
-   [Rev1] 04_Prepare-Data_Sentences_clean.ipynb - read raw file with sentences from wikipedia and clean it, save to two csv files
-   [Rev2] 05_Prepare_Data_Sentences_process.ipynb - read clean file with sentences from wikipedia and add nlp tags using udpipe.
    Final file: {data_processed_dir}/sent_wikipedia_nlp_more_features_no_repeat.csv

# Files in Google Drive/bg/data/raw

-   wl_bul_wikipedia_2021_1M-sentences.txt - original file, later cleaned and split into two parts
-   wl_bul_wikipedia_2021_1M-sentences_part1.csv
-   wl_bul_wikipedia_2021_1M-sentences_part2.csv

# File in Google Drive/bg/data/clean

# Possible raw data

1. Folder: wortschatz_leipzig
   Source: https://wortschatz.uni-leipzig.de/en/download/Bulgarian
   news - има нецензурни коментари, не става
   newscrawl - има нецензурни коментари, не става
   wikipedia - става

2. Folder: Wiki1000+ Bulgarian Dataset
   files: words.tsv.zip: it is OK
   sents.tsv.zip: do not use, very unclean
   Source: https://www.kaggle.com/datasets/auhide/wiki1000-bulgarian-dataset
   Info: This is the same as Wiki1000.zip but in csv format

3. File: Bulgarian Part Of Speech Dataset_alt,
   Bulgarian Part Of Speech Dataset_raw.csv <-- use this. I split into two to be smaller than 100MB and renamed to bg_pos_raw_part1.csv and bg_pos_raw_part2.csv.
   Source: https://www.kaggle.com/datasets/auhide/bulgarian-part-of-speech-dataset

4. File: Random_Bulgarian_paragraphs.zip
   Source: https://www.kaggle.com/datasets/nikitricky/random-bulgarian-paragraphs?resource=download

5. bulgarian_articles_with_keywords.zip
   Source: https://www.kaggle.com/datasets/auhide/bulgarian-articles-with-keywords

6. File: Wiki1000.zip
   Source: https://dcl.bas.bg/wikiCorpus.html
   Info: XML?

7. Folder: chitanka
   Source: https://www.kaggle.com/datasets/nikitricky/the-ultimate-bulgarian-nlp-dataset?select=chitanka_info+chunks

8. File: EMP-BTB-CSLI-MWA.zip

9. File: bg.txt.xz

10. File bg_rc-v1.0.json
    Source: https://github.com/mhardalov/bg-reason-BERT/tree/master
    Paper: https://paperswithcode.com/paper/beyond-english-only-reading-comprehension

11. Списък с думи за скрейпване: https://slovored.com/sitemap/english, https://slovored.com/sitemap/grammar, https://slovored.com/sitemap/pravopisen-rechnik
    Списък с думи за скрейпване с апи: https://slovored.com/search/unilingual/%D0%B0%D0%B1%D1%81%D1%82%D1%80%D0%B0%D1%85%D0%B8%D1%80%D0%B0%D0%BC+%D1%81%D0%B5
