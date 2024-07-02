# Reimagining Intent Prediction: Insights from Graph-Based Dialogue Modeling and Sentence Encoders

Authors: Daria Ledneva and Denis Kusnetsov.

LREC-Coling 2024. Paper: https://aclanthology.org/2024.lrec-main.1208

## Overview
The repository presents the implementation of a novel approach designed specifically for closed-domain dialogue systems. The method, as detailed in the accompanying research paper, uses scenario dialogue graphs to address the challenges inherent in closed-domain dialogue systems, where a nuanced understanding of context is essential. The focus is on achieving accurate intent prediction, representing a significant advance in dialogue system capabilities.

![image](https://github.com/LadaNikitina/Dialog-Graph-Intent-Prediction/assets/23546579/f4c4b96d-dfae-4181-bfbb-58af8b92ca39)

## Installation

Clone the repository:

```git clone https://github.com/LadaNikitina/Dialog-Graph-Intent-Prediction```

Install dependencies:

```pip install -r requirements.txt```

## Datasets

We use 3 public closed-domain datasets:

```
1. MultiWOZ
2. FoCus
3. Taskmaster
```

Additionally, we utilise 2 public open-domain datasets:

```
1. PersonaChat
2. DailyDialog
```

## Citation

```
@inproceedings{ledneva-kuznetsov-2024-reimagining-intent,
    title = "Reimagining Intent Prediction: Insights from Graph-Based Dialogue Modeling and Sentence Encoders",
    author = "Ledneva, Daria Romanovna  and
      Kuznetsov, Denis Pavlovich",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.1208",
    pages = "13847--13860",
}
```
