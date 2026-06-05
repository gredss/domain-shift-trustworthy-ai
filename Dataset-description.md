# Indonesian News Dataset

## Overview

This dataset contains 5,000 Indonesian news articles collected from five major Indonesian news sources: CNN Indonesia, Kompas, Detik, Tempo, and Tribun News. The articles are categorized into five distinct domains, with 1,000 articles per domain, covering news published between 2022 and 2026.

## Dataset Structure

The dataset consists of five CSV files, one for each domain:

1. indonesian_tech_dataset.csv (1,000 rows)
2. indonesian_politics_dataset.csv (1,000 rows)
3. indonesian_health_dataset.csv (1,000 rows)
4. indonesian_sport_dataset.csv (1,000 rows)
5. indonesian_education_dataset.csv (1,000 rows)

Each CSV file contains the following columns:

- id: Unique identifier for each article
- source: News source name
- date: Publication date (YYYY-MM-DD format)
- title: Article headline
- url: Direct link to the original article
- article_text: Full article content
- label: Empty field for annotation purposes

## Domain Coverage

Technology: Articles covering gadgets, smartphones, applications, artificial intelligence, cybersecurity, startups, fintech, e-commerce, gaming, telecommunications, and digital innovation.

Politics: Articles about national politics, elections, government policies, political parties, parliamentary activities, regional governance, political leaders, and democratic processes.

Health: Articles discussing diseases, medical treatments, healthcare systems, public health, nutrition, mental health, maternal and child health, vaccines, and medical research.

Sport: Articles covering football, badminton, basketball, volleyball, athletics, martial arts, esports, national and international competitions, athlete achievements, and sports events.

Education: Articles about schools, universities, curriculum, examinations, scholarships, educational policies, student activities, learning methods, vocational training, and academic research.

## Data Collection

Articles were collected through automated web scraping of official news sitemaps. The collection process employed comprehensive Indonesian keyword filtering to ensure domain relevance and content quality. Each article includes the complete text extracted from the original source, maintaining the integrity of the journalistic content.

## Use Cases

This dataset is suitable for various natural language processing tasks including text classification, sentiment analysis, topic modeling, named entity recognition, and Indonesian language model training. The multi-domain structure enables comparative analysis across different news categories and supports research in Indonesian computational linguistics.

## Data Quality

All articles contain substantial content (minimum 150 characters) and have been deduplicated based on URLs and titles. The dataset represents authentic journalistic content from reputable Indonesian news organizations, providing reliable data for academic and research purposes.
