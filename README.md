# SVD Search Engine

A simple document search engine (based on the Wikipedia dataset) that allows comparing different text representation and retrieval techniques. This project was prepared for the MOWNiT laboratory.

## Features

- **Side-by-Side Comparison**: Two result panels allow for simultaneous testing of different settings for the same query.
- **Search Methods**:
  - **Bag of Words (BoW)**: Simple word occurrence counting (CountVectorizer).
  - **TF-IDF**: Accounting for word rarity across the corpus (Inverse Document Frequency).
  - **SVD (LSA)**: Noise reduction and low-rank approximation using SVD decomposition (Latent Semantic Analysis).
- **Parameterization**: Ability to change the number of `k` components for the SVD method.
- **Preprocessing**: Stemming (Snowball), stop-words removal, and tokenization.

## Installation

1. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

2. Download data and build the index (this may take a few minutes as it downloads the dataset from HuggingFace):
   ```bash
   python data_builder.py
   ```
   This script will generate matrices and models in the `models/` folder.

## Running the Application

To start the web application, run:
```bash
python app.py
```
The application will be available at: [http://localhost:5000](http://localhost:5000)

## What can you test?

1. **TF-IDF vs BoW**: See how IDF weights improve search results.
2. **SVD (Noise Reduction)**: Compare standard TF-IDF with its dimensionally reduced version. SVD helps find semantically related documents even if they don't contain the exact same words as the query.
3. **Impact of the k parameter**: Change the number of components `k` in SVD. Smaller `k` results in more "blurred" topics (greater generalization), while larger `k` is closer to the TF-IDF matrix.
