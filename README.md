# Thesis

In this paper, we implement a semantic search engine using various algorithms such as
Word2Vec, fastText and TF-IDF.
These algorithms assist with converting words into valued vectors, called embeddings.
With these embeddings, we can perform different applications, such as document retrieving.
Converting words in vectors relies on the foundation of word embedding. This is a term
used to describe words in vectors which capture various meanings of a word. Using this
approach, we maintain the semantics of the words, and we can process them in a more
comprehensive manner.
The search engine performs search queries on websites. Therefore, we integrate a Chrome
browser plugin and a web server to assist with extracting contents from the websites.
For the document retrieval task, we use the cosine similarity metric which provides a
value to determine the similarity between two documents.
Before applying these algorithms, the extracted texts need to be cleaned and prepared for
the search functionality. Therefore, we consider splitting the text into tokens (words) and
removing numerals, stop words and non-alphabet characters. Also, we transform words
to the root form with a method called lemmatization, and we remove any redundant tokens.
For the evaluation of the search engine, we use a dataset from BBC news. For the documents in the BBC news, we generate queries using TF-IDF. This method extract keywords
from the documents that contribute to the context.
In later chapters of this paper, we consider replacing the Word2Vec model with the fastText model to evaluate the differences in the performance.
