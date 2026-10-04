# 🎬 Movie Recommendation System

A **content-based movie recommendation system** built with Python and machine learning techniques.

The system recommends movies that are similar to a movie selected by the user. It analyzes information such as **genres, keywords, tagline, cast, and director**, converts that text into numerical vectors using **TF-IDF**, and then compares movies using **cosine similarity**.

> This project is implemented as a Jupyter Notebook and is intended as a practical introduction to building a recommendation system using Natural Language Processing (NLP) and machine learning techniques.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [How the System Works](#-how-the-system-works)
- [Recommendation Approach](#-recommendation-approach)
- [Technologies Used](#-technologies-used)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Running the Project](#-running-the-project)
- [Step-by-Step Implementation](#-step-by-step-implementation)
  - [1. Import Dependencies](#1-import-dependencies)
  - [2. Load the Dataset](#2-load-the-dataset)
  - [3. Explore the Dataset](#3-explore-the-dataset)
  - [4. Select Relevant Features](#4-select-relevant-features)
  - [5. Handle Missing Values](#5-handle-missing-values)
  - [6. Combine the Features](#6-combine-the-features)
  - [7. Convert Text to Numerical Vectors](#7-convert-text-to-numerical-vectors)
  - [8. Calculate Cosine Similarity](#8-calculate-cosine-similarity)
  - [9. Get the User's Movie](#9-get-the-users-movie)
  - [10. Find a Matching Movie Title](#10-find-a-matching-movie-title)
  - [11. Find the Movie Index](#11-find-the-movie-index)
  - [12. Retrieve Similarity Scores](#12-retrieve-similarity-scores)
  - [13. Sort Movies by Similarity](#13-sort-movies-by-similarity)
  - [14. Display Recommendations](#14-display-recommendations)
- [Complete Recommendation Pipeline](#-complete-recommendation-pipeline)
- [Example](#-example)
- [Understanding the Machine Learning Concepts](#-understanding-the-machine-learning-concepts)
- [Why TF-IDF?](#-why-tf-idf)
- [Why Cosine Similarity?](#-why-cosine-similarity)
- [What Makes This a Content-Based System?](#-what-makes-this-a-content-based-system)
- [Limitations](#-limitations)
- [Possible Improvements](#-possible-improvements)
- [Learning Outcomes](#-learning-outcomes)
- [Future Improvements](#-future-improvements)
- [Author](#-author)
- [License](#-license)

---

# 🎯 Project Overview

Choosing a movie to watch can sometimes be difficult because there are thousands of movies available.

This project solves a simple version of that problem:

> **"If I like this movie, what other movies might I also like?"**

Instead of relying on user ratings or information from other users, this system looks at the **content of each movie**.

For example, if a user enters:

```text
Spider-Man
```

the system analyzes the characteristics of _Spider-Man_ and finds other movies with similar characteristics.

The system can recommend movies based on similarities in:

- 🎭 Genres
- 🔑 Keywords
- 📝 Taglines
- 🎬 Cast
- 🎥 Director

The project uses **TF-IDF vectorization** to represent the textual information numerically and **cosine similarity** to measure how similar two movies are.

---

# 🧠 How the System Works

The recommendation pipeline can be summarized as:

```text
                    Movie Dataset
                         │
                         ▼
                Load CSV with Pandas
                         │
                         ▼
              Select Relevant Features
                         │
                         ▼
                Handle Missing Values
                         │
                         ▼
              Combine Textual Features
                         │
                         ▼
                  TF-IDF Vectorizer
                         │
                         ▼
              Movie Feature Vectors
                         │
                         ▼
               Cosine Similarity
                         │
                         ▼
               Similarity Matrix
                         │
                         ▼
              User Enters Movie Name
                         │
                         ▼
               Find Closest Title
                         │
                         ▼
                Find Movie Index
                         │
                         ▼
             Retrieve Similarity Scores
                         │
                         ▼
              Sort by Similarity Score
                         │
                         ▼
             Display Recommended Movies
```

The important idea is that the system converts each movie into a mathematical representation and then compares those representations.

---

# 🤖 Recommendation Approach

This project uses a **content-based filtering** approach.

Content-based recommendation means that recommendations are generated based on the characteristics of the item the user already likes.

In this project:

```text
User's movie
     ↓
Movie characteristics
     ↓
Text representation
     ↓
Numerical vector
     ↓
Compare with other movie vectors
     ↓
Most similar movies
```

The system does **not** require:

- User accounts
- User ratings
- Previous viewing history
- Other users
- Collaborative filtering

Instead, it uses the information contained in the movie dataset itself.

---

# 🛠 Technologies Used

The project uses the following technologies and libraries:

| Technology       | Purpose                             |
| ---------------- | ----------------------------------- |
| Python           | Main programming language           |
| Pandas           | Data loading and manipulation       |
| NumPy            | Numerical operations                |
| scikit-learn     | TF-IDF and cosine similarity        |
| difflib          | Finding close movie-title matches   |
| Jupyter Notebook | Interactive development environment |

### Main Python libraries

```python
import pandas as pd
import numpy as np
import difflib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```

---

# 📁 Project Structure

The repository currently contains:

```text
movie-recommendation-system/
│
├── Movie_Recommendation_System.ipynb
│
├── sample_data/
│   └── movies.csv
│
└── README.md
```

### `Movie_Recommendation_System.ipynb`

This is the main notebook containing the complete recommendation system.

It contains:

- Dataset loading
- Data exploration
- Feature selection
- Missing-value handling
- Feature combination
- TF-IDF vectorization
- Cosine similarity
- User input
- Movie-title matching
- Recommendation generation

### `sample_data/movies.csv`

This is the movie dataset used by the notebook.

The dataset contains:

- **4,803 movies**
- **24 columns**

Some of the important columns include:

```text
title
genres
keywords
tagline
cast
director
overview
budget
popularity
vote_average
vote_count
runtime
original_language
```

---

# 📊 Dataset

The project loads the dataset using:

```python
movies_data = pd.read_csv("./sample_data/movies.csv")
```

The dataset contains 4,803 movie records.

The notebook verifies this using:

```python
movies_data.shape
```

which produces:

```text
(4803, 24)
```

This means:

```text
4,803 rows
24 columns
```

Each row represents one movie.

---

# ⚙️ Installation

## 1. Clone the Repository

Clone the project to your local machine:

```bash
git clone https://github.com/chiscookeke11/movie-recommendation-system.git
```

Move into the project directory:

```bash
cd movie-recommendation-system
```

---

## 2. Create a Virtual Environment

It is recommended to create a virtual environment so that the project's dependencies remain isolated from other Python projects.

### Windows

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

After activation, your terminal should indicate that the virtual environment is active.

---

# 📦 3. Install Dependencies

Install the required libraries:

```bash
pip install pandas numpy scikit-learn jupyter
```

The project uses Python 3.11 in the notebook environment, so **Python 3.11 is recommended**.

You can verify your Python version with:

```bash
python --version
```

---

# ▶️ Running the Project

Start Jupyter Notebook:

```bash
jupyter notebook
```

This should open Jupyter Notebook in your browser.

Open:

```text
Movie_Recommendation_System.ipynb
```

Then run the notebook cells from top to bottom.

You can also use JupyterLab:

```bash
jupyter lab
```

---

# 🔬 Step-by-Step Implementation

## 1. Import Dependencies

The first step is importing the libraries required by the project.

```python
import pandas as pd
import numpy as np
import difflib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```

### Pandas

Pandas is used to work with the movie dataset.

```python
import pandas as pd
```

It provides the `DataFrame` structure used throughout the project.

---

### NumPy

NumPy provides numerical computing functionality.

```python
import numpy as np
```

Although the project does not heavily rely on NumPy directly, it is included as part of the numerical-processing environment.

---

### difflib

Python's built-in `difflib` module is used to find movie titles that closely match what the user enters.

```python
import difflib
```

This is useful because users may not type a movie title perfectly.

For example:

```text
User enters:
spiderman
```

while the dataset contains:

```text
Spider-Man
```

The system can attempt to find the closest matching title.

---

### TfidfVectorizer

The `TfidfVectorizer` converts textual movie information into numerical vectors.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
```

Machine-learning algorithms generally need numerical representations, so we need to transform movie text into numbers.

---

### cosine_similarity

This function measures the similarity between movie vectors.

```python
from sklearn.metrics.pairwise import cosine_similarity
```

The result is a similarity score between movies.

---

# 2. Load the Dataset

The movie dataset is loaded into a Pandas DataFrame:

```python
movies_data = pd.read_csv("./sample_data/movies.csv")
```

The path:

```text
./sample_data/movies.csv
```

means:

```text
current project directory
        ↓
sample_data
        ↓
movies.csv
```

Once loaded, the dataset is stored in:

```python
movies_data
```

---

# 3. Explore the Dataset

The notebook performs basic data exploration.

### View the first five rows

```python
movies_data.head()
```

This helps us understand what the dataset looks like.

---

### View the last five rows

```python
movies_data.tail()
```

This allows us to inspect the end of the dataset.

---

### Check the dataset dimensions

```python
movies_data.shape
```

The project produces:

```text
(4803, 24)
```

Therefore, there are:

```text
4,803 movies
24 features/columns
```

---

# 4. Select Relevant Features

The dataset contains many columns, but not every column is useful for determining whether two movies are similar.

The project selects five textual features:

```python
selected_features = [
    "genres",
    "keywords",
    "tagline",
    "cast",
    "director"
]
```

These features were selected because they contain useful information about the content and identity of a movie.

### Genres

Examples:

```text
Action
Comedy
Drama
Science Fiction
Adventure
```

---

### Keywords

Keywords describe important concepts associated with the movie.

For example:

```text
space
war
future
secret agent
crime
superhero
```

---

### Tagline

A tagline is a short phrase associated with a movie.

For example:

```text
The Legend Ends
```

Taglines can provide additional textual information about a movie.

---

### Cast

The cast provides information about the actors appearing in a movie.

For example:

```text
Christian Bale
Michael Caine
Gary Oldman
```

Movies sharing actors can therefore receive higher similarity scores.

---

### Director

The director is another useful characteristic.

For example:

```text
Christopher Nolan
```

Movies directed by the same person can have similarities in style or themes.

---

# 5. Handle Missing Values

Real-world datasets frequently contain missing values.

For example, a movie might not have:

- A tagline
- Keywords
- A director
- Cast information

The project handles missing values with:

```python
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')
```

Instead of leaving a missing value as:

```text
NaN
```

it is replaced with:

```text
''
```

an empty string.

This is important because the next step combines the text fields.

We want:

```text
text + text + text
```

rather than:

```text
text + NaN + text
```

---

# 6. Combine the Features

The five selected features are combined into one large text representation for every movie.

```python
combined_features = (
    movies_data["genres"]
    + ' '
    + movies_data["keywords"]
    + ' '
    + movies_data["tagline"]
    + ' '
    + movies_data["cast"]
    + ' '
    + movies_data["director"]
)
```

For example, a movie might become something conceptually similar to:

```text
Action Adventure Science Fiction
space war future
Enter the World of Pandora
Sam Worthington Zoe Saldana Sigourney Weaver
James Cameron
```

The system now has one textual representation for every movie.

This is important because TF-IDF will operate on this combined text.

---

# 7. Convert Text to Numerical Vectors

Computers cannot directly calculate similarity between raw text in the way we need.

We therefore convert the text into numerical vectors.

First, we create a TF-IDF vectorizer:

```python
vectorizer = TfidfVectorizer()
```

Then we fit it to the combined movie information:

```python
feature_vectors = vectorizer.fit_transform(combined_features)
```

The result is a matrix containing a numerical representation of every movie.

Conceptually:

```text
Movie 1 → [0.00, 0.21, 0.00, 0.72, ...]
Movie 2 → [0.14, 0.00, 0.32, 0.00, ...]
Movie 3 → [0.00, 0.41, 0.11, 0.08, ...]
...
```

Each movie becomes a point in a high-dimensional mathematical space.

---

# 8. Calculate Cosine Similarity

Once every movie has been converted into a vector, the system compares the vectors.

```python
similarity = cosine_similarity(feature_vectors)
```

This creates a similarity matrix.

Because there are 4,803 movies, the resulting matrix is:

```text
(4803, 4803)
```

That means every movie can be compared against every other movie.

Conceptually:

```text
              Movie 1  Movie 2  Movie 3  Movie 4
Movie 1          1.0     0.2      0.4      0.1
Movie 2          0.2     1.0      0.3      0.7
Movie 3          0.4     0.3      1.0      0.5
Movie 4          0.1     0.7      0.5      1.0
```

A value closer to `1` indicates greater similarity.

A value closer to `0` indicates little or no similarity.

Every movie has a similarity score against every other movie.

---

# 9. Get the User's Movie

The user is asked to enter their favorite movie:

```python
movie_name = input('Enter your favourite movie name:')
```

For example:

```text
Enter your favourite movie name: Spider-Man
```

The value entered by the user is stored in:

```python
movie_name
```

---

# 10. Find a Matching Movie Title

The system first creates a list containing all movie titles:

```python
list_of_all_titles = movies_data['title'].tolist()
```

The result is a Python list containing titles such as:

```text
Avatar
Spectre
Spider-Man
The Dark Knight Rises
...
```

The project then uses `difflib.get_close_matches()`:

```python
find_close_match = difflib.get_close_matches(
    movie_name,
    list_of_all_titles
)
```

This attempts to find a title in the dataset that closely matches the user's input.

For example:

```text
User:
Spider Man
```

could potentially match:

```text
Spider-Man
```

This makes the system more user-friendly than requiring an exact string match.

---

# 11. Find the Movie Index

The closest matching title is selected:

```python
close_match = find_close_match[0]
```

The system then finds the movie's index:

```python
index_of_the_movie = (
    movies_data[movies_data.title == close_match]['index'].values[0]
)
```

For example, if:

```text
close_match = Spider-Man
```

the system retrieves the corresponding dataset index.

This index is important because it tells us which row in the similarity matrix belongs to the selected movie.

---

# 12. Retrieve Similarity Scores

The system retrieves the similarity scores for the selected movie:

```python
similarity_score = list(
    enumerate(similarity[index_of_the_movie])
)
```

The result looks conceptually like:

```text
[
    (0, 0.058),
    (1, 0.028),
    (2, 0.027),
    ...
]
```

Each tuple contains:

```text
(movie_index, similarity_score)
```

For example:

```text
(30, 0.3179)
```

means that movie index `30` has a similarity score of approximately:

```text
0.3179
```

with the selected movie.

---

# 13. Sort Movies by Similarity

The system sorts the movies from the most similar to the least similar:

```python
sorted_similar_movies = sorted(
    similarity_score,
    key=lambda x: x[1],
    reverse=True
)
```

The important part is:

```python
key=lambda x: x[1]
```

because:

```python
x[0]
```

is the movie index, while:

```python
x[1]
```

is the similarity score.

`reverse=True` means the highest similarity scores come first.

Therefore:

```text
1.0
0.8
0.7
0.6
0.5
...
```

will appear before:

```text
0.1
0.05
0.01
```

---

# 14. Display Recommendations

Finally, the system loops through the sorted movies:

```python
print('Movies suggested for you: \n')

i = 1

for movie in sorted_similar_movies:
    index = movie[0]

    title_from_index = (
        movies_data[movies_data.index == index]['title'].values[0]
    )

    if (i < 30):
        print(i, '.', title_from_index)
        i += 1
```

The system:

1. Gets the movie index.
2. Uses that index to find the movie title.
3. Prints the title.
4. Continues until the first 29 results have been displayed.

The current implementation therefore displays up to **29 movie suggestions**.

---

# 🔄 Complete Recommendation Pipeline

The entire algorithm can be represented as:

```text
                ┌──────────────────┐
                │   movies.csv     │
                └────────┬─────────┘
                         │
                         ▼
                Load with Pandas
                         │
                         ▼
              Select five features
                         │
                         ▼
           Fill missing values with ""
                         │
                         ▼
              Combine text fields
                         │
                         ▼
                 TF-IDF Vectorizer
                         │
                         ▼
               Movie feature vectors
                         │
                         ▼
               Cosine similarity
                         │
                         ▼
             4803 × 4803 matrix
                         │
                         ▼
                User enters movie
                         │
                         ▼
              Find closest title
                         │
                         ▼
                Find movie index
                         │
                         ▼
          Get corresponding similarity
                         │
                         ▼
             Sort by similarity
                         │
                         ▼
             Display recommendations
```

---

# 🎬 Example

Suppose the user enters:

```text
Spider-Man
```

The system searches the dataset for the closest matching title.

It then calculates the similarity between _Spider-Man_ and all other movies.

An example output from the notebook is:

```text
Movies suggested for you:

1 . Spider-Man
2 . Spider-Man 3
3 . Spider-Man 2
4 . The Notebook
5 . Seabiscuit
6 . Clerks II
7 . The Ice Storm
8 . Oz: The Great and Powerful
9 . Horrible Bosses
10 . The Count of Monte Cristo
...
```

The first result is the selected movie itself because a movie has a cosine similarity of `1.0` with its own vector.

The remaining results are movies ranked according to their calculated similarity scores.

---

# 🧮 Understanding the Machine Learning Concepts

## What is TF-IDF?

TF-IDF stands for:

> **Term Frequency-Inverse Document Frequency**

It is a technique used to determine how important a word is within a document compared with a collection of documents.

In this project:

```text
Document = Movie
```

and the words come from:

```text
genres
keywords
tagline
cast
director
```

---

## Term Frequency

Term Frequency measures how frequently a term appears in a document.

If a particular word appears several times in a movie's text representation, it can receive greater importance.

---

## Inverse Document Frequency

Inverse Document Frequency reduces the importance of words that appear across many documents.

For example, if a word appears in almost every movie, it does not help distinguish one movie from another.

TF-IDF therefore gives more importance to words that are useful for distinguishing documents.

---

# 📐 Why Cosine Similarity?

After TF-IDF converts each movie into a vector, we need a way to compare those vectors.

Cosine similarity measures the angle between two vectors.

Conceptually:

```text
              Vector B
             /
            /
           / θ
          /
---------/-------------- Vector A
```

If two vectors point in almost the same direction:

```text
Cosine similarity ≈ 1
```

If they are very different:

```text
Cosine similarity ≈ 0
```

This makes cosine similarity useful for comparing text documents.

---

# 🎯 What Makes This a Content-Based System?

The recommendation is based on the **content/features of the movies**.

For example, imagine two movies have:

```text
Genre:
Action Adventure

Keywords:
superhero crime city

Cast:
Actor A Actor B

Director:
Director X
```

Their textual representations may be similar.

The system can therefore give them a relatively high similarity score.

This is different from collaborative filtering.

### Content-based filtering

```text
Movie characteristics
        ↓
Similarity
        ↓
Recommendation
```

### Collaborative filtering

```text
User A liked Movie X
User B liked Movie X
User B also liked Movie Y
        ↓
Recommend Movie Y to User A
```

This project uses the first approach.

---

# ⚠️ Limitations

Although this project demonstrates the core ideas behind a recommendation system, there are several limitations.

## 1. It does not learn from user behavior

The system does not know:

- What the user watched previously
- What the user rated
- What the user skipped
- What other users liked

Every recommendation is based only on movie metadata.

---

## 2. Recommendations depend on the available metadata

If a movie has poor or incomplete:

- genre information
- keywords
- cast
- director
- tagline

its recommendations may not be very accurate.

---

## 3. Similarity does not mean quality

A high similarity score means:

> "These movies have similar textual features."

It does **not necessarily mean**:

> "The user will enjoy this movie."

Personal taste is more complicated than textual similarity.

---

## 4. The selected movie appears in the recommendations

Because a movie has perfect similarity with itself:

```text
similarity = 1.0
```

the selected movie is normally ranked first.

A production system would usually remove the selected movie from the recommendation list.

---

## 5. The similarity matrix can become expensive

The project calculates pairwise similarity for all 4,803 movies.

The resulting matrix has:

```text
4803 × 4803
```

entries.

For a much larger dataset, storing and calculating a full pairwise similarity matrix can become expensive.

---

## 6. Movie-title matching can fail

The code currently assumes that:

```python
find_close_match[0]
```

exists.

If `difflib` cannot find a match, the list may be empty, which can cause an error.

A production implementation should check whether a match exists before accessing index `0`.

---

# 🚀 Possible Improvements

There are several ways this project could be improved.

## 1. Remove the selected movie

Instead of showing:

```text
1. Spider-Man
2. Spider-Man 3
3. Spider-Man 2
```

the system could start with:

```text
1. Spider-Man 3
2. Spider-Man 2
3. ...
```

by excluding the selected movie from the recommendations.

---

## 2. Improve title matching

The system currently uses:

```python
difflib.get_close_matches()
```

A more robust system could:

- Normalize capitalization
- Remove punctuation
- Handle alternate titles
- Display multiple possible matches
- Ask the user to select the intended movie

---

## 3. Build a user interface

The notebook could be converted into an application using tools such as:

- Streamlit
- Flask
- FastAPI
- Django

For example:

```text
┌─────────────────────────────────┐
│      Movie Recommendation       │
│                                 │
│  Enter a movie:                 │
│  [ Spider-Man              ]    │
│                                 │
│       [ Recommend ]             │
└─────────────────────────────────┘
```

The recommendations could then be displayed as movie cards.

---

## 4. Add Movie Posters

A movie API could be used to retrieve:

- Poster images
- Release dates
- Ratings
- Descriptions
- Genres
- Trailers

This would make the recommendation system much more useful as an actual application.

---

## 5. Use More Movie Features

The current system uses:

```text
genres
keywords
tagline
cast
director
```

It could potentially incorporate additional information such as:

```text
overview
original language
production companies
release year
runtime
```

The `overview` field in particular could provide valuable semantic information.

---

## 6. Add Recommendation Scores

Instead of only displaying:

```text
Spider-Man 3
Spider-Man 2
```

the system could show:

```text
Spider-Man 3 — 31.8% similarity
Spider-Man 2 — 31.2% similarity
```

This would make the ranking more transparent.

---

## 7. Build a Hybrid Recommendation System

A more advanced version could combine:

### Content-based filtering

```text
Movie metadata
```

with:

### Collaborative filtering

```text
User ratings
User behavior
```

The resulting system could use both:

```text
Movie characteristics
        +
User preferences
        ↓
Hybrid recommendation
```

This would generally provide more personalized recommendations.

---

## 8. Deploy the Application

After converting the notebook into an application, it could be deployed using platforms such as:

- Streamlit Community Cloud
- Render
- Railway
- Hugging Face Spaces
- Vercel + API backend

A deployed version would allow users to interact with the recommendation system through a web browser.

---

# 📚 Learning Outcomes

This project demonstrates several important concepts in Python, machine learning, and NLP.

By studying this project, you can learn how to:

- Load datasets with Pandas
- Explore tabular data
- Handle missing values
- Select relevant features
- Work with textual data
- Combine multiple text features
- Convert text into numerical representations
- Use TF-IDF
- Calculate cosine similarity
- Work with similarity matrices
- Perform fuzzy string matching
- Sort recommendation results
- Build a basic content-based recommendation system

---

# 🧑‍💻 Project Workflow

The project follows a common machine-learning workflow:

```text
1. Collect Data
       ↓
2. Explore Data
       ↓
3. Select Features
       ↓
4. Clean Data
       ↓
5. Transform Data
       ↓
6. Generate Features
       ↓
7. Calculate Similarity
       ↓
8. Generate Recommendations
       ↓
9. Display Results
```

This workflow is applicable to many other machine-learning problems.

---

# 🔮 Future Improvements

Potential future versions of this project could include:

- [ ] Streamlit web interface
- [ ] Movie posters
- [ ] Movie descriptions
- [ ] Genre filtering
- [ ] Top-rated movie filtering
- [ ] Better title matching
- [ ] Exclude the selected movie from results
- [ ] Recommendation confidence/similarity scores
- [ ] Movie search autocomplete
- [ ] User ratings
- [ ] User profiles
- [ ] Collaborative filtering
- [ ] Hybrid recommendation
- [ ] REST API
- [ ] Model/vector persistence
- [ ] Cloud deployment

---

# 📂 Repository

You can find the complete project here:

[Movie Recommendation System on GitHub](https://github.com/chiscookeke11/movie-recommendation-system?utm_source=chatgpt.com)

---

# 👨🏽‍💻 Author

**Chinedu Okeke**

Software Engineer | AI/ML Engineer | Open Source Contributor

GitHub: [@chiscookeke11](https://github.com/chiscookeke11)

---

# 📄 License

This project is intended primarily for learning and experimentation.

If you reuse or extend the project, please review the licensing and attribution requirements associated with the dataset and any external resources used.

---

## ⭐ If You Found This Useful

If this project helped you understand recommendation systems, machine learning, NLP, or TF-IDF, consider giving the repository a ⭐ on GitHub.

Contributions, improvements, and suggestions are welcome.
