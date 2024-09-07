import numpy as np
from numpy import triu
from scipy.linalg import get_blas_funcs
from sklearn.decomposition import PCA
import gensim.downloader as api
from util import *

# Initialize streamlit app
page_title = "Visualizing Word Embeddings"
page_icon = "✨️"
st.set_page_config(page_title=page_title, page_icon=page_icon, layout="centered")

# Application title
st.title(page_title)
st.write(':blue[***Visualize the Semantic Universe of Words 🧠🌍***]')
st.write("Explore the fascinating world of word embeddings! Input words and see how they align in vector space. The "
         "closer they are, the more similar they are in meaning. Unlock insights into language relationships with this "
         "interactive tool! 🔍📊")

st.subheader('Enter Words:')
words = st.text_input("Enter at least two words to visualize them. Words should be comma separated.",
                      value='king, queen')
# Split the string by commas and remove any extra spaces
word_list = [word.strip().lower() for word in words.split(',') if word]
if len(word_list) < 2:
    st.error('Enter at least two words')
else:
    st.subheader('Visualize:')
    visualize = st.button('Visualize')
    if visualize:
        with st.spinner('Processing ...'):
            # Load the GloVe word Embeddings
            word_vectors = api.load('glove-wiki-gigaword-100')

            # Get word vectors
            vectors = np.array([word_vectors[word] for word in word_list])

            # Reduce dimensions using PCA
            pca = PCA(n_components=2)
            vectors_pca = pca.fit_transform(vectors)

            # Display word embedding visualization
            word_embedding_visualization(vectors_pca, word_list)

display_footer()
