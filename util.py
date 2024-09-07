import streamlit as st
import matplotlib.pyplot as plt


def word_embedding_visualization(vectors_pca, word_list):
    fig, axes = plt.subplots(1, 1, figsize=(3, 3))

    # Scatter plot
    axes.scatter(vectors_pca[:, 0], vectors_pca[:, 1])

    # Annotating each point with its corresponding word
    for i, word in enumerate(word_list):
        axes.annotate(word, (vectors_pca[i, 0] + .02, vectors_pca[i, 1] + .02))

    # Set title
    axes.set_title('Word Embeddings')

    # Display the plot in Streamlit
    st.pyplot(fig)


def display_footer():
    footer = """
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: transparent;
            text-align: center;
            color: grey;
            padding: 10px 0;
        }
        </style>
        <div class="footer">
            Made with ❤️ by <a href="mailto:zeeshan.altaf@92labs.ai">Zeeshan</a>.
            Source code <a href='https://github.com/mzeeshanaltaf/visualize-word-embeddings'>here</a>.</div> 
        </div>
    """
    st.markdown(footer, unsafe_allow_html=True)