import streamlit as st
import requests
from bs4 import BeautifulSoup
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from PIL import Image
from io import BytesIO

# Function to scrape images and descriptions from all pages of the provided link
def scrape_images_and_descriptions(link, num_pages):
    all_images = []
    all_descriptions = []

    for page in range(1, num_pages + 1):
        page_link = f"{link}?page={page}"
        response = requests.get(page_link)
        soup = BeautifulSoup(response.text, 'html.parser')

        images = [img['src'] for img in soup.find_all('img')]
        descriptions = [desc.text for desc in soup.find_all('div', class_='short-description')]

        all_images.extend(images)
        all_descriptions.extend(descriptions)

    return all_images, all_descriptions

# Function to perform clustering
def perform_clustering(images, descriptions, num_clusters):
    # Use TF-IDF vectorizer for text data
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(descriptions)

    # Use K-Means for clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(tfidf_matrix)

    return kmeans.labels_

# Streamlit web app
def main():
    st.title("Product Image Clustering App")

    # Input field for the customer to provide a link
    link = st.text_input("Enter the website link:")

    # Input field for the number of clusters
    num_clusters = st.number_input("Enter the number of clusters:", min_value=2, value=2)

    # Input field for the number of pages
    num_pages = st.number_input("Enter the number of pages:", min_value=1, value=1)

    if st.button("Cluster Images"):
        # Scrape images and descriptions
        images, descriptions = scrape_images_and_descriptions(link, num_pages)

        if images and descriptions:
            # Perform clustering
            labels = perform_clustering(images, descriptions, num_clusters)

            # Display the clustered images
            for cluster_num in range(num_clusters):
                cluster_indices = np.where(labels == cluster_num)[0]
                cluster_images = [Image.open(BytesIO(requests.get(images[i]).content)) for i in cluster_indices]

                # Display the cluster number
                st.subheader(f"Cluster {cluster_num + 1}:")

                # Display the images in the cluster
                st.image(cluster_images, width=200)
        else:
            st.warning("No images or descriptions found. Please check the provided link.")

if __name__ == "__main__":
    main()





# fun= download_images("https://web-scraping.dev/products")
# fun1= extract_text_descriptions(fun)
# print(fun)
# print(fun1)