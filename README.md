# Fashion-Product-Recommendation-using-Multimodal



# Introduction

This repository implements a multimodal recommender system for fashion products, combining the power of image and text features to suggest visually and conceptually similar items. By leveraging deep learning techniques for image analysis and natural language processing (NLP) for text processing, the system provides a richer understanding of product attributes, leading to more personalized recommendations for users.

# System Overview

The system is comprised of the following key components:

* **Image Feature Extraction:** A pre-trained convolutional neural network (CNN), such as ResNet50, is utilized to extract visual features from product images. These features capture the essential characteristics of the clothing, including colors, patterns, styles, and overall composition.
* **Text Feature Extraction:** NLP techniques are applied to product descriptions. Two approaches are explored in this implementation:
    * **TF-IDF:** This method transforms textual descriptions into numerical vectors, highlighting the importance of words within the product category.
    * **BERT Embeddings:** This method utilizes a pre-trained BERT model to capture deeper semantic relationships within product descriptions, going beyond simple word frequency.
* **Feature Concatenation:** The extracted image and text features are combined to create a comprehensive representation of each product, encompassing both visual and textual information.
* **Similarity Matching:** Cosine similarity is employed to measure the similarity between the combined features of the selected product and those of all other products in the dataset. Products with the highest similarity scores are considered the most relevant recommendations.

# Example
Sample Product Details:
        id gender masterCategory subCategory articleType baseColour season  \
4704  7909    Men        Apparel     Topwear     Tshirts  Navy Blue   Fall   

      year   usage                             productDisplayName  \
4704  2011  Casual  Proline Men Navy & Cream Striped Polo T-shirt   

                             cleaned_text     image  
4704  proline men navy cream striped polo  7909.jpg  

Recommended images
![image](https://github.com/SundharessB/Fashion-Product-Recommendation-using-Multimodal/assets/139948283/05a0deb9-b1f0-49f8-9a51-97c364b38184)


# Usage

This code requires Python 3 and several popular libraries, including:

* TensorFlow/Keras
* NumPy
* Pandas
* Matplotlib
* Scikit-learn (optional, for TF-IDF)
* Transformers (optional, for BERT embeddings)

**1. Installation:**

Ensure you have the necessary libraries installed. You can use package managers like `pip` for installation:

```bash
pip install tensorflow keras numpy pandas matplotlib scikit-learn transformers
```

**2. Data Preparation:**

* Replace the placeholder path `path = '../input/fashion-product-images-dataset/fashion-dataset/'` with the actual path to your dataset containing product images and a CSV file with product descriptions (e.g., `styles.csv`).
* Ensure the image folder structure aligns with the file names in the CSV (`id.jpg` format expected).

**3. Execution:**

Run the main script (`main.py`) to execute the entire system. This script performs the following steps:

* **Data Loading and Preprocessing:** Loads the product descriptions from the CSV file and the images from the corresponding folder.
* **Image Feature Extraction:** Extracts visual features from the product images using the pre-trained CNN.
* **Text Feature Extraction:**
    * (Optional) TF-IDF: Converts textual descriptions into numerical vectors.
    * (Optional) BERT Embeddings: Processes descriptions using the pre-trained BERT model to obtain semantic embeddings.
* **Feature Concatenation:** Combines the extracted image and text features.
* **Similarity Matching:** Calculates cosine similarity between the chosen product and all other products based on the combined features.
* **Recommendation Display:** Shows the top few most similar products as recommendations.

**4. Customization:**

* You can experiment with different CNN architectures for image feature extraction (e.g., VGG16).
* Fine-tuning the chosen CNN on your specific fashion dataset may further improve recommendations.
* Explore more advanced text preprocessing techniques for potentially better text feature extraction.
* Consider incorporating additional data sources (e.g., user reviews, brand information) to further enrich the product representations.

# Conclusion

This multimodal recommender system provides a foundation for building personalized fashion product recommendations. By combining image and text analysis, the system delivers a more comprehensive understanding of user preferences and product attributes, ultimately leading to more relevant and engaging user experiences.

# Further Enhancements

* Integration with a user interface to allow interactive product selection and recommendation display.
* Incorporation of user feedback mechanisms to refine recommendation accuracy over time.
* Exploration of hybrid deep learning models for joint image and text feature extraction.


