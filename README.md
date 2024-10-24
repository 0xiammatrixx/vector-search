# Dynamic Art Curation Assistant

## Project Overview

This project is a **Dynamic Art Curation Assistant** that provides a real-time search experience for artworks, along with a personalized feed. It combines features like search, image recommendations, real-time feedback, and the ability to refine image suggestions based on user interactions. Users can search for art styles, view related images, upvote or downvote their preferences, and receive personalized results based on their feedback. Additional features include fetching descriptions, generating audio via text-to-speech (TTS), and providing related YouTube videos.

### Project Flow:
1. **Input Query**: The user enters a search query (e.g., "cubism" or "surrealism").
2. **Web Image Retrieval**: The system uses SerpAPI to fetch images related to the search query.
3. **Image Processing**: The CLIP model processes and generates embeddings for the retrieved images.
4. **Feedback Interaction**: Users upvote/downvote images, and the system refines results based on this feedback.
5. **Personalized Feed**: After a threshold of feedback is reached, the system provides personalized art recommendations based on upvoted images.
6. **Auxiliary Features**:
   - Fetching a description of the query from Wikipedia.
   - Providing a TTS-generated audio summary.
   - Displaying a related YouTube video.

## Features
- **Real-time Art Search**: Users can search for artworks and instantly receive image recommendations.
- **Dynamic Feedback System**: Upvoting or downvoting images influences future recommendations, offering a personalized feed.
- **Multi-modal Output**: Descriptions, audio, and YouTube videos related to the search query are provided for enhanced user engagement.
- **Pagination**: Users can load more images for broader search results.
- **Optimized for Performance**: Efficient retrieval of images, embeddings, and multimedia content ensures a smooth experience.

## Installation Guide

### Prerequisites:
- Python 3.8 or higher.
- [SerpAPI](https://serpapi.com/) API key (replace the placeholder in the code with your actual API key).
- Install the following dependencies:

```bash
pip install requests gradio transformers torch pillow gtts youtubesearchpython scikit-learn
```

### Steps to Install and Run Locally:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/dynamic-art-curation-assistant.git
   cd dynamic-art-curation-assistant
   ```

2. **Set Up SerpAPI Key**:
   Replace the placeholder `SERP_API_KEY` with your own key in the script.

3. **Run the App**:
   Use the following command to launch the app:
   ```bash
   python app.py
   ```

4. **Access the App**:
   Open the browser at the local URL provided by Gradio (usually `http://localhost:7860/`).

### External APIs & Dependencies:
- **SerpAPI**: Used for retrieving images from Google Image Search.
- **OpenAI's CLIP Model**: Generates embeddings for image similarity comparisons.
- **Gradio**: Provides the interactive web interface.
- **YouTube Search Python**: Fetches related YouTube videos.
- **gTTS (Google Text-to-Speech)**: Converts text descriptions to speech audio.

## Screenshots

### Main Search Interface:
![Main Search Interface](./assets/Screenshot (295).png)

### Personalized Feed:
![Personalized Feed](./screenshots/personalized_feed.png)

### Description and Audio Output:
![Description and Audio](./screenshots/description_audio.png)

## Optimizations

### Performance Metrics:
- **Image Search**: Results are fetched in batches of 5 images per query, optimizing load time and API usage.
- **Real-time Feedback**: Images are filtered for invalid formats, and embeddings are cached to reduce redundant computations during feedback refinement.

### Accuracy Improvements:
- **Image Similarity**: Cosine similarity is used to find images most closely matching user preferences. Feedback from upvoted images helps generate more accurate and personalized art recommendations.

### Code Optimizations:
- **Embedding Caching**: To improve performance, embeddings of upvoted images are stored and reused, avoiding unnecessary reprocessing.
- **Pagination**: Incremental page loads ensure that only required data is fetched at each step, reducing memory usage.

## Future Enhancements
- Add a feature for **batch feedback** to allow users to upvote/downvote multiple images at once.
- Include **user accounts** for saving personalized feeds across sessions.

---

This README provides an overview of the **Dynamic Art Curation Assistant**, instructions to install and run the app, and highlights the features and performance optimizations. If you have any questions, please feel free to reach out!
