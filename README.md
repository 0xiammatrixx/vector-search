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
- [SerpAPI](https://serpapi.com/) API key (not necessary).
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
   Open the browser at the local URL provided by Gradio (usually `http://localhost:7860/`). Make sure to make the URL public.

### External APIs & Dependencies:
- **SerpAPI**: Used for retrieving images from Google Image Search.
- **OpenAI's CLIP Model**: Generates embeddings for image similarity comparisons.
- **Gradio**: Provides the interactive web interface.
- **YouTube Search Python**: Fetches related YouTube videos.
- **gTTS (Google Text-to-Speech)**: Converts text descriptions to speech audio.

## Screenshots

### Main Search Interface:
![Main Search Interface](./assets/interface.png)

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
- The functionality of the personalized feed from upvotes and downvotes seemed to be having issues in filtering images that can be displayed by gradio.
- The youtube video functionality also didnt respond for unknown reasons, i found another way to implement it using something like this, but for some reason, it created errors in my display even though the youtube links were showing:
  
```python  
YOUTUBE_API_KEY = "AIzaSyCJCavrTsQmdlilAxmuo0NlCcUCwDb0RbU"
YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"

def search_youtube_videos(query, num_results=5):
    # YouTube API search parameters
    params = {
        'part': 'snippet',
        'q': query,
        'key': YOUTUBE_API_KEY,
        'type': 'video',
        'maxResults': num_results
    }

    # Make request to YouTube API
    response = requests.get(YOUTUBE_SEARCH_URL, params=params)

    # Check if the response is successful
    if response.status_code == 200:
        video_results = response.json().get('items', [])
        video_links = []

        # Parse results
        for video in video_results:
            video_id = video['id']['videoId']
            video_title = video['snippet']['title']
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            video_links.append((video_title, video_url))

        return video_links
    else:
        return f"Error: {response.status_code}"

def display_videos(query, num_videos):
    videos = search_youtube_videos(query, num_videos)
    if isinstance(videos, str):  # Handle errors
        return [], videos
    else:
        # Display video titles and links
        return gr.Markdown('\n'.join([f"[{title}]({url})" for title, url in videos])), ""
```
- The project can be forked and proper changes or improvements made to correct these faults, it would be deeply appreciated.

---

This README provides an overview of the **Dynamic Art Curation Assistant**, instructions to install and run the app, and highlights the features and performance optimizations. If you have any questions, please feel free to reach out!
