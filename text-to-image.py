import requests
from serpapi import GoogleSearch
import torch
import tempfile
from urllib.parse import urlparse
from gtts import gTTS
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from youtubesearchpython import VideosSearch
import gradio as gr
import io
from mimetypes import guess_type
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# SerpAPI setup with API key (replace with your actual key)
SERP_API_KEY = '7b888855ce604d453ea06c8c9523e1072bc34e50b226cd75bc1942d106ee59c3'

# Global variables to store feedback, image embeddings, and search history
feedback_scores = {}
image_embeddings = {}  # Store embeddings for upvoted images
current_page = 0
FEEDBACK_THRESHOLD = 3  # Number of feedback clicks to trigger personalized feed
feedback_count = 0  # Global variable to track total feedback clicks

# Function to perform Google Image Search using SerpAPI
def fetch_images_from_web(query, page):
    try:
        search = GoogleSearch({
            "q": query, 
            "tbm": "isch",
            "ijn": str(page),
            "api_key": SERP_API_KEY
        })
        results = search.get_dict()
        image_urls = [img['original'] for img in results.get('images_results', [])[:5]]
        return image_urls
    except Exception as e:
        print(f"Error fetching images: {e}")
        return []

# Function to load image from URL
def load_image_from_url(url):
    try:
        mime_type, _ = guess_type(url)
        if mime_type and not mime_type.startswith('image/') or mime_type == 'image/svg+xml':
            print(f"Skipping unsupported image type: {mime_type}")
            return None
        
        response = requests.get(url, timeout=10)  # Add timeout to avoid hanging
        img = Image.open(io.BytesIO(response.content)).convert("RGB")  # Convert to RGB
        img.verify()  # Ensure image is valid
        return img
    except (IOError, requests.exceptions.RequestException) as e:
        print(f"Error loading image from {url}: {e}")
        return None

# Function to generate image embeddings
def generate_image_embeddings(image_urls):
    images = []
    for url in image_urls:
        img = load_image_from_url(url)
        if img:
            images.append(img)
        else:
            print(f"Skipping invalid image: {url}")
    
    if not images:
        print("No valid images to process.")
        return [], []
    
    # Convert to NumPy and scale images to [0, 1]
    images_np = [(np.array(img) / 255.0).astype(np.float32) for img in images]
    
    inputs = processor(images=images_np, return_tensors="pt", padding=True, do_rescale=False)
    
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs).numpy()
    
    return images, embeddings

# Function to search art and handle pagination
def search_art_with_web_images(query):
    global current_page
    image_urls = fetch_images_from_web(query, current_page)
    images, embeddings = generate_image_embeddings(image_urls)
    
    if not images:  # Check if no valid images were returned
        print("No valid images were processed.")
        return []  # Return empty list to prevent further errors
    
    return images  # Only return valid images

# Function to handle upvoting/downvoting of images
def handle_feedback(image_url, action):
    global feedback_scores, feedback_count, image_embeddings
    if image_url in feedback_scores:
        feedback_scores[image_url] += 1 if action == 'upvote' else -1
    else:
        feedback_scores[image_url] = 1 if action == 'upvote' else -1

    feedback_count += 1  # Increment feedback count

    # If image was upvoted, store its embedding
    if action == 'upvote':
        img = load_image_from_url(image_url)
        if img:
            _, embedding = generate_image_embeddings([image_url])
            if embedding:
                image_embeddings[image_url] = embedding[0]  # Store first embedding
    
    # Check if feedback count reaches the threshold, if yes, update personalized feed
    if feedback_count >= FEEDBACK_THRESHOLD:
        return update_feed()  # Trigger feed update after reaching threshold
    return None  # No update if threshold not reached

# Function to find similar images using embeddings
def find_similar_images(base_embedding, candidate_embeddings, candidate_urls):
    if not candidate_embeddings:
        print("No candidate embeddings available for similarity comparison.")
        return []
    
    # Compute cosine similarity between base embedding and all candidate embeddings
    similarities = cosine_similarity([base_embedding], candidate_embeddings)[0]
    
    # Sort the candidates by similarity in descending order
    sorted_indices = np.argsort(similarities)[::-1]  # Indices of images sorted by similarity score
    
    # Retrieve the URLs of the most similar images
    similar_images = [candidate_urls[i] for i in sorted_indices[:5]]  # Return top 5 similar images
    
    return similar_images


# Personalized feed based on search history and feedback
search_history = []

def validate_feed_images(feed_images):
    valid_feed_images = []
    for image_url in feed_images:
        try:
            img = load_image_from_url(image_url)
            if img:  # Only include valid images
                valid_feed_images.append(image_url)
            else:
                print(f"Skipping invalid personalized feed image: {image_url}")  # Debugging line
        except Exception as e:
            print(f"Error validating personalized feed image {image_url}: {e}")  # Debugging line
    return valid_feed_images

# Example usage within your update_feed function
def update_feed():
    if feedback_scores:  # Check if any feedback exists
        feed_images = []
        print("Feedback Scores:", feedback_scores)  # Debugging line
        
        for image_url, score in feedback_scores.items():
            if score >= 1:  # Show images with positive feedback
                feed_images.append(image_url)
        
        # Validate and filter feed images before updating
        valid_feed_images = validate_feed_images(feed_images)
        
        if not valid_feed_images:
            first_upvoted_image = next(iter(feed_images), None)
            
            # Check if there is an embedding for the first upvoted image
            if first_upvoted_image and first_upvoted_image in image_embeddings:
                base_embedding = image_embeddings[first_upvoted_image]
                
                # Generate new candidates (you can fetch from web or use existing embeddings)
                candidate_urls = fetch_images_from_web("art", 0)  # Fetch a batch of candidate images
                _, candidate_embeddings = generate_image_embeddings(candidate_urls)  # Get their embeddings
                
                # Find the most similar images based on the base embedding
                similar_images = find_similar_images(base_embedding, candidate_embeddings, candidate_urls)
                
                # Validate and filter the similar images
                valid_feed_images.extend(validate_feed_images(similar_images))
        
        print("Final Valid Feed Images:", valid_feed_images)  # Debugging line
        return valid_feed_images

    return []

# Function to load more images (pagination)
def load_more_images(query):
    global current_page
    current_page += 1  # Increment the page number to fetch the next set of images
    return search_art_with_web_images(query)

# Function to get a brief description of the search term (using Wikipedia API)
def get_term_description(query):
    try:
        response = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{query}")
        if response.status_code == 200:
            data = response.json()
            return data.get('extract', "No description available.")
        else:
            return "No description available."
    except Exception as e:
        print(f"Error fetching description: {e}")
        return "No description available."

# Function to generate TTS for the description and return a file path
def generate_audio(text):
    tts = gTTS(text)
    temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio_file.name)
    return temp_audio_file.name  # Return the file path to be used by Gradio

# Function to get the first YouTube video link related to the search term
def fetch_first_youtube_video(query):
    try:
        videos_search = VideosSearch(query, limit=1)
        results = videos_search.result()
        if results and 'result' in results and len(results['result']) > 0:
            video_url = f"https://www.youtube.com/watch?v={results['result'][0]['id']}"
            return video_url
        return "No video found."
    except Exception as e:
        print(f"Error fetching YouTube video: {e}")
        return "No video found."


# Gradio UI for the dynamic art curation assistant with real-time refinement, feedback, and pagination
with gr.Blocks() as art_interface:
    gr.Markdown("# Dynamic Art Curation Assistant with Real-Time Search and Feedback")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Input Panel")
            art_query = gr.Textbox(placeholder="Search for art (e.g., surrealism, cubism, impressionism)", label="Search Art", interactive=True)
            submit_button = gr.Button("Search")
            more_button = gr.Button("Load More")  # Load more images button for pagination
            description_output = gr.Textbox(label="Description", interactive=False)
            audio_output = gr.Audio(label="TTS Audio", interactive=False)
            video_output = gr.Video(label="Related YouTube Video")

            # Personalized feed section
            gr.Markdown("### Personalized Feed")
            feed_gallery = gr.Gallery(label="Your Personalized Feed")

        with gr.Column():
            gr.Markdown("### Art Recommendations")
            thumbnails = gr.Gallery(label="Recommended Art")  # Display multiple thumbnails
            
            # Upvote and downvote buttons for feedback
            upvote_button = gr.Button("Upvote")
            downvote_button = gr.Button("Downvote")
            feedback_label = gr.Textbox(label="Image URL for feedback")
    
    # Real-time refinement: Trigger search on input change
    art_query.change(fn=search_art_with_web_images, inputs=art_query, outputs=[thumbnails])

    # Trigger personalized feed update when new search happens
    def handle_search(query):
        global search_history
        search_history.append(query)
        description = get_term_description(query)
        audio = generate_audio(description)
        video_url = fetch_first_youtube_video(query)
        return search_art_with_web_images(query), update_feed(), description, audio, video_url

    submit_button.click(fn=handle_search, inputs=art_query, outputs=[thumbnails, feed_gallery, description_output, audio_output, video_output])

    # Load more images for pagination
    more_button.click(fn=load_more_images, inputs=art_query, outputs=[thumbnails])

    # Handle upvote/downvote feedback
    upvote_button.click(fn=lambda img_url: handle_feedback(img_url, 'upvote'), inputs=feedback_label, outputs=feed_gallery)
    downvote_button.click(fn=lambda img_url: handle_feedback(img_url, 'downvote'), inputs=feedback_label, outputs=feed_gallery)

# Launch the Gradio interface
art_interface.launch(share=True)