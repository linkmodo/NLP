import streamlit as st
from openai import OpenAI
from scipy.spatial import distance
import numpy as np

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Function to create embeddings
def create_embeddings(texts, model="text-embedding-3-small"):
    embeddings = []
    for text in texts:
        response = client.embeddings.create(
            input=text,
            model=model
        )
        embeddings.append(response.data[0].embedding)
    return embeddings

# Sample complaints dataset
customer_complaints = [
    "The delivery of my order was delayed by 3 days, and I had to constantly check the tracking system for updates. This caused inconvenience as it was a gift that I needed urgently.",
    "I received a damaged product in the package, and the box itself was torn. It seems there was no care taken during the shipping process.",
    "The refund process is incredibly slow. I submitted my request weeks ago and still haven't received any confirmation or updates on the status of my refund.",
    "The customer service representative I spoke to was extremely rude and unhelpful, refusing to listen to my concerns or provide a proper resolution.",
    "I never received the order I placed two weeks ago, even though the system marked it as delivered. I feel like my money has been wasted.",
    "The packaging was torn and damaged when my order arrived, making it look like the contents could have been tampered with or mishandled during shipping.",
    "The product I received doesn’t match the description on the website at all. It feels misleading, and I now have to go through the hassle of returning it.",
]

# Generate embeddings for complaints
complaints = [{"complaint": complaint, "embedding": emb} for complaint, emb in zip(customer_complaints, create_embeddings(customer_complaints))]

# Streamlit UI
st.title("Semantic Search for Customer Complaints")
st.write("Enter a customer complaint to find the most similar complaint in the database.")

query = st.text_input("Enter your complaint:")

if query:
    search_embedding = create_embeddings([query])[0]
    distances = [distance.cosine(search_embedding, c["embedding"]) for c in complaints]
    min_dist_ind = np.argmin(distances)
    closest_complaint = complaints[min_dist_ind]

    st.subheader("Most Similar Complaint:")
    st.write(closest_complaint["complaint"])
