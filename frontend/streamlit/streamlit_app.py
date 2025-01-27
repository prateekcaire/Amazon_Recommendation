import subprocess
import sys

import streamlit as st
import requests
import pandas as pd
from typing import Dict


def load_recommendations(user_id: int = 0):
    """Fetch recommendations from existing API service"""
    try:
        # Use the existing API endpoint from api.py
        response = requests.get(f'http://127.0.0.1:5000/api/recommendations/{user_id}')
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error fetching recommendations: {response.status_code}")
            return None
    except requests.RequestException as e:
        st.error(f"Failed to connect to recommendation service: {str(e)}")
        st.error("Please make sure the API service (api.py) is running")
        return None


def style_product_card(product: Dict):
    st.markdown(
        f"""
        <div style="
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 10px;
            margin: 10px 0;
            background-color: white;
        ">
            <img src="{product['image_url'] or 'https://via.placeholder.com/150'}" 
                 style="width: 100%; height: 200px; object-fit: cover; border-radius: 4px;">
            <h3 style="margin: 10px 0; font-size: 1rem;">{product['title'][:50]}...</h3>
            <div style="color: #f1c40f; font-size: 1.2rem;">{'⭐' * int(product['rating'])}</div>
            <div style="color: #2ecc71; font-size: 1.4rem; font-weight: bold; margin-top: 5px;">
                ${product['price']}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def create_recommendation_app():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f8f9fa;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🛍️ Smart Product Recommendations")

    # Add filters in sidebar
    with st.sidebar:
        st.header("Filters")
        user_id = st.number_input("User ID", min_value=0, value=0)
        min_rating = st.slider("Minimum Rating", 0.0, 5.0, 0.0)
        price_range = st.slider("Price Range", 0, 200, (0, 200))

    # Load data from existing API service
    data = load_recommendations(user_id)

    if data is None:
        st.error("Unable to load recommendations. Please make sure api.py is running.")
        return

    for category_id, product_ids in data["recommendations"].items():
        category_name = data["metadata"]["categories"][category_id]
        st.header(f"📦 {category_name}")

        # Filter products
        filtered_products = [
            data["metadata"]["products"][str(pid)]
            for pid in product_ids
            if str(pid) in data["metadata"]["products"]
               and data["metadata"]["products"][str(pid)]["rating"] >= min_rating
               and price_range[0] <= data["metadata"]["products"][str(pid)]["price"] <= price_range[1]
        ]

        if not filtered_products:
            st.warning("No products match your filters in this category.")
            continue

        cols = st.columns(3)
        for idx, product in enumerate(filtered_products):
            with cols[idx % 3]:
                style_product_card(product)


if __name__ == "__main__":
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        __file__,
        "--server.port=8501",
        "--server.address=localhost"
    ])

    st.set_page_config(
        page_title="Product Recommendations",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    create_recommendation_app()