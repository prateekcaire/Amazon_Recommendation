import streamlit as st
import requests
import subprocess
import sys
import os
from typing import Dict


def load_recommendations(user_id: int = 0):
    """Fetch recommendations from existing API service"""
    try:
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


def style_category_container():
    return """
        <style>
        .category-container {
            margin-bottom: 2rem;
        }
        .product-scroll {
            display: flex;
            overflow-x: auto;
            padding: 1rem 0;
            scroll-behavior: smooth;
        }
        .product-scroll::-webkit-scrollbar {
            height: 8px;
        }
        .product-scroll::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 4px;
        }
        .product-scroll::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 4px;
        }
        .product-scroll::-webkit-scrollbar-thumb:hover {
            background: #555;
        }
        .product-card {
            flex: 0 0 auto;
            width: 280px;
            margin-right: 1rem;
            padding: 1rem;
            border: 1px solid #ddd;
            border-radius: 8px;
            background: white;
        }
        .product-image {
            width: 100%;
            height: 200px;
            object-fit: cover;
            border-radius: 4px;
            margin-bottom: 0.5rem;
        }
        .product-title {
            font-weight: 500;
            margin-bottom: 0.5rem;
            height: 3em;
            overflow: hidden;
        }
        .product-rating {
            color: #f1c40f;
            margin-bottom: 0.5rem;
        }
        .product-price {
            color: #2ecc71;
            font-size: 1.25rem;
            font-weight: bold;
        }
        </style>
    """


def create_product_card(product: Dict) -> str:
    return f"""
        <div class="product-card">
            <img src="{product['image_url'] or 'https://via.placeholder.com/280x200'}" 
                 class="product-image" 
                 alt="{product['title']}">
            <div class="product-title">{product['title'][:50]}...</div>
            <div class="product-rating">{'⭐' * int(product['rating'])}</div>
            <div class="product-price">${product['price']:.2f}</div>
        </div>
    """


def create_category_section(category_name: str, products: list) -> str:
    products_html = ''.join(create_product_card(product) for product in products)
    return f"""
        <div class="category-container">
            <h2 style="margin-bottom: 1rem; font-size: 1.5rem;">📦 {category_name}</h2>
            <div class="product-scroll">
                {products_html}
            </div>
        </div>
    """


def main():
    st.set_page_config(page_title="Product Recommendations", layout="wide")

    # Custom CSS for the app
    st.markdown("""
        <style>
        .stApp {
            background-color: #f8f9fa;
        }
        .main {
            padding: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

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

    # Add the custom CSS for category container
    st.markdown(style_category_container(), unsafe_allow_html=True)

    # Create all category sections
    all_categories_html = []
    for category_id, product_ids in data["recommendations"].items():
        category_name = data["metadata"]["categories"][category_id]

        # Filter products based on user criteria
        filtered_products = [
            data["metadata"]["products"][str(pid)]
            for pid in product_ids
            if str(pid) in data["metadata"]["products"]
               and data["metadata"]["products"][str(pid)]["rating"] >= min_rating
               and price_range[0] <= data["metadata"]["products"][str(pid)]["price"] <= price_range[1]
        ]

        if filtered_products:
            category_html = create_category_section(category_name, filtered_products)
            all_categories_html.append(category_html)

    # Combine all categories and display
    st.markdown(''.join(all_categories_html), unsafe_allow_html=True)


if __name__ == "__main__":
    # Get the current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Change to the script directory
    os.chdir(script_dir)

    # Run streamlit command
    cmd = [sys.executable, '-m', 'streamlit', 'run', __file__,
           '--server.port=8501', '--server.address=localhost']

    if not os.environ.get('STREAMLIT_RUN_MODE'):
        os.environ['STREAMLIT_RUN_MODE'] = 'true'
        subprocess.run(cmd)
    else:
        main()