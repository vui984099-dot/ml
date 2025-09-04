"""
Streamlit frontend for Amazon product recommendation and Q&A system.
"""

import streamlit as st
import requests
import json
import pandas as pd
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="Amazon Product Q&A & Recommendations",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Configuration
API_BASE_URL = "http://localhost:8000"


class APIClient:
    """Client for communicating with the FastAPI backend."""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.json() if response.status_code == 200 else {"status": "error"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def search_products(self, query: str, category: str = None, 
                       min_rating: float = None, max_price: float = None, 
                       top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for products."""
        try:
            payload = {
                "query": query,
                "category": category,
                "min_rating": min_rating,
                "max_price": max_price,
                "top_k": top_k
            }
            # Remove None values
            payload = {k: v for k, v in payload.items() if v is not None}
            
            response = requests.post(f"{self.base_url}/search", json=payload, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Search failed: {response.text}")
                return []
                
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            return []
    
    def ask_question(self, question: str, category: str = None, 
                    max_chunks: int = 5) -> Dict[str, Any]:
        """Ask a product question."""
        try:
            payload = {
                "question": question,
                "category": category,
                "max_chunks": max_chunks
            }
            # Remove None values
            payload = {k: v for k, v in payload.items() if v is not None}
            
            response = requests.post(f"{self.base_url}/qa", json=payload, timeout=60)
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Q&A failed: {response.text}")
                return {}
                
        except Exception as e:
            st.error(f"Q&A error: {str(e)}")
            return {}
    
    def get_recommendations(self, asin: str = None, query: str = None, 
                          top_k: int = 6) -> Dict[str, Any]:
        """Get product recommendations."""
        try:
            payload = {
                "asin": asin,
                "query": query,
                "top_k": top_k,
                "exclude_asins": []
            }
            # Remove None values
            payload = {k: v for k, v in payload.items() if v is not None}
            
            response = requests.post(f"{self.base_url}/recommendations", json=payload, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Recommendations failed: {response.text}")
                return {}
                
        except Exception as e:
            st.error(f"Recommendations error: {str(e)}")
            return {}
    
    def compare_products(self, asin1: str, asin2: str) -> Dict[str, Any]:
        """Compare two products."""
        try:
            payload = {"asin1": asin1, "asin2": asin2}
            
            response = requests.post(f"{self.base_url}/compare", json=payload, timeout=45)
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Comparison failed: {response.text}")
                return {}
                
        except Exception as e:
            st.error(f"Comparison error: {str(e)}")
            return {}


def display_product_card(product: Dict[str, Any], score: float = None):
    """Display a product card with key information."""
    with st.container():
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Display product image or placeholder
            if product.get("image_url"):
                try:
                    st.image(product["image_url"], width=150)
                except:
                    st.image("https://via.placeholder.com/150x150?text=No+Image", width=150)
            else:
                st.image("https://via.placeholder.com/150x150?text=No+Image", width=150)
        
        with col2:
            # Product title
            st.subheader(product["title"])
            
            # Product details
            col2a, col2b, col2c = st.columns(3)
            
            with col2a:
                st.write(f"**Brand:** {product.get('brand', 'Unknown')}")
                if product.get("price"):
                    st.write(f"**Price:** ${product['price']:.2f}")
            
            with col2b:
                if product.get("avg_rating"):
                    st.write(f"**Rating:** {product['avg_rating']:.1f}/5.0")
                st.write(f"**Reviews:** {product.get('num_reviews', 0)}")
            
            with col2c:
                st.write(f"**ASIN:** {product['asin']}")
                if score is not None:
                    st.write(f"**Score:** {score:.3f}")
            
            # Product description (truncated)
            if product.get("description"):
                desc = product["description"]
                if len(desc) > 200:
                    desc = desc[:200] + "..."
                st.write(f"**Description:** {desc}")


def display_qa_response(qa_response: Dict[str, Any]):
    """Display Q&A response with citations."""
    if not qa_response:
        return
    
    # Display answer
    st.subheader("Answer")
    st.write(qa_response["answer"])
    
    # Display confidence
    confidence = qa_response.get("confidence_score", 0)
    confidence_color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
    st.markdown(f"**Confidence:** :{confidence_color}[{confidence:.1%}]")
    
    # Display citations
    if qa_response.get("citations"):
        st.subheader("Citations")
        for i, citation in enumerate(qa_response["citations"], 1):
            with st.expander(f"Source {i} - ASIN: {citation['asin']}"):
                st.write(citation["text_snippet"])
                if citation.get("rating"):
                    st.write(f"Rating: {citation['rating']}/5")
                if citation.get("helpful_votes", 0) > 0:
                    st.write(f"Helpful votes: {citation['helpful_votes']}")
    
    # Display suggested products
    if qa_response.get("suggested_products"):
        st.subheader("Related Products")
        for product in qa_response["suggested_products"]:
            display_product_card(product)
            st.divider()


def display_comparison(comparison: Dict[str, Any]):
    """Display product comparison results."""
    if not comparison:
        return
    
    product1 = comparison["product1"]
    product2 = comparison["product2"]
    
    st.subheader("Product Comparison")
    
    # Comparison summary
    st.write("### Summary")
    st.write(comparison["comparison_summary"])
    
    # Side-by-side comparison
    st.write("### Side-by-Side Details")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**{product1['title']}**")
        display_product_card(product1)
    
    with col2:
        st.write(f"**{product2['title']}**")
        display_product_card(product2)
    
    # Key differences
    if comparison.get("key_differences"):
        st.write("### Key Differences")
        for diff in comparison["key_differences"]:
            st.write(f"• {diff}")
    
    # Category winners
    if comparison.get("winner_categories"):
        st.write("### Category Winners")
        winners = comparison["winner_categories"]
        
        for category, winning_asin in winners.items():
            winner_product = product1 if winning_asin == product1["asin"] else product2
            st.write(f"**{category.title()}:** {winner_product['title']}")


def main():
    """Main Streamlit application."""
    # Initialize API client
    api_client = APIClient()
    
    # App header
    st.title("🛒 Amazon Product Q&A & Recommendations")
    st.markdown("*Powered by retrieval-augmented generation and vector similarity search*")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose functionality:",
        ["🔍 Product Search", "❓ Ask Questions", "💡 Get Recommendations", "⚖️ Compare Products", "🏥 System Health"]
    )
    
    # Check API health
    health_status = api_client.health_check()
    if health_status.get("status") != "healthy":
        st.sidebar.warning("⚠️ API not fully ready")
        st.sidebar.json(health_status)
    else:
        st.sidebar.success("✅ API Ready")
    
    # Main content based on selected mode
    if app_mode == "🔍 Product Search":
        st.header("Product Search")
        
        # Search form
        with st.form("search_form"):
            query = st.text_input("Search query:", placeholder="e.g., wireless headphones, gaming laptop")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                category = st.selectbox("Category (optional):", ["", "Electronics", "Home", "Sports"])
            with col2:
                min_rating = st.number_input("Min rating:", min_value=1.0, max_value=5.0, value=None, step=0.1)
            with col3:
                max_price = st.number_input("Max price ($):", min_value=0.0, value=None, step=10.0)
            
            top_k = st.slider("Number of results:", 1, 20, 10)
            search_submitted = st.form_submit_button("Search Products")
        
        if search_submitted and query:
            with st.spinner("Searching products..."):
                products = api_client.search_products(
                    query=query, 
                    category=category or None,
                    min_rating=min_rating,
                    max_price=max_price,
                    top_k=top_k
                )
            
            if products:
                st.success(f"Found {len(products)} products")
                
                for i, product in enumerate(products):
                    display_product_card(product)
                    
                    # Add to comparison button
                    if st.button(f"Add to comparison", key=f"add_comp_{i}"):
                        if "comparison_products" not in st.session_state:
                            st.session_state.comparison_products = []
                        
                        if len(st.session_state.comparison_products) < 2:
                            st.session_state.comparison_products.append(product)
                            st.success(f"Added {product['title']} to comparison")
                        else:
                            st.warning("Can only compare 2 products at a time")
                    
                    st.divider()
            else:
                st.info("No products found. Try a different search query.")
    
    elif app_mode == "❓ Ask Questions":
        st.header("Product Q&A")
        st.markdown("Ask questions about products and get answers with citations from reviews.")
        
        # Q&A form
        with st.form("qa_form"):
            question = st.text_area(
                "Your question:", 
                placeholder="e.g., How is the battery life? Is this good for gaming? What do customers say about sound quality?"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                category = st.selectbox("Category (optional):", ["", "Electronics", "Home", "Sports"])
            with col2:
                max_chunks = st.slider("Max sources:", 3, 10, 5)
            
            qa_submitted = st.form_submit_button("Ask Question")
        
        if qa_submitted and question:
            with st.spinner("Finding relevant information and generating answer..."):
                qa_response = api_client.ask_question(
                    question=question,
                    category=category or None,
                    max_chunks=max_chunks
                )
            
            if qa_response:
                display_qa_response(qa_response)
            else:
                st.error("Failed to get answer. Please try again.")
    
    elif app_mode == "💡 Get Recommendations":
        st.header("Product Recommendations")
        
        # Recommendation type selection
        rec_type = st.radio(
            "Recommendation type:",
            ["Query-based", "Similar products", "Popular products"]
        )
        
        if rec_type == "Query-based":
            with st.form("query_rec_form"):
                query = st.text_input("What are you looking for?", placeholder="e.g., budget laptop, wireless earbuds")
                top_k = st.slider("Number of recommendations:", 3, 12, 6)
                rec_submitted = st.form_submit_button("Get Recommendations")
            
            if rec_submitted and query:
                with st.spinner("Getting recommendations..."):
                    response = api_client.get_recommendations(query=query, top_k=top_k)
                
                if response and response.get("products"):
                    st.success(f"Found {len(response['products'])} recommendations")
                    
                    for product, score in zip(response["products"], response.get("scores", [])):
                        display_product_card(product, score)
                        st.divider()
        
        elif rec_type == "Similar products":
            with st.form("similar_rec_form"):
                asin = st.text_input("Product ASIN:", placeholder="e.g., B08N5WRWNW")
                top_k = st.slider("Number of recommendations:", 3, 12, 6)
                rec_submitted = st.form_submit_button("Find Similar Products")
            
            if rec_submitted and asin:
                with st.spinner("Finding similar products..."):
                    response = api_client.get_recommendations(asin=asin, top_k=top_k)
                
                if response and response.get("products"):
                    st.success(f"Found {len(response['products'])} similar products")
                    
                    for product, score in zip(response["products"], response.get("scores", [])):
                        display_product_card(product, score)
                        st.divider()
        
        else:  # Popular products
            with st.form("popular_rec_form"):
                top_k = st.slider("Number of recommendations:", 3, 12, 6)
                rec_submitted = st.form_submit_button("Get Popular Products")
            
            if rec_submitted:
                with st.spinner("Getting popular products..."):
                    response = api_client.get_recommendations(top_k=top_k)
                
                if response and response.get("products"):
                    st.success(f"Found {len(response['products'])} popular products")
                    
                    for product, score in zip(response["products"], response.get("scores", [])):
                        display_product_card(product, score)
                        st.divider()
    
    elif app_mode == "⚖️ Compare Products":
        st.header("Product Comparison")
        
        # Check if products are in session state from search
        if "comparison_products" in st.session_state and len(st.session_state.comparison_products) == 2:
            st.info("Products selected from search results:")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Product 1:** {st.session_state.comparison_products[0]['title']}")
            with col2:
                st.write(f"**Product 2:** {st.session_state.comparison_products[1]['title']}")
            
            if st.button("Compare These Products"):
                with st.spinner("Comparing products..."):
                    comparison = api_client.compare_products(
                        asin1=st.session_state.comparison_products[0]["asin"],
                        asin2=st.session_state.comparison_products[1]["asin"]
                    )
                
                if comparison:
                    display_comparison(comparison)
            
            if st.button("Clear Selection"):
                del st.session_state.comparison_products
                st.experimental_rerun()
        
        else:
            # Manual ASIN input
            with st.form("comparison_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    asin1 = st.text_input("Product 1 ASIN:", placeholder="e.g., B08N5WRWNW")
                
                with col2:
                    asin2 = st.text_input("Product 2 ASIN:", placeholder="e.g., B08F7PTF53")
                
                compare_submitted = st.form_submit_button("Compare Products")
            
            if compare_submitted and asin1 and asin2:
                with st.spinner("Comparing products..."):
                    comparison = api_client.compare_products(asin1=asin1, asin2=asin2)
                
                if comparison:
                    display_comparison(comparison)
    
    elif app_mode == "🏥 System Health":
        st.header("System Health & Status")
        
        # API Health
        health = api_client.health_check()
        
        if health.get("status") == "healthy":
            st.success("✅ System is healthy")
        else:
            st.error("❌ System issues detected")
        
        # Display detailed status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Database", health.get("database_status", "unknown"))
        
        with col2:
            st.metric("Search Index", health.get("index_status", "unknown"))
        
        with col3:
            st.metric("ML Models", health.get("model_status", "unknown"))
        
        # System information
        st.subheader("System Information")
        st.json(health)
        
        # Quick actions
        st.subheader("Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Test Search"):
                test_products = api_client.search_products("test", top_k=1)
                if test_products:
                    st.success("Search working ✅")
                else:
                    st.error("Search failed ❌")
        
        with col2:
            if st.button("Test Q&A"):
                test_qa = api_client.ask_question("test question")
                if test_qa:
                    st.success("Q&A working ✅")
                else:
                    st.error("Q&A failed ❌")
        
        with col3:
            if st.button("Test Recommendations"):
                test_recs = api_client.get_recommendations(query="test")
                if test_recs:
                    st.success("Recommendations working ✅")
                else:
                    st.error("Recommendations failed ❌")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with [Streamlit](https://streamlit.io/) • "
        "Powered by [FastAPI](https://fastapi.tiangolo.com/) • "
        "Data from [Amazon Reviews Dataset](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)"
    )


if __name__ == "__main__":
    main()