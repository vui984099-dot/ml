"""
Reusable UI components for the Streamlit app.
"""

import streamlit as st
from typing import Dict, List, Any, Optional


def product_grid(products: List[Dict[str, Any]], scores: List[float] = None, 
                cols: int = 2, show_add_to_compare: bool = False):
    """Display products in a grid layout."""
    if not products:
        st.info("No products to display")
        return
    
    scores = scores or [None] * len(products)
    
    # Create grid
    for i in range(0, len(products), cols):
        columns = st.columns(cols)
        
        for j, col in enumerate(columns):
            idx = i + j
            if idx < len(products):
                with col:
                    product = products[idx]
                    score = scores[idx]
                    
                    # Product card
                    with st.container():
                        # Image
                        if product.get("image_url"):
                            try:
                                st.image(product["image_url"], width=200)
                            except:
                                st.image("https://via.placeholder.com/200x200?text=No+Image", width=200)
                        
                        # Title and basic info
                        st.write(f"**{product['title'][:60]}...**" if len(product['title']) > 60 else f"**{product['title']}**")
                        st.write(f"Brand: {product.get('brand', 'Unknown')}")
                        
                        if product.get('price'):
                            st.write(f"💰 ${product['price']:.2f}")
                        
                        if product.get('avg_rating'):
                            st.write(f"⭐ {product['avg_rating']:.1f}/5 ({product.get('num_reviews', 0)} reviews)")
                        
                        if score is not None:
                            st.write(f"📊 Score: {score:.3f}")
                        
                        # Action buttons
                        if show_add_to_compare:
                            if st.button(f"Add to Compare", key=f"compare_{idx}"):
                                if "comparison_products" not in st.session_state:
                                    st.session_state.comparison_products = []
                                
                                if len(st.session_state.comparison_products) < 2:
                                    st.session_state.comparison_products.append(product)
                                    st.success("Added to comparison!")
                                    st.experimental_rerun()
                                else:
                                    st.warning("Can only compare 2 products")
                    
                    st.divider()


def search_filters():
    """Render search filter components."""
    with st.expander("🔧 Advanced Filters"):
        col1, col2 = st.columns(2)
        
        with col1:
            min_rating = st.slider("Minimum rating:", 1.0, 5.0, 1.0, 0.5)
            min_reviews = st.number_input("Minimum reviews:", 0, 1000, 0, 10)
        
        with col2:
            max_price = st.number_input("Maximum price ($):", 0.0, 1000.0, 1000.0, 25.0)
            category = st.selectbox("Category:", ["All", "Electronics", "Home & Kitchen", "Sports & Outdoors"])
        
        return {
            "min_rating": min_rating if min_rating > 1.0 else None,
            "min_reviews": min_reviews if min_reviews > 0 else None,
            "max_price": max_price if max_price < 1000.0 else None,
            "category": category if category != "All" else None
        }


def citation_display(citations: List[Dict[str, Any]]):
    """Display citations in an organized format."""
    if not citations:
        st.info("No citations available")
        return
    
    st.subheader(f"📚 Sources ({len(citations)})")
    
    for i, citation in enumerate(citations, 1):
        with st.expander(f"Source {i} - {citation.get('asin', 'Unknown')}"):
            # Citation text
            st.write(citation.get("text_snippet", "No text available"))
            
            # Metadata
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if citation.get("rating"):
                    st.metric("Rating", f"{citation['rating']}/5")
            
            with col2:
                if citation.get("helpful_votes", 0) > 0:
                    st.metric("Helpful Votes", citation["helpful_votes"])
            
            with col3:
                st.write(f"**ASIN:** {citation.get('asin', 'N/A')}")


def comparison_table(product1: Dict[str, Any], product2: Dict[str, Any], 
                    winner_categories: Dict[str, str] = None):
    """Display products in a comparison table format."""
    winner_categories = winner_categories or {}
    
    # Prepare comparison data
    comparison_data = {
        "Attribute": [],
        product1.get("title", "Product 1")[:30]: [],
        product2.get("title", "Product 2")[:30]: [],
        "Winner": []
    }
    
    # Add attributes to compare
    attributes = [
        ("Price", "price", lambda x: f"${x:.2f}" if x else "N/A"),
        ("Rating", "avg_rating", lambda x: f"{x:.1f}/5" if x else "N/A"),
        ("Reviews", "num_reviews", lambda x: str(x) if x else "0"),
        ("Brand", "brand", lambda x: x or "Unknown"),
        ("Category", "category", lambda x: x or "Unknown")
    ]
    
    for attr_name, attr_key, formatter in attributes:
        comparison_data["Attribute"].append(attr_name)
        comparison_data[product1.get("title", "Product 1")[:30]].append(
            formatter(product1.get(attr_key))
        )
        comparison_data[product2.get("title", "Product 2")[:30]].append(
            formatter(product2.get(attr_key))
        )
        
        # Determine winner
        winner_asin = winner_categories.get(attr_key.lower())
        if winner_asin == product1["asin"]:
            winner = "Product 1"
        elif winner_asin == product2["asin"]:
            winner = "Product 2"
        else:
            winner = "Tie"
        
        comparison_data["Winner"].append(winner)
    
    # Display table
    df = pd.DataFrame(comparison_data)
    st.table(df)


def metrics_dashboard():
    """Display system metrics and performance indicators."""
    st.subheader("📊 System Metrics")
    
    # Mock metrics for demo
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Products", "15,420", "+1.2%")
    
    with col2:
        st.metric("Total Reviews", "89,332", "+2.4%")
    
    with col3:
        st.metric("Index Size", "2.1M vectors", "+0.8%")
    
    with col4:
        st.metric("Avg Response Time", "245ms", "-12ms")
    
    # Performance chart (mock data)
    chart_data = pd.DataFrame({
        "Day": range(1, 8),
        "Queries": [120, 135, 148, 162, 158, 175, 192],
        "Success Rate": [0.94, 0.96, 0.95, 0.97, 0.96, 0.98, 0.97]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Daily Queries")
        st.line_chart(chart_data.set_index("Day")["Queries"])
    
    with col2:
        st.subheader("Success Rate")
        st.line_chart(chart_data.set_index("Day")["Success Rate"])


def sidebar_info():
    """Display helpful information in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("💡 Tips")
    
    st.sidebar.markdown("""
    **For better search results:**
    - Use specific product names or features
    - Try different keywords if no results
    - Use filters to narrow down options
    
    **For Q&A:**
    - Ask specific questions about features
    - Questions about "battery life", "sound quality", etc. work well
    - Check citations for source information
    
    **For comparisons:**
    - Add products from search results
    - Or enter ASINs manually
    - Review the summary and key differences
    """)


if __name__ == "__main__":
    main()