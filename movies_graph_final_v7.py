import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import community as community_louvain
import streamlit as st

# Function to load and preprocess data
def load_data():
    # Load movie dataset from CSV file
    data = pd.read_csv('movie_dataset.csv')
    
    # Ensure 'vote_average' is numeric and handle any missing values appropriately
    data['vote_average'] = pd.to_numeric(data['vote_average'], errors='coerce')
    
    # Drop rows where 'vote_average' is not a valid number
    data = data.dropna(subset=['vote_average'])
    
    # Data cleaning and sampling
    data = data.sample(frac=0.33)  # Sample 33% of the dataset
    data['director'].fillna('', inplace=True)  # Fill missing director information
    data['genres'].fillna('', inplace=True)  # Fill missing genre information
    data['revenue'].fillna(0, inplace=True)  # Fill missing revenue information with 0
    return data  # Return cleaned and sampled dataset

# Initialize an empty undirected graph
G = nx.Graph()

# Function to add nodes and edges to the graph based on each row of data
def add_to_graph(row):
    movie_title = row['title']  # Extract movie title from the row
    
    # Add movie node with attributes
    G.add_node(movie_title, type='movie', role='Movie', release_date=row['release_date'], revenue=row['revenue'], vote_average=row['vote_average'])
    
    # Extract and add director node and edge if director information exists
    director = row['director'].strip()
    if director:
        # Check if director node already exists; if not, add it
        if not G.has_node(director):
            G.add_node(director, type='director', role='Director')  # Add director node
        # Create an edge between movie and director
        G.add_edge(movie_title, director, relation='directed_by')  # Directed by relation
    
    # Extract genres, add genre nodes and edges based on movie genres
    genres = row['genres'].split(',')  # Split genres string into a list
    for genre in genres:
        genre = genre.strip()  # Remove leading/trailing whitespace
        if genre:
            # Check if genre node already exists; if not, add it
            if not G.has_node(genre):
                G.add_node(genre, type='genre', role='Genre')  # Add genre node
            # Create an edge between movie and genre
            G.add_edge(movie_title, genre, relation='belongs_to')  # Belongs to relation

# Load and preprocess data using the defined function
data = load_data()

# Apply add_to_graph function to each row of data to construct the graph
data.apply(add_to_graph, axis=1)

# Community detection using Louvain method
partition = community_louvain.best_partition(G)

# Assign community to each node
for node, community in partition.items():
    G.nodes[node]['community'] = community

# Calculate centrality measures
betweenness = nx.betweenness_centrality(G)
eigenvector = nx.eigenvector_centrality(G, max_iter=1000)

# Normalize centrality values for better visualization
max_betweenness = max(betweenness.values())
max_eigenvector = max(eigenvector.values())
normalized_betweenness = {k: v / max_betweenness for k, v in betweenness.items()}
normalized_eigenvector = {k: v / max_eigenvector for k, v in eigenvector.items()}

# Add centrality measures to node attributes
for node in G.nodes():
    G.nodes[node]['betweenness'] = betweenness[node]
    G.nodes[node]['eigenvector'] = eigenvector[node]
    G.nodes[node]['influence'] = (normalized_betweenness[node] + normalized_eigenvector[node]) / 2

# Streamlit logic starts here
if __name__ == "__main__":
    # Streamlit sidebar filters
    st.sidebar.title("Filters")
    
    # Define filter options
    filter_options = {
        'Top 1%': 0.01,
        'Top 5%': 0.05,
        'Top 10%': 0.10,
        'Top 20%': 0.20,
        'All': 1.00
    }

    # Define revenue tiers
    revenue_tiers = {
        'All': (0, float('inf')),  # Adjusted to cover all revenue ranges
        'Under 100M': (0, 100000000),
        '100M - 300M': (100000000, 300000000),
        '300M - 500M': (300000000, 500000000),
        '500M - 700M': (500000000, 700000000),
        '700M - 1B': (700000000, 1000000000),
        'Above 1B': (1000000000, float('inf'))
    }
    
    # Define vote average ranges
    vote_average_ranges = {
        'All': (0, float('inf')),  # All ranges
        '0-1': (0, 1),
        '1-2': (1, 2),
        '2-3': (2, 3),
        '3-4': (3, 4),
        '4-5': (4, 5),
        '5-6': (5, 6),
        '6-7': (6, 7),
        '7-8': (7, 8),
        '8-9': (8, 9),
        '9-10': (9, 10)
    }
    
    # Sidebar selectbox for influential nodes
    selected_filter = st.sidebar.selectbox("Select top influential nodes", list(filter_options.keys()))
    
    # Sidebar selectbox for revenue tier
    selected_revenue_tier = st.sidebar.selectbox("Select revenue tier", list(revenue_tiers.keys()))
    
    # Sidebar slider for vote average range
    selected_vote_range = st.sidebar.slider("Select vote average range", 0.0, 10.0, (0.0, 10.0), 0.1)

    # Apply filters based on selected type
    nodes_of_type = [node for node, data in G.nodes(data=True) if data['type'] == 'director' or data['type'] == 'movie']

    # Apply revenue filter
    revenue_range = revenue_tiers[selected_revenue_tier]
    nodes_of_type = [node for node in nodes_of_type if G.nodes[node].get('revenue', 0) >= revenue_range[0] and G.nodes[node].get('revenue', 0) < revenue_range[1]]

    # Apply vote average filter
    vote_min, vote_max = selected_vote_range
    nodes_of_type = [node for node in nodes_of_type if G.nodes[node].get('vote_average', 0) >= vote_min and G.nodes[node].get('vote_average', 0) <= vote_max]

    # Apply top influential nodes filter
    percentage = filter_options[selected_filter]
    num_top_nodes = max(1, int(len(nodes_of_type) * percentage))
    top_nodes = sorted(nodes_of_type, key=lambda x: G.nodes[x]['influence'], reverse=True)[:num_top_nodes]

    # Include related nodes and edges for the filtered subset
    related_nodes = set(top_nodes)
    for node in top_nodes:
        related_nodes.update(G.neighbors(node))

    G_filtered = G.subgraph(related_nodes).copy()

    # Define the shell layout with specified nodes in inner circles
    director_nodes = [node for node in G_filtered if G_filtered.nodes[node]['type'] == 'director']
    genre_nodes = [node for node in G_filtered if G_filtered.nodes[node]['type'] == 'genre']
    movie_nodes = [node for node in G_filtered if G_filtered.nodes[node]['type'] == 'movie']
    shells = [director_nodes, genre_nodes, movie_nodes]
    pos = nx.shell_layout(G_filtered, nlist=shells)

    x_nodes = [pos[node][0] for node in G_filtered.nodes()]
    y_nodes = [pos[node][1] for node in G_filtered.nodes()]
    x_edges = []
    y_edges = []
    for edge in G_filtered.edges():
        x_edges.extend([pos[edge[0]][0], pos[edge[1]][0], None])
        y_edges.extend([pos[edge[0]][1], pos[edge[1]][1], None])

    # Get node colors and sizes based on influence
    node_colors = [G_filtered.nodes[node]['influence'] for node in G_filtered.nodes()]
    node_sizes = [10 + 20 * G_filtered.nodes[node]['influence'] for node in G_filtered.nodes()]

    # Create Plotly plot with enhanced hover information
    fig = go.Figure(
        data=[
            go.Scatter(
                x=x_edges, 
                y=y_edges, 
                mode='lines',
                line=dict(width=2, color='Grey'), 
                hoverinfo='none'
            ),
            go.Scatter(
                x=x_nodes, 
                y=y_nodes, 
                mode='markers', 
                hoverinfo='text',
                text=[f"{node} ({G_filtered.nodes[node]['role']})<br>Betweenness: {G_filtered.nodes[node]['betweenness']:.4f}<br>Eigenvector: {G_filtered.nodes[node]['eigenvector']:.4f}" +
                      f"<br>Community: {G_filtered.nodes[node]['community']}" for node in G_filtered.nodes()],
                marker=dict(
                    size=node_sizes,
                    color=node_colors,  # Colors based on influence
                    colorscale='Viridis', 
                    colorbar=dict(thickness=15, title='Influence')
                )
            )
        ],
        layout=go.Layout(
            title='Movie-Director-Genre Knowledge Graph with Influence Detection',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
            plot_bgcolor='rgba(0,0,0,0)'  # Transparent plot background
        )
    )

    # Display the Plotly figure using Streamlit
    st.plotly_chart(fig)
