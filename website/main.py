# Custom imports 
from multipage import MultiPage
from pages import intro, graph, template, graph_layout_compare

# Create an instance of the app 
app = MultiPage()

# Add all your applications (pages) here
app.add_page("1. Introduction", intro.app)
app.add_page("2. Network Visualization and Statistics", graph.app)
app.add_page("3. Comparison of Random Network", graph_layout_compare.app)
app.add_page("4. Wordclouds", template.app)
app.add_page("5. Communities and TF-IDF", template.app)
app.add_page("6. Sentiment of Communities", template.app)
app.add_page("7. Conclusion", template.app)



# The main app
app.run()