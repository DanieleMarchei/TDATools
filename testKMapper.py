# Import the class
import kmapper as km
import matplotlib.pyplot as plt

from numpy import genfromtxt
from sys import argv

data = genfromtxt(argv[1], delimiter=',')
# Initialize
mapper = km.KeplerMapper(verbose=0)

# Fit to and transform the data
projected_data = mapper.fit_transform(data, projection="sum") # X-Y axis

# Create dictionary called 'graph' with nodes, edges and meta-information
graph = mapper.map(projected_data)

# Visualize it
s = argv[1].replace(".csv","")
mapper.visualize(graph, path_html=s+".html", title=s)