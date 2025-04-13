import synapses

# model = synapses.UpliftModel()
# model.load("criteo-uplift-v2.1.csv.gz")

# model.train()

# Create the app instance
app = synapses.AppBuilder()

# Launch the Streamlit interface
app.create()
app.change()