import synapses

model = synapses.UpliftModel()
model.load("criteo-uplift-v2.1.csv.gz")

model.train()