import json

# Load the JSON file
with open('dataset.json', 'r') as f:
    data = json.load(f)

# Step 2: Print the First 100 Entries
for i, entry in enumerate(data[:100]):
    print(f"Entry {i + 1}: {entry}")