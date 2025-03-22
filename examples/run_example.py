# examples/run_example.py
from FuzzRush import FuzzRush

source_names = ["Apple Inc", "Microsoft Corp"]
target_names = ["Apple", "Microsoft", "Google"]

matcher = FuzzRush(source_names, target_names)
matcher.tokenize(n=3)
matches = matcher.match()
print(matches)