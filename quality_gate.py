import json
import sys

MINIMUM_ACCURACY = 0.85

with open("metrics.json", "r") as f:
    metrics = json.load(f)

accuracy = metrics["accuracy"]

print(f"Model accuracy: {accuracy:.2%}")
print(f"Required accuracy: {MINIMUM_ACCURACY:.2%}")

if accuracy < MINIMUM_ACCURACY:
    print("QUALITY GATE FAILED")
    sys.exit(1)

print("QUALITY GATE PASSED")
sys.exit(0)
