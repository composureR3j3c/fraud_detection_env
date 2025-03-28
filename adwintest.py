from river.drift import ADWIN

# Initialize ADWIN
adwin = ADWIN()

# Simulated data stream with a drift
data_stream = [0.1] * 50 + [0.5] * 50  # Change occurs at index 50

# Process the stream
for i, value in enumerate(data_stream):
    adwin.update(value)  # Update ADWIN with the new value
    
    if adwin.drift_detected:
        print(f"Change detected at index {i}")
    else:
        print(f"Change not detected at index {i}")
