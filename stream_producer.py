from kafka import KafkaProducer
import pandas as pd
import time, json
from datetime import datetime

df = pd.read_csv('creditcard.csv', parse_dates=['timestamp'])
df = df.sort_values('timestamp')

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

start_time = df['timestamp'].iloc[0]
for i, row in df.iterrows():
    event = row.to_dict()
    producer.send('realtime_fraud_stream', value=event)

    if i < len(df) - 1:
        next_time = df['timestamp'].iloc[i + 1]
        sleep_time = (next_time - row['timestamp']).total_seconds()
        time.sleep(min(sleep_time, 2))  

    print(f"Sent: {event}")
