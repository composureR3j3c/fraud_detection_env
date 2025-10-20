from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    'realtime_fraud_stream',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda x: json.loads(x.decode('utf-8')),
    auto_offset_reset='earliest'
)

for msg in consumer:
    print(f"Received: {msg.value}")


#usage
from stream_consumer import consumer

records = consumer(limit=100)

fraud_count = sum(1 for r in records if r.get('is_fraud') == 1)
print(f"Fraud cases detected in batch: {fraud_count}")