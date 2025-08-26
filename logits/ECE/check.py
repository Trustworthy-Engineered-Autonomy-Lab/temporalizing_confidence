import json

CoT_qwen3 = json.load(open("/Users/anivenkat/temporalizing_confidence/logits/results/CoT_qwen3.json"))

empty_logits_count = 0
total_options = 0

for question in CoT_qwen3:
    for option_key, option_data in question.items():    
        total_options += 1
        logits = option_data.get('logits', {})
        
        # Check if logits is empty
        if not logits:
            empty_logits_count += 1
            print(f"Option {option_key}: EMPTY LOGITS")
        else:
            print(f"Option {option_key}: {logits}")

print(f"\n📊 Summary:")
print(f"Total options: {total_options}")
print(f"Empty logits: {empty_logits_count}")
print(f"Percentage empty: {(empty_logits_count/total_options)*100:.1f}%")
