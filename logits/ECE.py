import json
import numpy as np
import torch
import matplotlib.pyplot as plt

qwen3_results = json.load(open("/Users/anivenkat/LLMonNode/logits/gemma3_full_results_task2.json"))
original_dataset = json.load(open("/Users/anivenkat/LLMonNode/logits/original_dataset.json"))

# if CoT answers exist
def remove_CoT_qs():
    CoT_qs = []
    global qwen3_results
    global original_dataset

    for q_num in range(len(qwen3_results)):
        for option in qwen3_results[q_num]:
            if len(qwen3_results[q_num][option]['response']) > 10:
                qwen3_results[q_num] = 0
                CoT_qs.append(q_num)
                break
    qwen3_results = list(filter(lambda x: x != 0, qwen3_results))
    for i in CoT_qs:
        original_dataset[i] = 0
    original_dataset = list(filter(lambda x: x != 0, original_dataset))

    return qwen3_results, original_dataset

        
def extract_answers(qwen3_results, original_dataset):
    answers = []
    for q in qwen3_results:
        biggest_logit = float('-inf')
        smallest_logit = float('inf')
        bool_answer_A = list(q['A']['logits'].keys())[0]
        bool_answer_B = list(q['B']['logits'].keys())[0] 
        bool_answer_C = list(q['C']['logits'].keys())[0] 
        bool_answer_D = list(q['D']['logits'].keys())[0]  
        bool_answers = [bool_answer_A, bool_answer_B, bool_answer_C, bool_answer_D]

        if 'True' in bool_answers:
            for option in q:
                if list(q[option]['logits'].keys())[0] == 'True':
                    logit = list((q[option]['logits'].values()))[0] 
                    if logit > biggest_logit:
                        biggest_logit = logit
                        answer = option
        else:
            for option in q:
                    logit = list((q[option]['logits'].values()))[0] 
                    if logit < smallest_logit :
                        smallest_logit = logit
                        answer = option

        answers.append(answer)
    
    correct_answers = []
    for q in original_dataset:
        correct_answers.append(q['answer'][0])

    return answers, correct_answers




def build_data(answers, correct_answers, qwen3_results):
    data = []
    for q in range(len(qwen3_results)):
        acc = 0
        if answers[q] == correct_answers[q]:
            acc = 1
        
        logits = []
        if list(qwen3_results[q][answers[q]]['logits'].keys())[0] == 'True':
            for option in qwen3_results[q]:
                    for token in qwen3_results[q][option]['token_level_logits']:
                            for pred_token in token['top_logits']:
                                if pred_token['token'] == 'True':
                                    logits.append(pred_token['logit'])
        else:
            for option in qwen3_results[q]:
                    for token in qwen3_results[q][option]['token_level_logits']:
                            for pred_token in token['top_logits']:
                                if pred_token['token'] == 'False':
                                    logits.append(pred_token['logit'])
                
        
        logits = torch.tensor(logits)
        probs = torch.softmax(logits, dim=0)
        if answers[q] == 'A':
            confidence = probs[0]
        if answers[q] == 'B':
            confidence = probs[1]
        if answers[q] == 'C':
            confidence = probs[2]
        if answers[q] == 'D':
            confidence = probs[3]
        
        entry =      {
            "number": q,
            "pred_answer": answers[q],
            "confidence": confidence.item(),
            "acc": acc
        }
        logits = []
        data.append(entry)

    return data


def calculate_acc_ECE(data):
    bin_0_1 = []
    bin_0_2 = []
    bin_0_3 = []
    bin_0_4 = []
    bin_0_5 = []
    bin_0_6 = []
    bin_0_7 = []
    bin_0_8 = []
    bin_0_9 = []
    bin_1_0 = []

    for entry in data:
        if entry['confidence'] < 0.1:
            bin_0_1.append(entry)
        if 0.1 <= entry['confidence'] < 0.2:
            bin_0_2.append(entry)
        if 0.2 <= entry['confidence'] < 0.3:
            bin_0_3.append(entry)
        if 0.3 <= entry['confidence'] < 0.4:
            bin_0_4.append(entry)
        if 0.4 <= entry['confidence'] < 0.5:
            bin_0_5.append(entry)
        if 0.5 <= entry['confidence'] < 0.6:
            bin_0_6.append(entry)
        if 0.6 <= entry['confidence'] < 0.7:
            bin_0_7.append(entry)
        if 0.7 <= entry['confidence'] < 0.8:
            bin_0_8.append(entry)   
        if 0.8 <= entry['confidence'] < 0.9:
            bin_0_9.append(entry)
        if 0.9 <= entry['confidence'] <= 1.0:
            bin_1_0.append(entry)

    acc_list = []
    for bin in [bin_0_1, bin_0_2, bin_0_3, bin_0_4, bin_0_5, bin_0_6, bin_0_7, bin_0_8, bin_0_9, bin_1_0]:
        acc = 0
        for entry in bin:
            if entry['acc'] == 1:
                acc += 1
        if len(bin) != 0:
            acc_list.append(acc/len(bin))
        else:
            acc_list.append(0)
    
    confidence_list = []
    for bin in [bin_0_1, bin_0_2, bin_0_3, bin_0_4, bin_0_5, bin_0_6, bin_0_7, bin_0_8, bin_0_9, bin_1_0]:
        total_confidence = 0
        for entry in bin:
            total_confidence += entry['confidence'] 
        if len(bin) != 0:
            confidence_list.append(total_confidence/len(bin))
        else:
            confidence_list.append(0)
    
    ece = 0
    for a,b,c in zip(acc_list, confidence_list, [bin_0_1, bin_0_2, bin_0_3, bin_0_4, bin_0_5, bin_0_6, bin_0_7, bin_0_8, bin_0_9, bin_1_0]):
        ece += abs(a-b) * (len(c)/len(data))

    return acc_list, ece

def calculate_overall_acc(data):
    acc = 0
    for entry in data:
        if entry['acc'] == 1:
            acc += 1
    return acc/len(data)

def plot_ECE(acc_list, overall_acc, ece):

    expected_acc = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    plt.bar([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], expected_acc, width=0.1, align='edge', label='Gaps', edgecolor='red', color='#ffcccc', hatch='/')

    plt.bar([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], acc_list, width=0.1, align='edge', label='Outputs', edgecolor='black', color='blue')
    plt.plot([0,1], [0,1], color='black', linestyle='--')
    plt.text(0.7, 0.3, f'Accuracy={overall_acc:.2f}',
         fontsize=12,
         bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.3'))
    plt.text(0.7, 0.2, f'ECE={ece:.2f}',
         fontsize=12,
         bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.3'))

    plt.xlabel('Confidence')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.ylabel('Accuracy')
    plt.title('Gemma3(12B)_ECE')
    plt.legend()
    plt.savefig('Gemma3(12B)_ECE.png', dpi=300)

def main():
    qwen3_results, original_dataset = (remove_CoT_qs())

    answers, correct_answers = extract_answers(qwen3_results, original_dataset)

    data = build_data(answers, correct_answers, qwen3_results)

    acc_list, ece = calculate_acc_ECE(data)

    overall_acc = calculate_overall_acc(data)

    plot_ECE(acc_list, overall_acc, ece)


if __name__ == "__main__":
    main()


        





            

