
import torch
from clip import clip
import numpy as np
import torch.nn.functional as F

# Class prompts from models/Text.py
class_descriptor_5 = [
'Relaxed mouth,open eyes,neutral eyebrows,no noticeable emotional changes,engaged with study materials, or natural body posture.',
'Upturned mouth corners,sparkling eyes,relaxed eyebrows,focused on course content,or occasionally nodding in agreement.',
'Furrowed eyebrows, slightly open mouth, wandering or puzzled gaze, chin rests on the palm,or eyes lock on learning material.',
'Mouth opens in a yawn, eyelids droop, head tilts forward, eyes lock on learning material, or hand writing.',
'Shifting eyes, restless or fidgety posture, relaxed but unfocused expression,frequently checking phone,or averted gaze from study materials.'
]

class_names_5 = [
'Neutrality',
'Enjoyment',
'Confusion',
'Fatigue',
'Distraction'
]

def check_similarity():
    device = "cpu"
    # Load CLIP
    model, preprocess = clip.load("ViT-B/16", device=device)
    
    print("Encoding text descriptions...")
    text_inputs = clip.tokenize(class_descriptor_5).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Calculate cosine similarity matrix
        similarity_matrix = text_features @ text_features.t()
        
    print("\nCosine Similarity Matrix between Classes (based on descriptors):")
    print(f"{ '':12s} | " + " | ".join([f"{name[:7]}" for name in class_names_5]))
    print("-" * 70)
    
    for i, row in enumerate(similarity_matrix):
        row_str = " | ".join([f"{val:.3f}" for val in row])
        print(f"{class_names_5[i]:12s} | {row_str}")

    print("\nAnalysis:")
    # Specifically check Confusion (idx 2) vs Neutral (idx 0)
    conf_neut = similarity_matrix[2][0].item()
    print(f"Similarity between Confusion and Neutrality: {conf_neut:.3f}")
    
    # Check Confusion vs others
    print(f"Confusion neighbors sorted:")
    conf_row = similarity_matrix[2]
    indices = torch.argsort(conf_row, descending=True)
    for idx in indices:
        if idx != 2:
            print(f"  - {class_names_5[idx]}: {conf_row[idx]:.3f}")

if __name__ == "__main__":
    check_similarity()
