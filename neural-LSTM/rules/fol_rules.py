
import torch
from data.dataset import tag_to_ix, ix_to_tag, word_to_ix, ix_to_word

def apply_logic_rules(teacher_output, sentence, pi=0.5):
    """
    Adjust teacher's predictions based on FOL rules.
    teacher_output: raw output probabilities from the teacher network.
    sentence: list of words in the sentence.
    pi: weight parameter for the rules' influence.
    """
    adjusted_output = teacher_output.clone()  # Clone to avoid modifying the original output

    for i in range(len(sentence) - 1):
        
        # Morphological Rules
        if sentence[i] == "k'as" and predicted_tag(i, teacher_output) != "VERB":
            adjusted_output[i, tag_to_ix["VERB"]] = (1 - pi) * adjusted_output[i, tag_to_ix["VERB"]] + pi
        
        # Rule 1: Determiner-Noun Agreement
        # if predicted_tag(i, teacher_output) == "DET" and predicted_tag(i + 1, teacher_output) != "NOUN":
        #     adjusted_output[i + 1, tag_to_ix["NOUN"]] = (1 - pi) * adjusted_output[i + 1, tag_to_ix["NOUN"]] + pi

        # # Rule 2: Adjective-Noun Sequence
        # elif predicted_tag(i, teacher_output) == "ADJ" and predicted_tag(i + 1, teacher_output) != "NOUN":
        #     adjusted_output[i + 1, tag_to_ix["NOUN"]] = (1 - pi) * adjusted_output[i + 1, tag_to_ix["NOUN"]] + pi

        # # Rule 3: Verb-Object Requirement
        # elif predicted_tag(i, teacher_output) == "VERB" and predicted_tag(i + 1, teacher_output) not in ["NOUN", "PRON"]:
        #     adjusted_output[i + 1, tag_to_ix["NOUN"]] = (1 - pi) * adjusted_output[i + 1, tag_to_ix["NOUN"]] + pi
        #     adjusted_output[i + 1, tag_to_ix["PRON"]] = (1 - pi) * adjusted_output[i + 1, tag_to_ix["PRON"]] + pi

    return adjusted_output

def predicted_tag(index, output):
    """ Helper function to get the predicted tag from the output probabilities. """
    _, predicted_idx = torch.max(output[index], 0)
    return ix_to_tag[predicted_idx.item()]

def input_segment(index, input):
    """ Helper function to get the input segment. """
    return ix_to_word[input[index].item()]