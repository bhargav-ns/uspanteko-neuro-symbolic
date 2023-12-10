import os
import pandas as pd

def extract_blocks(file_path):
    with open(file_path, 'r', encoding='iso-8859-15') as file:
        lines = file.readlines()
        blocks = []
        data = {}
        
        for i,line in enumerate(lines):
            if line.startswith('\\ref'):
                if 'segmented_text' in data and 'gloss' in data and 'detailed_gloss' in data and (len(data['segmented_text']) == len(data['gloss'])):  # if data is not empty, append it to blocks
                    blocks.append(data)
                data = {}  # start a new block
            elif line.startswith('\\t'):
                data['text'] = line[2:].strip().split()
            elif line.startswith('\\m'):
                data['segmented_text'] = line[2:].strip().replace(" - ", " ").split()
            elif line.startswith('\\c'):
                data['gloss'] = line[2:].strip().split()
            elif line.startswith('\\l'):
                data['translation'] = line[2:].strip().split()
            elif line.startswith('\\p'):
                data['detailed_gloss'] = line[2:].strip().split()
        if 'segmented_text' in data and 'gloss' in data and 'detailed_gloss' in data and (len(data['segmented_text']) == len(data['gloss'])):  # append the last block
            blocks.append(data)
            
        
        token_mismatch_count = 0
        for i, block in enumerate(blocks):
            if len(block['segmented_text']) != len(block['gloss']):
                token_mismatch_count += 1
        
        print("Token mismatch count: ", token_mismatch_count)
        print("Total blocks: ", len(blocks))
        return blocks

def process_directory(directory_path):
    data_list = []
    for i, filename in enumerate(os.listdir(directory_path)):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            blocks = extract_blocks(file_path)
            data_list.extend(blocks)
    df = pd.DataFrame(data_list)
    return df

uspanteko_data = process_directory('uspanteko_data_original')
uspanteko_data.to_csv('uspanteko_data.csv', index=False)