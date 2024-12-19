import pandas as pd
import random
import os
from tqdm import tqdm
import csv
import sys
import argparse




def process_triples(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    # Extract existing triples and collect citizenship information
    individuals = {}
    citizenships = []

    for line in tqdm(lines):
        if line.strip():  # Ensure the line isn't empty
            # print("line: ",line)
            subject, predicate, obj = line.replace(',', '').strip().split('\t')
            if subject not in individuals:
                individuals[subject] = {}
            if predicate not in individuals[subject]:
                individuals[subject][predicate] = obj
            else:
                individuals[subject][predicate] = individuals[subject][predicate] + ',' + obj
            if predicate == sensitive_attribute:
                if obj not in citizenships:
                    citizenships.append(obj)

    # Assign random citizenships to individuals who lack them
    updated_triples = []
    for subject, predicates in individuals.items():
        # Add existing triples to the updated list
        for predicate, obj in predicates.items():
            updated_triples.append(f"{subject}\t{predicate}\t{obj}\n")

        # Check if '<isCitizenOf>' is missing and add it
        if sensitive_attribute not in predicates:
            random_citizenship = random.choice(citizenships)  # Choose a random existing citizenship
            updated_triples.append(f"{subject}\t<{sensitive_attribute}>\t{random_citizenship}\n")

    # Write the updated triples back to a new file
    with open(output_file, 'w', encoding='utf-8') as file:
        for line in updated_triples:
            file.write(line)

def number_subjects(file_paths):
    subject_mapping = {}
    next_subject_id = 1
    processed_files = []

    for file_path in file_paths:
        processed_lines = []
        with open(file_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for line in reader:
                if len(line) < 3:
                    continue  # Skip lines that are not triples
                subject, predicate, obj = line
                if subject not in subject_mapping:
                    subject_mapping[subject] = next_subject_id
                    next_subject_id += 1
                numbered_subject = subject_mapping[subject]
                processed_lines.append([numbered_subject, predicate, obj])
        processed_files.append(processed_lines)
    
    return processed_files, subject_mapping

def save_numbered_files(processed_files, file_paths, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for idx, processed_lines in enumerate(processed_files):
        output_file = f"{output_dir}/numbered_{idx + 1}.tsv"
        with open(output_file, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(processed_lines)
        print(f"Processed file saved to {output_file}")

def save_subject_mapping(subject_mapping, output_file):
    with open(output_file, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(["Subject", "ID"])
        for subject, number in subject_mapping.items():
            writer.writerow([subject, number])
    print(f"Subject mapping saved to {output_file}")


def newColForPlaceOfBirth(train_file_path, output_file_path):
    # Read the CSV file
    df = pd.read_csv(train_file_path, sep='\t', names=['subject', 'predicate', 'object'])

    # Filter rows where predicate is 'isCitizenOf'
    citizenship_df = df[df['predicate'] == sensitive_attribute][['subject', 'object']].rename(columns={'object': sensitive_attribute})

    # Remove the <isCitizenOf> triples from the original DataFrame
    df = df[df['predicate'] != sensitive_attribute]

    # Merge the citizenship information back to the original dataframe
    result_df = pd.merge(df, citizenship_df, on='subject', how='left')

    # Saving the updated DataFrame to a new CSV file
    result_df.to_csv(output_file_path, sep='\t', index=False)
    # Optionally, print the head of the resulting DataFrame to check
    print(result_df.head())


def process_csv(input_file, output_file):
    # Read CSV into a pandas DataFrame
    df = pd.read_csv(input_file, delimiter='\t')
    gk = df.groupby('subject')
    # Write output to file
    with open(output_file, 'w') as f:
        f.write(f'sentences\tsensitiveAttr\tindividual_ID')
        f.write(f'\n')
        for subject, group_data in gk:
            preds = group_data['predicate'].tolist()
            objs = group_data['object'].tolist()
            objs = [value.replace('<', '').replace('>', '') for value in objs]
            sen_attr = group_data[sensitive_attribute].iloc[0]
            # f.write(f'Name: {subject}\n')
            # f.write(f'sensitive_attribute: {sen_attr}\n')
            # f.write(f'sentence: ')
            sent = ''
            for i in range(0, len(preds)):
                sent = sent + preds[i]+objs[i] + ','
                # sent = sent + objs[i] + ','

            f.write(f'{sent}\t{sen_attr}\t{subject}')
            f.write(f'\n')



def main(data, sensitive_attribute):
    if data == "fb13":
        dir_path = "Datasets/Original_Dataset/fb13/"
        output_dir_path = "Datasets/Anonymised_dataset/fb13/"
        data = "FB13"
        print("fb13")
    elif data == "yago":
        dir_path = "Datasets/Original_Dataset/yago/"
        output_dir_path = "Datasets/Anonymised_dataset/yago/"
        data = "YAGO"
        print("yago")
    elif data == "icews14":
        dir_path = "Datasets/Original_Dataset/icews14/"
        output_dir_path = "Datasets/Anonymised_dataset/icews14/"
        data = "ICEWS14"
        print("icews14")
    else:
        print("data does not exist.")
    col_names=['Subject', 'Relation', 'Object'] 
    originalDataset = dir_path + "x_train_"+data+".tsv"
    print("originalDataset ::: ",originalDataset)
    anonymisedDataset = output_dir_path + "x_train_"+data+"_Anon.tsv"
    X_train = pd.read_csv(originalDataset, sep = "\t", names = col_names)

    #combining values for same relation
    process_triples(originalDataset, anonymisedDataset)

    # mapping each subject to a number for all datasets
    file_paths = [output_dir_path + "x_train_"+data+"_Anon.tsv", dir_path + "x_test_"+data+".tsv",
                dir_path + "x_valid_"+data+".tsv"]
    output_dir = output_dir_path + "mapping_output"  
    subject_mapping_file = f"{output_dir}/subject_mapping.tsv" 

    processed_files, subject_mapping = number_subjects(file_paths)
    save_numbered_files(processed_files, file_paths, output_dir)
    save_subject_mapping(subject_mapping, subject_mapping_file)

    # Adding new column for sensitive_attribute
    train_file_path = output_dir_path + "mapping_output/numbered_1.tsv"
    output_file_path = output_dir_path + "with_new_col_PlaceOfBirth_x_train_"+data+"_Anon.tsv"
    newColForPlaceOfBirth(train_file_path, output_file_path)


    # Creating Final Sentences
    output_final_sentences_file = "Datasets/Sentences_Dataset/final_data_ready_for_anonymisation"+data+".tsv"
    process_csv(output_file_path, output_final_sentences_file)


    #crosscheck for all unique subjects
    test = output_dir_path + "mapping_output/numbered_2.tsv"
    valid = output_dir_path + "mapping_output/numbered_3.tsv" 


    X_test = pd.read_csv(test, sep = "\t", names = col_names)
    X_valid = pd.read_csv(valid, sep = "\t", names = col_names)
    X_train = pd.read_csv(output_final_sentences_file, sep = "\t", names = col_names)


    print("Number of unique subjects in Training data : ", len(X_train.iloc[:, 2].unique()))
    print("Number of Rows in Training data : ", X_train.shape)

    print("Number of unique subjects in Testing data : ", len(X_test.iloc[:, 0].unique()))
    print("Number of triples in Testing data : ", X_test.shape)

    print("Number of unique subjects in Validation data : ", len(X_valid.iloc[:, 0].unique()))
    print("Number of triples in Validation data : ", X_valid.shape)



if __name__ == "__main__":
    sensitive_attribute = ''

    parser = argparse.ArgumentParser(description="Process a single data input.")
    
    # Define the --data argument
    parser.add_argument("--data", required=True, help="Input data (e.g., gb13)")
    
    # Parse the arguments
    args = parser.parse_args()
    
    if args.data == "fb13":
        sensitive_attribute = '<place_of_birth>'
        print("fb13")
    elif  args.data == "yago":
        sensitive_attribute = '<wasBornIn>'
        print("yago")
    elif  args.data == "icews14":
        sensitive_attribute = '<Make_a_visit>'
        print("icews14")
    else:
        print("data does not exist.")
    # Call the main function with the parsed data
    main(args.data, sensitive_attribute)