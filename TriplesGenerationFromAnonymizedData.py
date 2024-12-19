import ast
import pandas as pd
import re

dir_path = "Datasets/ICEWS/anonymisedOutputs_final_x_train_make_a_visit_icews/"
# File path to read data from
input_path = dir_path + "valid_groups_with_count_of_nodes.txt"

# Read the file content and parse it as a dictionary
with open(input_path, "r") as file:
    # Read the single line containing the dictionary
    line = file.readline().strip()
    # Parse the string into a dictionary
    data = ast.literal_eval(line)


# File path to read data from
input_path = dir_path + "invalid_to_valid_groups.txt"

# Read the file content and parse it as a dictionary
with open(input_path, "r") as file:
    # Read the single line containing the dictionary
    line = file.readline().strip()
    # Parse the string into a dictionary
    data1 = ast.literal_eval(line)


# File path to read data from
input_path = dir_path + "groups_to_individual_IDs.txt"

# Read the file content and parse it as a dictionary
with open(input_path, "r") as file:
    # Read the single line containing the dictionary
    line = file.readline().strip()
    # Parse the string into a dictionary
    data2 = ast.literal_eval(line)




print(len(data))
print(len(data1))
print(len(data2))

# Initialize counters for groups and starting individual numbers
individual_id_start = 1  # Tracks starting individual number for each group
output_path = "Datasets/ICEWS/Anonymised/Final_final_training_data_ICEWS.tsv"


# print(len(df.iloc[:,0].unique()))
with open(output_path, "w") as file:
    # Process each key in the data
    for key, num_individuals in data.items():
        
        sens_vals = data1.get(key)
        indi_ids = data2.get(key)
        # print("sens_vals: ",sens_vals)

        # Split key into sections by '&&&' to separate different relationships
        relationships = key.split('&&&')
        
        # Process each relationship in the group
        for relationship in relationships:
            # Extract relation name and values
            relation, values = relationship.split("Values:")
            relation = relation.strip().rstrip(":")  # e.g., "isAffiliatedTo"
            values = values.strip().split(", ")  # List of values for the relation
            
            
            # Generate triples for individuals in this group using the same individual IDs
            for i in indi_ids:
                    for value in values:
                        cleaned_value = re.sub(r"(_\d+|,)$", "", value)
                        file.write(f"{i}\t<{relation}>\t<{cleaned_value}>\n")

        # sensitive values
        # for i in indi_ids:
        #         for value in sens_vals:
        #             cleaned_value = re.sub(r"(_\d+|,)$", "", value)
        #             file.write(f"{i}\t<wasBornIn>\t{cleaned_value}\n")
        
        for i in indi_ids:
            for value in sens_vals:
                # Find all <...> patterns in the value
                matches = re.findall(r"<[^>]+>", value)
                for match in matches:
                    # Clean the match by removing trailing _digits if present
                    cleaned_value = re.sub(r"_\d+$", "", match)
                    file.write(f"{i}\t<Make_a_visit>\t{cleaned_value}\n")

            
        
        # Update the starting individual ID for the next group
        individual_id_start += num_individuals


