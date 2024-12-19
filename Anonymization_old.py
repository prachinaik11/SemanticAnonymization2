# !pip install -U sentence-transformers
# !pip install scikit-learn-extra

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import random
import re
from collections import defaultdict
from sklearn_extra.cluster import KMedoids
import random
from math import comb
from tqdm import tqdm


for file in files1:
  print("snapshot file: ", file)
  # df=pd.read_csv("dataset.csv")
  # df = pd.read_csv("final_snapshot_2003_2017.tsv", delimiter='\t')
  # df = pd.read_csv("final_with_new_col_isCitizenOf_snapshot_1795_1899.tsv", delimiter='\t')
  df = pd.read_csv(file, delimiter='\t')
  last_ID_in_Dataset = df.iloc[:, 2].max()         #addedOn25Nov24
  df_copy = df.copy()

  # Number duplicate sentences
  sentence_count = {}
  for i, sentence in enumerate(df_copy['sentences']):
      if sentence not in sentence_count:
          sentence_count[sentence] = 1
      else:
          sentence_count[sentence] += 1
          df_copy.at[i, 'sentences'] = f"{sentence},__{sentence_count[sentence]}"

  # Printing the modified DataFrame
  print(df_copy)

  # Rest of your code...
  all_dict = df_copy.set_index('sentences').to_dict()
  sentences_dict = all_dict.get('sensitiveAttr')
  individual_id_dict = all_dict.get('individual_ID')    #addedOn25Nov24
  sentences = list(sentences_dict.keys())

  model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

  embeddings = model.encode(sentences, convert_to_tensor=True)


  all_z = [4]
  # all_l = [1, 2, 3, 4, 5]
  all_l = [2]
  RtUA_num = 0
  # z = 8
  # k = 8
  # l=2


  for l in all_l:
      for z in all_z:
          print(z," -- ",l)
          k = z
          # Convert embeddings to numpy array
          # embeddings_np = np.array(embeddings.cpu())  # If using GPU, move embeddings to CPU first

          # Perform K-means clustering
          num_clusters = int(len(sentences)/k)  # Specify the number of clusters
          kmeans = KMeans(n_clusters=num_clusters)
          kmeans.fit(embeddings)

          # Get cluster labels
          cluster_labels = kmeans.labels_

          groups = [[] for _ in range(num_clusters)]

          for i, sentence in enumerate(sentences):
              cluster_index = cluster_labels[i]
              groups[cluster_index].append(sentence)

          # Display groups
          count_of_outliers = 0
          avg_grp_size = 0
          total_individuals_cross_check = 0
          for i, group in enumerate(groups):
              # print(f'Group {i + 1}:', group)
              # print(f'Group {i + 1} size:', len(group))
              avg_grp_size = avg_grp_size + len(group)
              total_individuals_cross_check = total_individuals_cross_check + len(group)
              if len(group) < (k/2):
                count_of_outliers = count_of_outliers + len(group)
          avg_grp_size /= num_clusters
          # print(f'Average group size: {avg_grp_size}')
          # Calculate the variance
          variance = 0
          for group in groups:
              variance += (len(group) - avg_grp_size) ** 2
          variance /= num_clusters
          # print(f'Variance: {variance}')
          # Print cluster labels for each sentence
          # for sentence, label in zip(sentences, cluster_labels):
          #     print(f"Sentence: {sentence} - Cluster: {label}")
          # print ("total_individuals_cross_check : ",total_individuals_cross_check)

          # print("Number of outliers: ", count_of_outliers) ##################CALCULATE THIS AS WELL



          all_unique_SA = set()
          pattern = r'<([^>]+)>([^,<]+(?:,[^,<]+)*)'
          all_category_sets = defaultdict(set)

          for group in groups:
            # print("\n")
            for i in group:
              # print("i:", i)
              all_unique_SA.add(sentences_dict[i])
              matches = re.findall(pattern, i)
              # print("matches: ", matches)

              # Process each match to handle multiple comma-separated values
              result = []
              for tag, values in matches:
                  # Split the values by comma and strip whitespace if any
                  count_for_values_for_a_tag = len(values.split(','))
                  # print("tag: ",tag)
                  # print("count_for_values_for_a_tag: ",count_for_values_for_a_tag)
                  for value in values.split(','):
                      result.append((tag, value.strip()))
                      all_category_sets[tag].add(value)
              # print("Desired output: ", result)



          #added 14feb
          num_valid_groups = 0
          new_SE_groups = {}
          new_SA_groups = {}
          non_unique_SAs_per_group = []
          new_index = 0
          temp_group = []


          invalid_to_valid_groups = {}
          valid_groups_with_count_of_nodes = {}

          groups_to_individual_IDs = {}       #addedOn25Nov24


          count_for_node_keys = {}  # added on 6May
          fake_nodes = 0
          discarded_entries = 0
          # VtU_num = 0    #### Doesn't contain fake nodes as of now
          VtU_num = 0    #### 28th may--added fake nodes as well


          # pattern = r'<([^>]+)>([^,<]+)'
          pattern = r'<([^>]+)>([^,<]+(?:,[^,<]+)*)'

          AL_sum = 0

          for group in tqdm(groups):
              num_of_nodes_in_a_group = len(group)
              node_key = ""
              node_val = ""

              individuals_ID_value = ""     #addedOn25Nov24


              # Dictionary to hold sets for each category
              category_sets = defaultdict(set)  ### for each group, it contains all attributes shared within that group for each predicate

              unique_SA = set()
              individual_ID_set_per_group = set()   #addedOn25Nov24
              fake_nodes_in_a_group = 0

              non_unique_SAs_per_group = []   #added 14feb
              temp_group = group   #added 14feb

              # print("New group of ", len(group))


              for i in group:
                  # print("i: ",i, " ----SA: ",sentences_dict[i])
                  unique_SA.add(sentences_dict[i])
                  individual_ID_set_per_group.add(individual_id_dict[i])     #addedOn25Nov24
                  print("individual_ID_set_per_group :::: ", individual_ID_set_per_group)     #addedOn25Nov24

                  non_unique_SAs_per_group.append(sentences_dict[i])  #added 14feb
                  # Process each match to handle multiple comma-separated values
                  matches = re.findall(pattern, i)
                  # print("matches: ",matches)
                  for tag, values in matches:
                    for value in values.split(','):     # Split the values by comma and strip whitespace if any
                      category_sets[tag].add(value)
                  # print("category_sets: ",category_sets, end='')
                  category_strings = ', &&& '.join(f"{tag}: Values: {', '.join(sorted(values))}" for tag, values in category_sets.items())
                  # print("\ncategory_strings: ",category_strings)


              ####### cluster size less than k
              if len(group) < k:
                # print("less than k")
                ######### cluster size less than k/2
                if len(group)< k/2 or len(group)== 1:
                  # print("discarding group -- len<k/2")
                  discarded_entries = discarded_entries + len(group)

                ######### cluster size more than k/2, less k
                else:
                  ###### FOR AL_SUM -- new updated on 17th May
                  for i in group:
                      # print("i: ",i, " ----SA: ",sentences_dict[i])
                      matches = re.findall(pattern, i)
                      tags_in_a_sentence = [tag[0] for tag in matches]
                      # print("tags_in_a_sentence: ",tags_in_a_sentence)
                      tag_and_Ia_t_count_dict = {}

                      for tag, values in matches:
                          tag_and_Ia_t_count_dict[tag] = len(values.split(','))
                          # print("tag_and_Ia_t_count_dict: ",tag_and_Ia_t_count_dict)


                      for key in category_sets.keys():
                          Ia_t_count = 0
                          if key in tag_and_Ia_t_count_dict:
                            Ia_t_count = tag_and_Ia_t_count_dict[key]
                          # print("key: ",key, "   len(category_sets[key]): ",len(category_sets[key]), "    Ia_t_count: ",Ia_t_count)
                          numerator = len(category_sets[key]) - Ia_t_count
                          # print("len(all_category_sets[key]): ",len(all_category_sets[key]))
                          denominator = len(all_category_sets[key]) - Ia_t_count
                          if denominator > 0:
                              AL_sum = AL_sum + (numerator/denominator)
                          # print("AL_sum", AL_sum)
                  ###### FOR AL_SUM

                  VtU_num = VtU_num + len(group)

                  #########  l-diversity not satisfied (SAs < l)
                  if len(unique_SA) < l:
                      # print("l-diversity not satisfied --- selecting random SA which will satisfy l-diversity")

                      fake_nodes_in_a_group = max( (l - len(unique_SA)) , (k -len(group)) )
                      fake_nodes = fake_nodes + fake_nodes_in_a_group
                      num_of_nodes_in_a_group = num_of_nodes_in_a_group + fake_nodes_in_a_group

                      for i in range(0, fake_nodes_in_a_group):
                        random_SA = random.choice(list(all_unique_SA))
                        if random_SA in unique_SA:
                          random_SA = random.choice(list(all_unique_SA))
                        # print('random_SA:',random_SA)
                        unique_SA.add(random_SA)
                        non_unique_SAs_per_group.append(random_SA)  #added 14feb
                        temp_group.append(group[0])  #added 14feb


                      # node_key = f" {country_set}, {age_set}, {alumni_of_set}, {lang_spoken_set}"
                      node_key = category_strings
                      node_val = unique_SA

                      if num_of_nodes_in_a_group > len(individual_ID_set_per_group):      #addedOn25Nov24
                        while num_of_nodes_in_a_group != len(individual_ID_set_per_group):
                          individual_ID_set_per_group.add(last_ID_in_Dataset + 1)
                          last_ID_in_Dataset = last_ID_in_Dataset + 1
                      individuals_ID_value = individual_ID_set_per_group      #addedOn25Nov24

                      # print("new l : ",unique_SA)
                      # # added on 6May
                      if node_key not in count_for_node_keys:
                        invalid_to_valid_groups[node_key] = node_val
                        valid_groups_with_count_of_nodes[node_key] = num_of_nodes_in_a_group
                        groups_to_individual_IDs[node_key] = individuals_ID_value     #addedOn25Nov24
                        count_for_node_keys[node_key] = 1
                      else:
                        count_for_node_keys[node_key] = count_for_node_keys[node_key] + 1
                        node_key = node_key + "_" + str(count_for_node_keys[node_key])
                        invalid_to_valid_groups[node_key] = node_val
                        valid_groups_with_count_of_nodes[node_key] = num_of_nodes_in_a_group
                        groups_to_individual_IDs[node_key] = individuals_ID_value     #addedOn25Nov24


                      #added 14feb
                      # print("temp_group: ",temp_group)
                      # print("non_unique_SAs_per_group: ",non_unique_SAs_per_group)
                      new_SE_groups[new_index] = temp_group
                      new_SA_groups[new_index] = non_unique_SAs_per_group
                      new_index = new_index + 1

                  #########  l-diversity satisfied, k not satisfied
                  else:
                      # print("l-diversity satisfied, k not satisfied")
                      fake_nodes_in_a_group = k -len(group)
                      fake_nodes = fake_nodes + fake_nodes_in_a_group

                      for i in range(0, fake_nodes_in_a_group):    #added 14feb
                        random_SA = random.choice(list(all_unique_SA))
                        if random_SA in unique_SA:
                          random_SA = random.choice(list(all_unique_SA))
                        #print('random_SA:',random_SA)
                        unique_SA.add(random_SA)
                        non_unique_SAs_per_group.append(random_SA)
                        temp_group = group
                        temp_group.append(group[0])  #added 14feb

                      #added 14feb
                      # print("temp_group: ",temp_group)
                      # print("non_unique_SAs_per_group: ",non_unique_SAs_per_group)
                      new_SE_groups[new_index] = temp_group
                      new_SA_groups[new_index] = non_unique_SAs_per_group
                      new_index = new_index + 1

                      num_of_nodes_in_a_group = num_of_nodes_in_a_group + fake_nodes_in_a_group
                      # node_key = f" {country_set}, {age_set}, {alumni_of_set}, {lang_spoken_set}"
                      node_key = category_strings
                      node_val = unique_SA

                      if num_of_nodes_in_a_group > len(individual_ID_set_per_group):      #addedOn25Nov24
                        while num_of_nodes_in_a_group != len(individual_ID_set_per_group):
                          individual_ID_set_per_group.add(last_ID_in_Dataset + 1)
                          last_ID_in_Dataset = last_ID_in_Dataset + 1
                      individuals_ID_value = individual_ID_set_per_group      #addedOn25Nov24

                      # print(node_key)
                      # invalid_to_valid_groups[node_key] = node_val
                      # valid_groups_with_count_of_nodes[node_key] = num_of_nodes_in_a_group
                      # # added on 6May
                      if node_key not in count_for_node_keys:
                        invalid_to_valid_groups[node_key] = node_val
                        valid_groups_with_count_of_nodes[node_key] = num_of_nodes_in_a_group
                        groups_to_individual_IDs[node_key] = individuals_ID_value     #addedOn25Nov24
                        count_for_node_keys[node_key] = 1
                      else:
                        count_for_node_keys[node_key] = count_for_node_keys[node_key] + 1
                        node_key = node_key + "_" + str(count_for_node_keys[node_key])
                        invalid_to_valid_groups[node_key] = node_val
                        valid_groups_with_count_of_nodes[node_key] = num_of_nodes_in_a_group
                        groups_to_individual_IDs[node_key] = individuals_ID_value     #addedOn25Nov24



              ########## unique SAs less than l, k satisfied
              elif len(unique_SA) < l:

                # ###### FOR AL_SUM
                # for i in group:
                #     print("i: ",i, " ----SA: ",sentences_dict[i])
                #     matches = re.findall(pattern, i)
                #     tags_in_a_sentence = [tag[0] for tag in matches]
                #     print("tags_in_a_sentence: ",tags_in_a_sentence)

                #     for key in category_sets.keys():
                #         Ia_t_count = 0
                #         if key in tags_in_a_sentence:
                #           Ia_t_count = Ia_t_count + 1
                #         print("key: ",key, "   len(category_sets[key]): ",len(category_sets[key]), "    Ia_t_count: ",Ia_t_count)
                #         numerator = len(category_sets[key]) - Ia_t_count
                #         print("len(all_category_sets[key]): ",len(all_category_sets[key]))
                #         denominator = len(all_category_sets[key]) - Ia_t_count
                #         if denominator > 0:
                #             AL_sum = AL_sum + (numerator/denominator)
                #         print("AL_sum", AL_sum)
                # ###### FOR AL_SUM
                ###### FOR AL_SUM -- new updated on 17th May
                for i in group:
                    # print("i: ",i, " ----SA: ",sentences_dict[i])
                    matches = re.findall(pattern, i)
                    tags_in_a_sentence = [tag[0] for tag in matches]
                    # print("tags_in_a_sentence: ",tags_in_a_sentence)
                    tag_and_Ia_t_count_dict = {}

                    for tag, values in matches:
                        tag_and_Ia_t_count_dict[tag] = len(values.split(','))
                        # print("tag_and_Ia_t_count_dict: ",tag_and_Ia_t_count_dict)


                    for key in category_sets.keys():
                        Ia_t_count = 0
                        if key in tag_and_Ia_t_count_dict:
                          Ia_t_count = tag_and_Ia_t_count_dict[key]
                        # print("key: ",key, "   len(category_sets[key]): ",len(category_sets[key]), "    Ia_t_count: ",Ia_t_count)
                        numerator = len(category_sets[key]) - Ia_t_count
                        # print("len(all_category_sets[key]): ",len(all_category_sets[key]))
                        denominator = len(all_category_sets[key]) - Ia_t_count
                        if denominator > 0:
                            AL_sum = AL_sum + (numerator/denominator)
                        # print("AL_sum", AL_sum)
                ###### FOR AL_SUM
                # print("l-diversity not satisfied --- selecting random SA which will satisfy l-diversity")
                VtU_num = VtU_num + len(group)
                fake_nodes_in_a_group = l - len(unique_SA)
                fake_nodes = fake_nodes + fake_nodes_in_a_group
                num_of_nodes_in_a_group = num_of_nodes_in_a_group + fake_nodes_in_a_group

                for i in range(0, fake_nodes_in_a_group):
                  random_SA = random.choice(list(all_unique_SA))
                  if random_SA in unique_SA:
                    random_SA = random.choice(list(all_unique_SA))
                  # print('random_SA:',random_SA)
                  unique_SA.add(random_SA)
                  non_unique_SAs_per_group.append(random_SA)  #added 14feb
                  temp_group.append(group[0])  #added 14feb

                #added 14feb
                # print("temp_group: ",temp_group)
                # print("non_unique_SAs_per_group: ",non_unique_SAs_per_group)
                new_SE_groups[new_index] = temp_group
                new_SA_groups[new_index] = non_unique_SAs_per_group
                new_index = new_index + 1

                # node_key = f" {country_set}, {age_set}, {alumni_of_set}, {lang_spoken_set}"
                node_key = category_strings
                node_val = unique_SA


                if num_of_nodes_in_a_group > len(individual_ID_set_per_group):      #addedOn25Nov24
                  while num_of_nodes_in_a_group != len(individual_ID_set_per_group):
                    individual_ID_set_per_group.add(last_ID_in_Dataset + 1)
                    last_ID_in_Dataset = last_ID_in_Dataset + 1
                individuals_ID_value = individual_ID_set_per_group      #addedOn25Nov24
                # print("new l : ",unique_SA)
                # invalid_to_valid_groups[node_key] = node_val
                # valid_groups_with_count_of_nodes[node_key] = num_of_nodes_in_a_group
                # # added on 6May
                if node_key not in count_for_node_keys:
                  invalid_to_valid_groups[node_key] = node_val
                  valid_groups_with_count_of_nodes[node_key] = num_of_nodes_in_a_group
                  groups_to_individual_IDs[node_key] = individuals_ID_value     #addedOn25Nov24
                  count_for_node_keys[node_key] = 1
                else:
                  count_for_node_keys[node_key] = count_for_node_keys[node_key] + 1
                  node_key = node_key + "_" + str(count_for_node_keys[node_key])
                  invalid_to_valid_groups[node_key] = node_val
                  valid_groups_with_count_of_nodes[node_key] = num_of_nodes_in_a_group
                  groups_to_individual_IDs[node_key] = individuals_ID_value     #addedOn25Nov24




              ######## cluster size more than 2k-1
              elif len(group) > (2*k-1):
                # print(f"length of the group {len(group)} is more than 2k-1 --- splitting the group")
                num_of_groups = int(len(group) / k)
                # print("num_of_groups : ",num_of_groups)
                subgroups = [group[i:i + (k+1)] for i in range(0, len(group), (k+1))]
                # print(len(subgroups))
                groups.extend(subgroups)

              ######## Cluster is valid
              else:
              ###### FOR AL_SUM -- new updated on 17th May
                for i in group:
                    # print("i: ",i, " ----SA: ",sentences_dict[i])
                    matches = re.findall(pattern, i)
                    tags_in_a_sentence = [tag[0] for tag in matches]
                    # print("tags_in_a_sentence: ",tags_in_a_sentence)
                    tag_and_Ia_t_count_dict = {}

                    for tag, values in matches:
                        tag_and_Ia_t_count_dict[tag] = len(values.split(','))
                        # print("tag_and_Ia_t_count_dict: ",tag_and_Ia_t_count_dict)


                    for key in category_sets.keys():
                        Ia_t_count = 0
                        if key in tag_and_Ia_t_count_dict:
                          Ia_t_count = tag_and_Ia_t_count_dict[key]
                        # print("key: ",key, "   len(category_sets[key]): ",len(category_sets[key]), "    Ia_t_count: ",Ia_t_count)
                        numerator = len(category_sets[key]) - Ia_t_count
                        # print("len(all_category_sets[key]): ",len(all_category_sets[key]))
                        denominator = len(all_category_sets[key]) - Ia_t_count
                        if denominator > 0:
                            AL_sum = AL_sum + (numerator/denominator)
                        # print("AL_sum", AL_sum)
                ###### FOR AL_SUM

                # print("in else: Cluster is valid")
                # valid_groups.append(group)
                num_valid_groups = num_valid_groups + 1
                VtU_num = VtU_num + len(group)
                for i in group:
                  # print(i+"---"+sentences_dict[i])
                  unique_SA.add(sentences_dict[i])
                # node_key = f" {country_set}, {age_set}, {alumni_of_set}, {lang_spoken_set}"
                node_key = category_strings
                node_val = unique_SA

                if num_of_nodes_in_a_group > len(individual_ID_set_per_group):      #addedOn25Nov24
                  while num_of_nodes_in_a_group != len(individual_ID_set_per_group):
                    individual_ID_set_per_group.add(last_ID_in_Dataset + 1)
                    last_ID_in_Dataset = last_ID_in_Dataset + 1
                individuals_ID_value = individual_ID_set_per_group      #addedOn25Nov24
                # invalid_to_valid_groups[node_key] = node_val
                # valid_groups_with_count_of_nodes[node_key] = len(group)
                # # added on 6May
                if node_key not in count_for_node_keys:
                  invalid_to_valid_groups[node_key] = node_val
                  valid_groups_with_count_of_nodes[node_key] = num_of_nodes_in_a_group
                  groups_to_individual_IDs[node_key] = individuals_ID_value     #addedOn25Nov24
                  count_for_node_keys[node_key] = 1
                else:
                  count_for_node_keys[node_key] = count_for_node_keys[node_key] + 1
                  node_key = node_key + "_" + str(count_for_node_keys[node_key])
                  invalid_to_valid_groups[node_key] = node_val
                  valid_groups_with_count_of_nodes[node_key] = num_of_nodes_in_a_group
                  groups_to_individual_IDs[node_key] = individuals_ID_value     #addedOn25Nov24



                #added 14feb
                # print("temp_group: ",temp_group)
                # print("non_unique_SAs_per_group: ",non_unique_SAs_per_group)
                new_SE_groups[new_index] = temp_group
                new_SA_groups[new_index] = non_unique_SAs_per_group
                new_index = new_index + 1



          total = sum(valid_groups_with_count_of_nodes.values())
          print("Sum of values:", total)
          print("original number of individuals: ", total-fake_nodes+discarded_entries)  ### CROSSCHECK
          print("fake_nodes : ",fake_nodes)         #addedOn25Nov24
          print("discarded_entries : ",discarded_entries)               #addedOn25Nov24
          print("invalid_to_valid_groups : ",invalid_to_valid_groups)
          print("valid_groups_with_count_of_nodes : ",valid_groups_with_count_of_nodes)
          print("groups_to_individual_IDs : ", groups_to_individual_IDs)                #addedOn25Nov24
          print("last_ID_in_Dataset : ", last_ID_in_Dataset)                #addedOn25Nov24

          with open('invalid_to_valid_groups.txt', 'w') as file:
            file.write(str(invalid_to_valid_groups))
          with open('valid_groups_with_count_of_nodes.txt', 'w') as file:
            file.write(str(valid_groups_with_count_of_nodes))
          with open('groups_to_individual_IDs.txt', 'w') as file:
            file.write(str(groups_to_individual_IDs))




          AAIL_list = []
          AIL_list = []
          SE_list = []
          SA_list= []
          m = k

          small_SE_avgs = 0
          small_SA_avgs = 0
          SE_similarity = 0
          SA_similarity = 0

          # Create groups of three from each cluster
          # groups = [[] for _ in range(num_clusters)]
          # for i, sentence in enumerate(sentences):
          #     cluster_index = int(labels[i])
          #     groups[cluster_index].append(sentence)

          groups = list(new_SE_groups.values())

          valid_groups_count = 0


          for i, group in enumerate(groups):
                # print(group)
                small_SE_avgs = 0

                unique_SA = set()
              # if len(group) >= m:
                # print((group))
                # print('size:', len(group))
                SE_embeddings = model.encode(group)

                for sentence_index in range(len(group)):
                  # print(sentence_index)
                  # print(group[sentence_index])
                  unique_SA.add(sentences_dict[group[sentence_index]])
                  for sentence_index1 in range(sentence_index+1, len(group)):
                    small_SE_avgs = small_SE_avgs + util.cos_sim(SE_embeddings[sentence_index], SE_embeddings[sentence_index1])


                SE_similarity = SE_similarity + small_SE_avgs/comb(len(group), 2)
                valid_groups_count = valid_groups_count + 1
                # print(unique_SA)

          groups1 = list(new_SA_groups.values())

          for i, group in enumerate(groups1):
                small_SA_avgs = 0
                SE_embeddings = model.encode(group)

                for sentence_index in range(len(group)):
                  # print(sentence_index)
                  # print(group[sentence_index])
                  for sentence_index1 in range(sentence_index+1, len(group)):
                    small_SA_avgs = small_SA_avgs + util.cos_sim(SE_embeddings[sentence_index], SE_embeddings[sentence_index1])
                SA_similarity = SA_similarity + small_SA_avgs/comb(len(group), 2)
                # print("small_SE_avgs/comb(len(group), 2) :",small_SE_avgs/comb(len(group), 2))
                # unique_SA_list = list(unique_SA)
                # SA_embeddings2 = model.encode(unique_SA_list)
                # for sentence_index in range(len(unique_SA_list)):
                #   for sentence_index1 in range(sentence_index+1, len(unique_SA_list)):
                #     # print(sentence_index1)
                #     small_SA_avgs = small_SA_avgs + util.cos_sim(SA_embeddings2[sentence_index], SA_embeddings2[sentence_index1])
                # # print(unique_SA_list)
                # if len(unique_SA_list) >= l:
                # # print(small_SA_avgs/comb(len(unique_SA_list), 2))
                #   SA_similarity = SA_similarity + small_SA_avgs/comb(len(unique_SA_list), 2)


          print(SE_similarity)
          print("valid_groups_count : ", valid_groups_count)
          print("len(groups) : ", len(groups))
          final_SE_similarity = SE_similarity / valid_groups_count
          final_SA_similarity = SA_similarity / valid_groups_count
          print("final_SE_similarity : ",final_SE_similarity)
          print("final_SA_similarity : ",final_SA_similarity)

          RtUA_num = len(all_category_sets)
          print("RtUA_num: ",RtUA_num)

          # loss1 = 0
          # for tag in all_category_sets:
          #     # print(f"Tag: {tag}, {all_category_sets[tag]}")
          #     if len(all_category_sets[tag]) > 1 :
          #       loss1 = loss1 + (all_category_sum[tag]/(len(all_category_sets[tag])-1))

          # loss1
          AL = AL_sum / RtUA_num
          VtU_num = VtU_num + fake_nodes ######### added on 28th May
          AAIL = AL / VtU_num
          print("AAIL : ",AAIL)
          print("fake_nodes : ",fake_nodes)
          print("discarded_entries : ",discarded_entries)
          Utu = len(sentences_dict) - discarded_entries + fake_nodes    #### 28th May
          # Utu = len(sentences_dict) + discarded_entries + fake_nodes   # changed on 17th may    ************ask this to sir
          ###### but discarded entries are already part of len(sentences_dict). Why add again?   28th May
          AIL = (AL + (fake_nodes + discarded_entries)) / Utu
          print("AIL : ",AIL)
          AAIL_list.append(AAIL)
          AIL_list.append(AIL)
          SE_list.append(final_SE_similarity)
          SA_list.append(final_SA_similarity)
  
  
  print("AAIL_list : ",AAIL_list)
  print("AIL_list : ",AIL_list)
  print("SE_list : ",SE_list)
  print("SA_list : ",SA_list)

