import os
import json
import argparse
import numpy as np
from Bio.PDB import PDBParser
import matplotlib.pyplot as plt 



def is_aa_type_correct(peptide, residues):
    return (peptide[0:3] == 'NGL' and peptide[-1] == 'C')



def is_distance_below_6(peptide, residues):
    first_atom = residues[0]['N']  
    last_atom = residues[-1]['C']  
    distance = np.linalg.norm(first_atom.coord - last_atom.coord)

    if distance < 6:
        return distance, True
    return distance, False



def evaluate_json(json_path):
    save_json_path = os.path.join(os.path.dirname(json_path), 'filtered_results.jsonl')
    total_cnt = 0
    aa_success_cnt = 0
    distance_success_cnt = 0
    success_cnt = 0
    distance_list = []


    with open(json_path, 'r', encoding='utf-8') as f, open(save_json_path, 'w', encoding='utf-8') as output_file:
        for line in f:
            if line.strip():
                total_cnt += 1
                python_object = json.loads(line.strip())
                json_object = json.loads(line)
                id = json_object['id']
                peptide_path = json_object['gen_pdb']


                peptide = json_object['gen_seq']
                parser = PDBParser(QUIET=True)
                structure = parser.get_structure('peptide', peptide_path)
                chain = structure[0][json_object['lig_chain']]  
                residues = list(chain.get_residues())


                aa_satisfied = is_aa_type_correct(peptide, residues)
                distance, dis_satisfied = is_distance_below_6(peptide, residues)
                distance_list.append(distance)

                if aa_satisfied:
                    aa_success_cnt += 1

                if dis_satisfied:
                    distance_success_cnt += 1
                    

                if aa_satisfied and dis_satisfied:
                    success_cnt += 1
                     

    print(f'generated num {total_cnt}')
    print(f'amino acid success rate {aa_success_cnt / total_cnt}')
    print(f'distance success rate {distance_success_cnt / total_cnt}')
    print(f'success rate {success_cnt / total_cnt}')
    plt.hist(distance_list, bins=20)
    plt.savefig(os.path.dirname(json_path) + '/distance_histogram_w5.png')


def parse():
    parser = argparse.ArgumentParser(description='Generate peptides given epitopes')
    parser.add_argument('--json_path', type=str, required=True, help='Path to the test configuration')
    return parser.parse_known_args()


if __name__ == '__main__':

    args, opt_args = parse()
    evaluate_json(args.json_path)
