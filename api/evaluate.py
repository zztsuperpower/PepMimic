import os
import json
import argparse
import numpy as np
from Bio.PDB import PDBParser




def is_peptide_KD(peptide, residues):
    for i in range(len(residues) - 4):
        atom1 = residues[i]['C']
        atom2 = residues[i + 3]['C']
        atom3 = residues[i + 4]['C']
        distance1 = np.linalg.norm(atom1.coord - atom2.coord)
        distance2 = np.linalg.norm(atom1.coord - atom3.coord)
        if peptide[i] == 'K':
            if ((peptide[i + 3] in ['D', 'E'] and 4 < distance1 < 6.5) or
                (peptide[i + 4] in ['D', 'E'] and 4 < distance2 < 6.5)):
                return True
    return False


def is_peptide_head2tail(peptide, residues):
    first_atom = residues[0]['N']  
    last_atom = residues[-1]['C']  
    if np.isnan(first_atom.coord.any()):
        return False 
    distance = np.linalg.norm(first_atom.coord - last_atom.coord)
    if np.isnan(distance):
        return False
    if distance < 6:
        return True
    return False


def is_peptide_SS(peptide, residues):
    first_atom = residues[0]['N']  
    last_atom = residues[-1]['C']  
    if np.isnan(first_atom.coord.any()):
        return False 
    distance = np.linalg.norm(first_atom.coord - last_atom.coord)
    if np.isnan(distance):
        return False
    if distance < 6:
        return True
    return False


def is_peptide_SS(peptide, residues):
    for i in range(len(residues) - 4):
        atom1 = residues[i]['C']
        atom2 = residues[i + 3]['C']
        atom3 = residues[i + 4]['C']
        distance1 = np.linalg.norm(atom1.coord - atom2.coord)
        distance2 = np.linalg.norm(atom1.coord - atom3.coord)
        if peptide[i] == 'K':
            if ((peptide[i + 3] in ['D', 'E'] and 4 < distance1 < 6.5) or
                (peptide[i + 4] in ['D', 'E'] and 4 < distance2 < 6.5)):
                return True
    return False


def evaluate_json(json_path, cyc_type):
    save_json_path = os.path.join(os.path.dirname(json_path), 'filtered_results.jsonl')
    success_id_set = set()
    all_id_set = set()

    with open(json_path, 'r', encoding='utf-8') as f, open(save_json_path, 'w', encoding='utf-8') as output_file:
        for line in f:
            if line.strip():
                python_object = json.loads(line.strip())
                json_object = json.loads(line)
                id = json_object['id']
                peptide_path = json_object['gen_pdb']

                # relaxed_peptide_path = peptide_path[:-4] + '_relaxed' + '.pdb'
                # if os.path.exists(relaxed_peptide_path):
                #     peptide_path = relaxed_peptide_path

                peptide = json_object['gen_seq']
                parser = PDBParser(QUIET=True)
                structure = parser.get_structure('peptide', peptide_path)
                chain = structure[0][json_object['lig_chain']]  
                residues = list(chain.get_residues())

                if len(peptide)<5:
                    continue

                all_id_set.add(id)

                if cyc_type == 'kd':
                    if is_peptide_KD(peptide, residues):
                        success_id_set.add(id)
                        output_file.write(json.dumps(python_object) + '\n')
                elif cyc_type == 'head2tail':
                    if is_peptide_head2tail(peptide, residues):
                        success_id_set.add(id)
                        output_file.write(json.dumps(python_object) + '\n')                        
                else:
                    raise NotImplementedError

    print(len(all_id_set), len(success_id_set), f'success rate {len(success_id_set)/len(all_id_set)}')


def parse():
    parser = argparse.ArgumentParser(description='Generate peptides given epitopes')
    parser.add_argument('--json_path', type=str, required=True, help='Path to the test configuration')
    parser.add_argument('--cyc_type', type=str, required=True, help='Path to the test configuration')
    return parser.parse_known_args()


if __name__ == '__main__':

    args, opt_args = parse()
    evaluate_json(args.json_path, args.cyc_type)
