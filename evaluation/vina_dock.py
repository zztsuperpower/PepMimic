
import os 
import string
import random
import argparse 
import contextlib
import json

import subprocess
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
import tempfile



import numpy as np
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.mol import SDRecord, SDFile
from openbabel import pybel, openbabel
from meeko import MoleculePreparation
from meeko import obutils
from vina import Vina
import AutoDockTools

def get_random_id(length=30):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))



class BaseDockingTask(object):

    def __init__(self, pdb_block, ligand_rdmol):
        super().__init__()
        self.pdb_block = pdb_block
        self.ligand_rdmol = ligand_rdmol

    def run(self):
        raise NotImplementedError()

    def get_results(self):
        raise NotImplementedError()


def suppress_stdout(func):
    def wrapper(*a, **ka):
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                return func(*a, **ka)
    return wrapper


lignd_chain = ['K']
def pdb2sdf(pdb_path, out_sdf):
    pdb_file = PDBFile.read(os.path.join(pdb_path))
    structure = pdb_file.get_structure(include_bonds=True, model=1)
    mask = np.isin(structure.chain_id, lignd_chain)
    selected_structure = structure[mask]

    record = SDRecord()
    record.set_structure(selected_structure)
    sdf_file = SDFile()
    sdf_file["pep1"] = record
    sdf_file.write(out_sdf)
    
    
class PrepLig(object):
    def __init__(self, input_mol, mol_format):
        if mol_format == 'smi':
            self.ob_mol = pybel.readstring('smi', input_mol)
        elif mol_format == 'sdf': 
            self.ob_mol = next(pybel.readfile(mol_format, input_mol))
        else:
            raise ValueError(f'mol_format {mol_format} not supported')
        
    def addH(self, polaronly=False, correctforph=True, PH=7): 
        self.ob_mol.OBMol.AddHydrogens(polaronly, correctforph, PH)
        obutils.writeMolecule(self.ob_mol.OBMol, 'tmp_h.sdf')

    def gen_conf(self):
        sdf_block = self.ob_mol.write('sdf')
        rdkit_mol = Chem.MolFromMolBlock(sdf_block, removeHs=False)
        AllChem.EmbedMolecule(rdkit_mol, Chem.rdDistGeom.ETKDGv3())
        self.ob_mol = pybel.readstring('sdf', Chem.MolToMolBlock(rdkit_mol))
        obutils.writeMolecule(self.ob_mol.OBMol, 'conf_h.sdf')

    @suppress_stdout
    def get_pdbqt(self, lig_pdbqt=None):
        preparator = MoleculePreparation()
        preparator.prepare(self.ob_mol.OBMol)
        if lig_pdbqt is not None: 
            preparator.write_pdbqt_file(lig_pdbqt)
            return 
        else: 
            return preparator.write_pdbqt_string()


class PrepProt(object): 
    def __init__(self, pdb_file): 
        self.prot = pdb_file
    
    def del_water(self, dry_pdb_file): # optional
        with open(self.prot) as f: 
            lines = [l for l in f.readlines() if l.startswith('ATOM') or l.startswith('HETATM')] 
            dry_lines = [l for l in lines if not 'HOH' in l]
        
        with open(dry_pdb_file, 'w') as f:
            f.write(''.join(dry_lines))
        self.prot = dry_pdb_file
        
    def addH(self, prot_pqr):  # call pdb2pqr
        self.prot_pqr = prot_pqr
        subprocess.Popen(['pdb2pqr30','--ff=AMBER',self.prot, self.prot_pqr],
                         stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL).communicate()

    def get_pdbqt(self, prot_pdbqt):
        prepare_receptor = os.path.join(AutoDockTools.__path__[0], 'Utilities24/prepare_receptor4.py')
        subprocess.Popen(['python3', prepare_receptor, '-r', self.prot_pqr, '-o', prot_pdbqt],
                         stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL).communicate()


class VinaDock(object): 
    def __init__(self, lig_pdbqt, prot_pdbqt, ref_ligand_path=''): 
        self.lig_pdbqt = lig_pdbqt
        self.prot_pdbqt = prot_pdbqt
        self.ref_ligand_path = ref_ligand_path
    
    def _max_min_pdb(self, pdb, buffer):
        with open(pdb, 'r') as f: 
            lines = [l for l in f.readlines() if l.startswith('ATOM') or l.startswith('HEATATM')]
            xs = [float(l[31:39]) for l in lines]
            ys = [float(l[39:47]) for l in lines]
            zs = [float(l[47:55]) for l in lines]
            print(max(xs), min(xs))
            print(max(ys), min(ys))
            print(max(zs), min(zs))
            pocket_center = [(max(xs) + min(xs))/2, (max(ys) + min(ys))/2, (max(zs) + min(zs))/2]
            box_size = [(max(xs) - min(xs)) + buffer, (max(ys) - min(ys)) + buffer, (max(zs) - min(zs)) + buffer]
            return pocket_center, box_size
    
    def get_box(self, ref=None, buffer=0):
        '''
        ref: reference pdb to define pocket. 
        buffer: buffer size to add 

        if ref is not None: 
            get the max and min on x, y, z axis in ref pdb and add buffer to each dimension 
        else: 
            use the entire protein to define pocket 
        '''
        if ref is None: 
            ref = self.prot_pdbqt
        self.pocket_center, self.box_size = self._max_min_pdb(ref, buffer)
        print(self.pocket_center, self.box_size)

    def dock(self, score_func='vina', seed=0, mode='dock', exhaustiveness=8, save_pose=False, save_dir='./tmp', save_name=None, **kwargs):  # seed=0 mean random seed
        v = Vina(sf_name=score_func, seed=seed, verbosity=0, **kwargs)
        v.set_receptor(self.prot_pdbqt)
        v.set_ligand_from_file(self.lig_pdbqt)
        v.compute_vina_maps(center=self.pocket_center, box_size=self.box_size)
        if mode == 'score_only': 
            # print(v.score())
            score = v.score()[0]
            print('score_only', score)
            # v.write_maps()
        elif mode == 'minimize':
            score = v.optimize()[0]
            print('minimize', score)
        elif mode == 'dock':
            v.dock(exhaustiveness=exhaustiveness, n_poses=1)
            score = v.energies(n_poses=1)[0][0]
            print('dock', score)
        else:
            raise ValueError
        
        if not save_pose: 
            return score
        else: 
            if mode == 'score_only': 
                pose = None 
            elif mode == 'minimize': 
                tmp = tempfile.NamedTemporaryFile()
                with open(tmp.name, 'w') as f: 
                    v.write_pose(tmp.name, overwrite=True)             
                with open(tmp.name, 'r') as f: 
                    pose = f.read()          
            elif mode == 'dock': 
                pose = v.poses(n_poses=1)

            
            v.write_pose(os.path.join('./tmp', "saved_pose_{}".format(os.path.basename(self.lig_pdbqt))), overwrite=True)
            obConversion = openbabel.OBConversion()
            obConversion.SetInAndOutFormats("pdbqt", "sdf")
            ob_mol = openbabel.OBMol()
            obConversion.ReadFile(ob_mol, os.path.join('./tmp', "saved_pose_{}".format(os.path.basename(self.lig_pdbqt))))
            save_name = os.path.basename(save_name).split('.')[0] if save_name is not None else 'pose'
            saved_path = os.path.join(save_dir, f"docked_{save_name}.sdf")
            print(f'saved to {saved_path}')
            obConversion.WriteFile(ob_mol, saved_path)
                
            return score, pose

class VinaDockingTask(BaseDockingTask):
    @classmethod
    def from_generated_mol(cls, ligand_rdmol, ligand_filename, protein_path, center=None):
        # load original pdb
        return cls(protein_path, ligand_rdmol, center=center, ligand_filename=ligand_filename)


    def __init__(self, protein_path, ligand_rdmol, tmp_dir='./tmp', center=None, ligand_filename=None,
                 size_factor=1., buffer=5.0):
        super().__init__(protein_path, ligand_rdmol)
        # self.conda_env = conda_env
        self.tmp_dir = os.path.realpath(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        self.task_id = get_random_id()
        self.receptor_id = self.task_id + '_receptor'
        self.ligand_id = self.task_id + '_ligand'

        self.receptor_path = protein_path
        self.ligand_path = os.path.join(self.tmp_dir, self.ligand_id + '.sdf')
        self.ligand_filename = ligand_filename

        self.recon_ligand_mol = ligand_rdmol
        # ligand_rdmol = Chem.AddHs(ligand_rdmol, addCoords=True)


        sdf_writer = Chem.SDWriter(self.ligand_path)
        sdf_writer.write(ligand_rdmol)
        sdf_writer.close()
        self.ligand_rdmol = ligand_rdmol

        pos = ligand_rdmol.GetConformer(0).GetPositions()
        if center is None:
            self.center = (pos.max(0) + pos.min(0)) / 2
        else:
            self.center = center

        if size_factor is None:
            self.size_x, self.size_y, self.size_z = 20, 20, 20
        else:
            self.size_x, self.size_y, self.size_z = (pos.max(0) - pos.min(0)) * size_factor + buffer

        self.proc = None
        self.results = None
        self.output = None
        self.error_output = None
        self.docked_sdf_path = None


    def run(self, mode='dock', exhaustiveness=8, save_dir='./tmp', **kwargs):
        ligand_pdbqt = self.ligand_path[:-4] + '.pdbqt'
        protein_pqr = self.receptor_path[:-4] + '.pqr'
        protein_pdbqt = self.receptor_path[:-4] + '.pdbqt'

        
        lig = PrepLig(self.ligand_path, 'sdf')
        lig.get_pdbqt(ligand_pdbqt)

        prot = PrepProt(self.receptor_path)

        if not os.path.exists(protein_pqr):
            print('Excute addH', protein_pqr)
            prot.addH(protein_pqr)
        if not os.path.exists(protein_pdbqt):
            print('Excute get pdbqt', protein_pdbqt)
            prot.get_pdbqt(protein_pdbqt)

        dock = VinaDock(ligand_pdbqt, protein_pdbqt, ref_ligand_path=self.ligand_path)
        dock.pocket_center, dock.box_size = self.center, [self.size_x, self.size_y, self.size_z]
        score, pose = dock.dock(score_func='vina', mode=mode, 
                                exhaustiveness=exhaustiveness, 
                                save_pose=True, 
                                save_dir = save_dir,
                                save_name = self.ligand_filename,
                                **kwargs)
        return {'affinity': score, 'pose': pose}


receptor_chain = ['X', 'Z']
def save_receptor_pdb(pdb_path, out_pdb):
    pdb_file = PDBFile.read(os.path.join(pdb_path))
    structure = pdb_file.get_structure(include_bonds=True, model=1)
    mask = np.isin(structure.chain_id, receptor_chain)
    selected_structure = structure[mask]

    new_file = PDBFile()
    new_file.set_structure(selected_structure)
    new_file.write(out_pdb)


def dock_single(relaxed_path):
    item = os.path.basename(relaxed_path) # 7dha_gen_0_relaxed.pdb
    binder_path = os.path.dirname(relaxed_path) # results_phage14mer_1000/7dha/
    out_sdf = os.path.join(binder_path, 'sdf_path', item[0:-4]+'.sdf')

    save_receptor_pdb(relaxed_path, os.path.join(binder_path, 'prepared_relax_protein', item))
    pdb2sdf(relaxed_path, out_sdf)

    mol = Chem.SDMolSupplier(out_sdf, sanitize=False)[0]
    saved_pose_sdf = os.path.join(binder_path, 'minimized_docked_pose', item[0:-4]+'.sdf')
    vina_task = VinaDockingTask.from_generated_mol(
    mol, saved_pose_sdf, protein_path=os.path.join(binder_path, 'prepared_relax_protein', item), center=None)

    minimized_save_path = os.path.join(binder_path, 'minimized_docked_pose')
    result_summary = open(os.path.join(os.path.dirname(binder_path), 'vina_summary.jsonl'), 'a')
        
    try: 
        minimize_results = vina_task.run(mode='minimize', 
                            exhaustiveness=32,
                            save_dir=minimized_save_path)
        result_summary.write(json.dumps({
                'sdf_file': os.path.join(binder_path, item),
                'minimize': str(minimize_results['affinity']),
            }) + '\n')
        result_summary.flush()
    except Exception as e:
        print(e, item, 'Fail')

    result_summary.close()


def parse():

    parser = argparse.ArgumentParser(description='Run docking for relaxed structures')
    parser.add_argument('--relaxed_path', type=str, required=True, help='Path to the test configuration')

    return parser.parse_known_args()

if __name__ == '__main__':

    args, opt_args = parse()
    dock_single(args.relaxed_path)

    '''
    dock in batch
    '''
    # protein_path = 'example_data/7DHA/7dha_gen_0_relaxed.pdb'
    # binder_path = 'results_phage14mer_with_infer_dis05_base_relax/7dha'

    # os.makedirs(os.path.join(binder_path, 'sdf_path'), exist_ok=True)
    # os.makedirs(os.path.join(binder_path, 'minimized_docked_pose'), exist_ok=True)
    # os.makedirs(os.path.join(binder_path, 'prepared_relax_protein'), exist_ok=True)

    # result_summary = open(os.path.join(os.path.dirname(binder_path), 'vina_summary.jsonl'), 'a')
    # exhaustiveness = 32

    # mol_path_list = os.listdir(binder_path)
    # for item in mol_path_list:
    #     if '.pdb' not in item:
    #         continue
    #     if 'relax' in item:
            

    #         mol_file = os.path.join(binder_path, item)
    #         out_sdf = os.path.join(binder_path, 'sdf_path', item[0:-4]+'.sdf')
    #         docked_sdf = os.path.join(binder_path, 'minimized_docked_pose', item[0:-4]+'.sdf')

    #         save_receptor_pdb(mol_file, os.path.join(binder_path, 'prepared_relax_protein', item))
    #         pdb2sdf(mol_file, out_sdf)
    #         mol = Chem.SDMolSupplier(out_sdf, sanitize=False)[0]

    #         saved_pose_sdf = os.path.join(binder_path, 'minimized_docked_pose', item[0:-4]+'.sdf')
    #         vina_task = VinaDockingTask.from_generated_mol(
    #             mol, saved_pose_sdf, protein_path=os.path.join(binder_path, 'prepared_relax_protein', item), center=None)

    #         minimized_save_path = os.path.join(binder_path, 'minimized_docked_pose')
            
    #         try: 
    #             print(item)
    #             minimize_results = vina_task.run(mode='minimize', 
    #                                 exhaustiveness=exhaustiveness,
    #                                 save_dir=minimized_save_path)
    #             result_summary.write(json.dumps({
    #                     'sdf_file': os.path.join(binder_path, item),
    #                     'minimize': str(minimize_results['affinity']),
    #                 }) + '\n')
    #             result_summary.flush()
    #         except Exception as e:
    #             print(e, item, 'Fail')
    #             continue

    # result_summary.close()
  