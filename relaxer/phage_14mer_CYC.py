#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import time
import io
import logging
import numpy as np
import pdbfixer
import openmm
from openmm import Vec3
from openmm.app import Modeller, PDBFile
from openmm import app as openmm_app
from openmm import unit
ENERGY = unit.kilocalories_per_mole
LENGTH = unit.angstroms

from .base import ForceFieldMinimizer



class ForceFieldMinimizerPhage14mer(ForceFieldMinimizer):

    def _fix_cyclic(self, fixer, cyclic_chains, cyclic_opts):
        # For Phage14mer, cyclic_opts should be (({chain_id}, 0), (chain_id), 13)

        assert cyclic_opts is not None, f'cyclic_opts should not be None, but list of pairs ((chain_id, res_pos), (chain_id, res_pos))'
        assert cyclic_opts
        
        modeller = Modeller(fixer.topology, fixer.positions)
        
        print('Delete N terminus H')
        for chain in modeller.topology.chains():
            if chain.id not in cyclic_chains: continue
            atoms_to_remove = []
            for i, res in enumerate(chain.residues()):
                if i == 0:
                    for atom in res.atoms():
                        print(atom)
                        if atom.name == 'H2' or atom.name == 'H3':
                            atoms_to_remove.append(atom)
                elif i == len(chain) - 1:
                    for atom in res.atoms():
                        print(atom)
                        if atom.name == 'OXT': 
                            atoms_to_remove.append(atom)

            print(atoms_to_remove)
            modeller.delete(atoms_to_remove)
        
        fixer.topology = modeller.topology
        fixer.positions = modeller.positions

        
        out_handle = io.StringIO()
        openmm_app.PDBFile.writeFile(fixer.topology, fixer.positions, out_handle, keepIds=True)
        pdb_fixed = out_handle.getvalue()
        new_fixer = pdbfixer.PDBFixer(pdbfile=io.StringIO(pdb_fixed))


        connects = []
        print('Add Connect for N terminus N and ACE\'s C(=O)')
        for chain in new_fixer.topology.chains():
            if chain.id not in cyclic_chains: continue
            for i, residue in enumerate(chain.residues()):
                if i==0 and residue.name == 'ASN':
                    resid = (chain.id, i)
                    for atom in residue.atoms():
                        if atom.name == 'N': ASN_id = atom
                elif residue.name == 'CYC':
                    for atom in residue.atoms():
                        if atom.name in ['C']:
                            ACE_CO_id = atom
                    
        
        
        connects.append('CONECT' + str(ASN_id.id).rjust(5) + str(ACE_CO_id.id).rjust(5))
        connects.append('CONECT' + str(ACE_CO_id.id).rjust(5) + str(ASN_id.id).rjust(5))
        print(connects)


        # reorganize CONECT record
        # pattern = r'^CONECT\b.*(?:\n|$)'

        # exist_connects = re.findall(pattern, pdb_fixed, flags=re.MULTILINE)
        # pdb_fixed = re.sub(pattern, "", pdb_fixed, flags=re.MULTILINE)
        # connects = _reorganize_connects(exist_connects, connects, [])
    
        
        pdb_fixed = self._add_connects(pdb_fixed, connects)

        
        return pdb_fixed, connects


if __name__ == '__main__':
    import sys
    force_field = ForceFieldMinimizerPhage14mer()
    force_field(sys.argv[1], sys.argv[2], cyclic_chains=sys.argv[3], cyclic_opts=[(sys.argv[3], int(sys.argv[4])), (sys.argv[3], int(sys.argv[5]))])