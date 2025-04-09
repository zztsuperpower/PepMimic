import torch
import random 



def condition_stapled(batch_ids,mask_generate):
    '''
    k-D/E i-i+3/i+4 distance 4-6.5
    '''
    unique_vals = torch.unique(batch_ids)
    sampled_indices = []
    positions1 = []
    positions2 = []
    for val in unique_vals:
        valid_indices = (batch_ids == val) & mask_generate
        indices = valid_indices.nonzero(as_tuple=True)[0]
        if len(indices) < 5:
            continue
        random_indices = indices[(max(indices)-indices>=4)]

        indice = random_indices[0] # random.choice(random_indices)
        hop = random.choice([3,4])
        sampled = [indice, indice + hop]
        positions1.append(indice)
        positions2.append(indice + hop)
        sampled_indices += sampled
    
    positions1 = torch.stack(positions1).to(batch_ids.device)
    positions2 = torch.stack(positions2).to(batch_ids.device)
    
    # control the type of K 
    guidance_node_attr = torch.zeros((mask_generate.shape[0], 20)).to(mask_generate.device)
    one_hot_vector = torch.zeros(20).to(mask_generate.device)
    one_hot_vector[12] = 1 # K
    one_hot_vector = one_hot_vector.unsqueeze(0).repeat(len(positions1),1)
    guidance_node_attr[positions1] = one_hot_vector

    # control the type of D/E
    one_hot_vector = torch.zeros(20).to(mask_generate.device)
    if hop == 3:
        one_hot_vector[8] = 1 # D
    elif hop == 4:
        one_hot_vector[11] = 1 # E
    one_hot_vector = one_hot_vector.unsqueeze(0).repeat(len(positions2),1)
    guidance_node_attr[positions2] = one_hot_vector

    edges = torch.stack([positions1, positions2], dim=0)
    reversed_edges = edges.flip(0)
    sampled_edges = torch.cat([edges,reversed_edges], dim=1)
    return guidance_node_attr, sampled_edges


def condition_head2tail(batch_ids, mask_generate):
    '''
    The distance of head and tail is less than 6 A
    '''
    
    # No type constraint
    guidance_node_attr = torch.zeros((mask_generate.shape[0], 20)).to(mask_generate.device)

    head_positions = []
    tail_positions = []
    for i in range(1, len(mask_generate)):
        if mask_generate[i] != mask_generate[i-1]:
            if mask_generate[i]:
                head_positions.append(i)
            else:
                tail_positions.append(i-1)

    tail_positions.append(len(mask_generate)-1)
    head_positions = torch.tensor(head_positions).to(mask_generate.device)
    tail_positions = torch.tensor(tail_positions).to(mask_generate.device)

    sampled_ht_edges = torch.stack([head_positions, tail_positions], dim=0)
    reversed_edges = sampled_ht_edges.flip(0)
    
    sampled_edges = torch.cat([sampled_ht_edges, reversed_edges], dim=1)
    
    return guidance_node_attr, sampled_edges


def condition_disulfide(batch_ids, mask_generate):
    '''
    two positions Cys with distance between 3.5-5 A
    '''
    unique_vals = torch.unique(batch_ids)
    sampled_indices = []
    positions1 = []
    positions2 = []
    for val in unique_vals:
        valid_indices = (batch_ids == val) & mask_generate
        indices = valid_indices.nonzero(as_tuple=True)[0]
        
        if len(indices)<4:
            continue
        head_indices = indices[(max(indices)-indices>=3)]
        head_indice = random.choice(head_indices)
        sampled = [head_indice, head_indice+3]
        positions1.append(head_indice)
        positions2.append(head_indice+ 3 )
        sampled_indices += sampled

    sampled_indices = torch.tensor(sampled_indices).to(mask_generate.device)
    guidance_node_attr = torch.zeros((mask_generate.shape[0], 20)).to(mask_generate.device)
    one_hot_vector = torch.zeros(20).to(mask_generate.device)
    one_hot_vector[18] = 1 # C
    one_hot_vector = one_hot_vector.unsqueeze(0).repeat(len(sampled_indices),1)
    guidance_node_attr[sampled_indices] = one_hot_vector

    positions1 = torch.stack(positions1)
    positions2 = torch.stack(positions2)
    edges = torch.stack([positions1, positions2], dim=0)
    reversed_edges = edges.flip(0)

    sampled_edges = torch.cat([edges, reversed_edges], dim=1)
   
    return guidance_node_attr, sampled_edges




def condition_phage14mer(batch_ids, mask_generate):
    '''
    ASN-GLY-LEU-(AAA)10-CYS
    '''
    unique_vals = torch.unique(batch_ids)
    sampled_indices = []
    positions_ASN = []
    positions_GLY = []
    positions_LEU = []
    positions_CYS = []

    for val in unique_vals:
        valid_indices = (batch_ids == val) & mask_generate

        indices = valid_indices.nonzero(as_tuple=True)[0]
        length = len(indices) - 1
        # assert len(indices) == 14, print('for phage, length must be 14!')

        indice = indices[0] 
 
        sampled = [indice, indice + 1, indice + 2, indice + 3, indice + length]
        positions_ASN.append(indice)
        positions_GLY.append(indice + 1)
        positions_LEU.append(indice+ 2)
        positions_CYS.append(indice + length)

        sampled_indices += sampled
    
    positions_ASN = torch.stack(positions_ASN).to(batch_ids.device)
    positions_GLY = torch.stack(positions_GLY).to(batch_ids.device)
    positions_LEU = torch.stack(positions_LEU).to(batch_ids.device)
    positions_CYS = torch.stack(positions_CYS).to(batch_ids.device)
    
    
    guidance_node_attr = torch.zeros((mask_generate.shape[0], 20)).to(mask_generate.device)

    one_hot_vector = torch.zeros(20).to(mask_generate.device)
    one_hot_vector[10] = 1 # ASN
    one_hot_vector = one_hot_vector.unsqueeze(0).repeat(len(positions_ASN),1)
    guidance_node_attr[positions_ASN] = one_hot_vector

    one_hot_vector = torch.zeros(20).to(mask_generate.device)
    one_hot_vector[0] = 1 # GLY
    one_hot_vector = one_hot_vector.unsqueeze(0).repeat(len(positions_GLY),1)
    guidance_node_attr[positions_GLY] = one_hot_vector

    one_hot_vector = torch.zeros(20).to(mask_generate.device)
    one_hot_vector[3] = 1 # LEU
    one_hot_vector = one_hot_vector.unsqueeze(0).repeat(len(positions_LEU),1)
    guidance_node_attr[positions_LEU] = one_hot_vector


    one_hot_vector = torch.zeros(20).to(mask_generate.device)
    one_hot_vector[18] = 1 # CYS
    one_hot_vector = one_hot_vector.unsqueeze(0).repeat(len(positions_LEU),1)
    guidance_node_attr[positions_CYS] = one_hot_vector


    edges = torch.stack([positions_ASN, positions_CYS], dim=0)
    reversed_edges = edges.flip(0)
    sampled_edges = torch.cat([edges, reversed_edges], dim=1)

    return guidance_node_attr, sampled_edges    