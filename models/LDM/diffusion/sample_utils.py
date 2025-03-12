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
        if len(indices)<5:
            continue
        random_indices = indices[(max(indices)-indices>=4)]
        indice = random.choice(random_indices)
        hop = random.choice([3,4])
        sampled = [indice,indice+hop]
        positions1.append(indice)
        positions2.append(indice+hop)
        sampled_indices += sampled
    
    positions1 = torch.stack(positions1).to(batch_ids.device)
    positions2 = torch.stack(positions2).to(batch_ids.device)
    
    # control the type of K 
    guidance_node_attr = torch.zeros((mask_generate.shape[0], 20)).to(mask_generate.device)
    one_hot_vector = torch.zeros(20).to(mask_generate.device)
    one_hot_vector[16] = 1 # K
    one_hot_vector = one_hot_vector.unsqueeze(0).repeat(len(positions1),1)
    guidance_node_attr[positions1] = one_hot_vector

    # control the type of D/E
    one_hot_vector = torch.zeros(20).to(mask_generate.device)
    if hop == 3:
        one_hot_vector[12] = 1 # D
    elif hop == 4:
        one_hot_vector[15] = 1 # E
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
    atom_full = torch.zeros((mask_generate.shape[0], 20)).to(mask_generate.device)

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
    
    return atom_full, sampled_edges


def condition11(self,atom_gt,batch_ids,mask_generate,X_true,atom_mask):
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
        if len(indices)<=10:
            continue
        random_indices = indices[(max(indices)-indices>=10)]
        indice = random.choice(random_indices)
        hop = random.choice([3,4])
        sampled = [indice,indice+hop]
        positions1.append(indice)
        positions2.append(indice+hop)
        sampled_indices+=sampled

        random_indices = indices[(indices>indice+hop)&(max(indices)-indices>=4)]
        indice = random.choice(random_indices)
        hop = random.choice([3,4])
        sampled = [indice,indice+hop]
        positions1.append(indice)
        positions2.append(indice+hop)
        sampled_indices+=sampled

    positions1 = torch.stack(positions1).to(atom_gt.device)
    positions2 = torch.stack(positions2).to(atom_gt.device)
    
    # control the type of K 
    atom_full = torch.zeros((mask_generate.shape[0],atom_gt.shape[1])).to(atom_gt.device)
    one_hot_vector = torch.zeros(20).to(atom_gt.device)
    one_hot_vector[11] = 1 # K
    one_hot_vector = one_hot_vector.unsqueeze(0).repeat(len(positions1),1)
    atom_full[positions1] = one_hot_vector

    # control the type of D/E
    atom_full = torch.zeros((mask_generate.shape[0],atom_gt.shape[1])).to(atom_gt.device)
    one_hot_vector = torch.zeros(20).to(atom_gt.device)
    if random.random()<0.5:
        one_hot_vector[3] = 1 # D
    else:
        one_hot_vector[5] = 1 # E
    one_hot_vector = one_hot_vector.unsqueeze(0).repeat(len(positions2),1)
    atom_full[positions2] = one_hot_vector

    edges = torch.stack([positions1, positions2], dim=0)
    reversed_edges = edges.flip(0)

    sampled_edges = torch.cat([edges,reversed_edges], dim=1)
    guidance_edge_attr = self._get_edge_dist(X_true, sampled_edges, atom_mask)       
    guidance_edge_attr.fill_(4.5)      
    guidance_edge_attr = self.guidance_dist_rbf(guidance_edge_attr).view(sampled_edges.shape[1], -1)
    return atom_full,sampled_edges,guidance_edge_attr

def condition13(self,atom_gt,batch_ids,mask_generate,X_true,atom_mask):
    '''
    k-D/E i-i+3/i+4 distance 4-6.5
    and
    two positions Cys with distance between 3.5-5 A
    '''
    unique_vals = torch.unique(batch_ids)
    sampled_indices = []
    positions1 = []
    positions2 = []
    positions3 = []
    positions4 = []
    for val in unique_vals:
        valid_indices = (batch_ids == val) & mask_generate
        indices = valid_indices.nonzero(as_tuple=True)[0]
        if len(indices)<=10:
            continue
        random_indices = indices[(max(indices)-indices>=10)]
        indice = random.choice(random_indices)
        hop = random.choice([3,4])
        
        positions1.append(indice)
        positions2.append(indice+hop)
        
        head_indices = indices[(indices>indice+hop)&(max(indices)-indices>=3)]
        head_indice = random.choice(head_indices)

        sampled_indices+= [indice,indice+3]
        positions3.append(head_indice)
        positions4.append(head_indice+3)

    # Given the K-D\E guidance
    positions1 = torch.stack(positions1).to(atom_gt.device)
    positions2 = torch.stack(positions2).to(atom_gt.device)
    
    # control the type of K 
    atom_full = torch.zeros((mask_generate.shape[0],atom_gt.shape[1])).to(atom_gt.device)
    one_hot_vector = torch.zeros(20).to(atom_gt.device)
    one_hot_vector[11] = 1 # K
    one_hot_vector = one_hot_vector.unsqueeze(0).repeat(len(positions1),1)
    atom_full[positions1] = one_hot_vector

    # control the type of D/E
    atom_full = torch.zeros((mask_generate.shape[0],atom_gt.shape[1])).to(atom_gt.device)
    one_hot_vector = torch.zeros(20).to(atom_gt.device)
    if random.random()<0.5:
        one_hot_vector[3] = 1 # D
    else:
        one_hot_vector[5] = 1 # E
    one_hot_vector = one_hot_vector.unsqueeze(0).repeat(len(positions2),1)
    atom_full[positions2] = one_hot_vector

    edges = torch.stack([positions1, positions2], dim=0)
    reversed_edges = edges.flip(0)

    sampled_edges1 = torch.cat([edges,reversed_edges], dim=1)

    # Control the 
    sampled_indices = torch.tensor(sampled_indices).to(atom_gt.device)
    one_hot_vector = torch.zeros(20).to(atom_gt.device)
    one_hot_vector[4] = 1
    one_hot_vector = one_hot_vector.unsqueeze(0).repeat(len(sampled_indices),1)
    atom_full[sampled_indices] = one_hot_vector

    positions3 = torch.stack(positions3)
    positions4 = torch.stack(positions4)
    edges = torch.stack([positions3, positions4], dim=0)
    reversed_edges = edges.flip(0)

    sampled_edges2 = torch.cat([edges,reversed_edges], dim=1)

    sampled_edges = torch.cat([sampled_edges1,sampled_edges2],dim=1)

    guidance_edge_attr = self._get_edge_dist(X_true, sampled_edges, atom_mask)       
    guidance_edge_attr.fill_(4.5)      
    guidance_edge_attr = self.guidance_dist_rbf(guidance_edge_attr).view(sampled_edges.shape[1], -1)
    return atom_full,sampled_edges,guidance_edge_attr



def condition23(self,atom_gt,batch_ids,mask_generate,X_true,atom_mask):
    '''
    The distance of head and tail is less than 6 A and two positions Cys with distance between 3.5-5 A
    '''

    device = atom_gt.device
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
        sampled = [head_indice,head_indice+3]
        positions1.append(int(head_indice))
        positions2.append(int(head_indice+3))
        sampled_indices+=sampled
    
    for i in range(1, len(mask_generate)):
        if mask_generate[i] != mask_generate[i-1]:
            if mask_generate[i]:
                positions1.append(i)
            else:
                positions2.append(i-1)

    positions2.append(len(mask_generate)-1)
    
    sampled_indices = torch.tensor(sampled_indices).to(atom_gt.device)
    atom_full = torch.zeros((mask_generate.shape[0],atom_gt.shape[1])).to(atom_gt.device)
    one_hot_vector = torch.zeros(20).to(atom_gt.device)
    one_hot_vector[4] = 1
    one_hot_vector = one_hot_vector.unsqueeze(0).repeat(len(sampled_indices),1)
    atom_full[sampled_indices] = one_hot_vector

    positions1 = torch.tensor(positions1).to(device)
    positions2 = torch.tensor(positions2).to(device)
    edges = torch.stack([positions1, positions2], dim=0)
    reversed_edges = edges.flip(0)

    sampled_edges = torch.cat([edges,reversed_edges], dim=1)
    guidance_edge_attr = self._get_edge_dist(X_true, sampled_edges, atom_mask)       
    guidance_edge_attr.fill_(3.8)      
    guidance_edge_attr = self.guidance_dist_rbf(guidance_edge_attr).view(sampled_edges.shape[1], -1)

    return atom_full,sampled_edges,guidance_edge_attr


def condition3(self,atom_gt,batch_ids,mask_generate,X_true,atom_mask):
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
        sampled = [head_indice,head_indice+3]
        positions1.append(head_indice)
        positions2.append(head_indice+3)
        sampled_indices+=sampled
    sampled_indices = torch.tensor(sampled_indices).to(atom_gt.device)
    atom_full = torch.zeros((mask_generate.shape[0],atom_gt.shape[1])).to(atom_gt.device)
    one_hot_vector = torch.zeros(20).to(atom_gt.device)
    one_hot_vector[4] = 1
    one_hot_vector = one_hot_vector.unsqueeze(0).repeat(len(sampled_indices),1)
    atom_full[sampled_indices] = one_hot_vector

    positions1 = torch.stack(positions1)
    positions2 = torch.stack(positions2)
    edges = torch.stack([positions1, positions2], dim=0)
    reversed_edges = edges.flip(0)

    sampled_edges = torch.cat([edges,reversed_edges], dim=1)
    guidance_edge_attr = self._get_edge_dist(X_true, sampled_edges, atom_mask)       
    guidance_edge_attr.fill_(3.8)      
    guidance_edge_attr = self.guidance_dist_rbf(guidance_edge_attr).view(sampled_edges.shape[1], -1)

    return atom_full,sampled_edges,guidance_edge_attr

def condition33(self,atom_gt,batch_ids,mask_generate,X_true,atom_mask):
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
        
        if len(indices)<8:
            continue
        head_indices = indices[(max(indices)-indices>=7)]
        head_indice = random.choice(head_indices)
        sampled = [head_indice,head_indice+3]
        positions1.append(head_indice)
        positions2.append(head_indice+3)
        
        head_indices = indices[(indices>head_indice+3)&(max(indices)-indices>=3)]
        head_indice = random.choice(head_indices)
        sampled += [head_indice,head_indice+3]
        positions1.append(head_indice)
        positions2.append(head_indice+3)

        
        sampled_indices+=sampled
    sampled_indices = torch.tensor(sampled_indices).to(atom_gt.device)
    atom_full = torch.zeros((mask_generate.shape[0],atom_gt.shape[1])).to(atom_gt.device)
    one_hot_vector = torch.zeros(20).to(atom_gt.device)
    one_hot_vector[4] = 1
    one_hot_vector = one_hot_vector.unsqueeze(0).repeat(len(sampled_indices),1)
    atom_full[sampled_indices] = one_hot_vector

    positions1 = torch.stack(positions1)
    positions2 = torch.stack(positions2)
    edges = torch.stack([positions1, positions2], dim=0)
    reversed_edges = edges.flip(0)

    sampled_edges = torch.cat([edges,reversed_edges], dim=1)
    guidance_edge_attr = self._get_edge_dist(X_true, sampled_edges, atom_mask)       
    guidance_edge_attr.fill_(3.8)      
    guidance_edge_attr = self.guidance_dist_rbf(guidance_edge_attr).view(sampled_edges.shape[1], -1)

    return atom_full,sampled_edges,guidance_edge_attr

def condition333(self,atom_gt,batch_ids,mask_generate,X_true,atom_mask):
    '''
    two positions Cys with distance between 3.5-5 A *3
    '''
    unique_vals = torch.unique(batch_ids)
    sampled_indices = []
    positions1 = []
    positions2 = []
    for val in unique_vals:
        valid_indices = (batch_ids == val) & mask_generate
        indices = valid_indices.nonzero(as_tuple=True)[0]
        
        if len(indices)<12:
            continue
        head_indices = indices[(max(indices)-indices>=11)]
        head_indice = random.choice(head_indices)
        sampled = [head_indice,head_indice+3]
        positions1.append(head_indice)
        positions2.append(head_indice+3)
        
        head_indices = indices[(indices>head_indice+3)&(max(indices)-indices>=7)]
        head_indice = random.choice(head_indices)
        sampled += [head_indice,head_indice+3]
        positions1.append(head_indice)
        positions2.append(head_indice+3)
        sampled_indices+=sampled

        head_indices = indices[(indices>head_indice+3)&(max(indices)-indices>=3)]
        head_indice = random.choice(head_indices)
        sampled += [head_indice,head_indice+3]
        positions1.append(head_indice)
        positions2.append(head_indice+3)
        sampled_indices+=sampled
    sampled_indices = torch.tensor(sampled_indices).to(atom_gt.device)
    atom_full = torch.zeros((mask_generate.shape[0],atom_gt.shape[1])).to(atom_gt.device)
    one_hot_vector = torch.zeros(20).to(atom_gt.device)
    one_hot_vector[4] = 1
    one_hot_vector = one_hot_vector.unsqueeze(0).repeat(len(sampled_indices),1)
    atom_full[sampled_indices] = one_hot_vector

    positions1 = torch.stack(positions1)
    positions2 = torch.stack(positions2)
    edges = torch.stack([positions1, positions2], dim=0)
    reversed_edges = edges.flip(0)

    sampled_edges = torch.cat([edges,reversed_edges], dim=1)
    guidance_edge_attr = self._get_edge_dist(X_true, sampled_edges, atom_mask)       
    guidance_edge_attr.fill_(3.8)      
    guidance_edge_attr = self.guidance_dist_rbf(guidance_edge_attr).view(sampled_edges.shape[1], -1)

    return atom_full,sampled_edges,guidance_edge_attr

def condition4(self,atom_gt,batch_ids,mask_generate,X_true,atom_mask):
    '''
    Contruct Bicycle
    '''
    unique_vals = torch.unique(batch_ids)
    sampled_indices = []
    positions1 = []
    positions2 = []
    for val in unique_vals:
        valid_indices = (batch_ids == val) & mask_generate
        indices = valid_indices.nonzero(as_tuple=True)[0]
        if len(indices)<13:
            continue
        bicycle_indices = indices[(max(indices)-indices>=12)]
        bicycle_indice = random.choice(bicycle_indices)
        sampled = [bicycle_indice,bicycle_indice+6,bicycle_indice+12]
        positions1.append(bicycle_indice)
        positions1.append(bicycle_indice)
        positions1.append(bicycle_indice+6)
        positions2.append(bicycle_indice+6)
        positions2.append(bicycle_indice+12)
        positions2.append(bicycle_indice+12)
        sampled_indices+=sampled
    sampled_indices = torch.tensor(sampled_indices).to(atom_gt.device)
    atom_full = torch.zeros((mask_generate.shape[0],atom_gt.shape[1])).to(atom_gt.device)
    one_hot_vector = torch.zeros(20).to(atom_gt.device)
    one_hot_vector[4] = 1
    one_hot_vector = one_hot_vector.unsqueeze(0).repeat(len(sampled_indices),1)
    atom_full[sampled_indices] = one_hot_vector

    positions1 = torch.stack(positions1)
    positions2 = torch.stack(positions2)
    edges = torch.stack([positions1, positions2], dim=0)
    reversed_edges = edges.flip(0)

    sampled_edges = torch.cat([edges,reversed_edges], dim=1)
    guidance_edge_attr = self._get_edge_dist(X_true, sampled_edges, atom_mask)       
    guidance_edge_attr.fill_(8)      
    guidance_edge_attr = self.guidance_dist_rbf(guidance_edge_attr).view(sampled_edges.shape[1], -1)

    return atom_full,sampled_edges,guidance_edge_attr

    
