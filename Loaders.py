### Created on Jan 9, 2024
### @author: Dr. Hani Z. Girgis, Bioinformatics Toolsmith Laboratory, Texas A&M University-Kingsville
### @author: Luis Solis, Bioinformatics Toolsmith Laboratory, Texas A&M University-Kingsville

import random
import os
import numpy as np
from tensorflow import keras

class SimDisTripletsLoader(keras.utils.Sequence):
    def __init__(self, f, rc, triplet_sim_file, triplet_dis_file, batch_size=32, is_generator=False, need_index=False, strand='B'):
        '''
        Loads a batch of similar and dissimilar triplets
        '''
        assert os.path.exists(triplet_sim_file), f'This similar triplet file {triplet_sim_file} does not exist.'
        assert os.path.exists(triplet_dis_file), f'This dissimilar triplet file {triplet_dis_file} does not exist.'
        assert len(f) == len(rc), f'Length disagreament between forward sequences and reverse complements {len(f)} vs {len(rc)}.'
        
        
        # Initialization
        self.f  = f
        self.rc = rc
        self.triplet_sim_file = triplet_sim_file
        self.triplet_dis_file = triplet_dis_file
        self.batch_size       = batch_size
        self.is_generator     = is_generator
        self.need_index       = need_index
        self.strand           = strand

        # Make input data
        triplet_sim_tensor = np.load(triplet_sim_file)
        triplet_dis_tensor = np.load(triplet_dis_file)
        
        #desired_length = triplet_dis_tensor.shape[1]
        #triplet_sim_tensor_shortened = triplet_sim_tensor[:, :desired_length, :]
        
        self.triplet_tensor = np.concatenate((triplet_sim_tensor, triplet_dis_tensor), axis=1)
        _, self.datalen, _ = self.triplet_tensor.shape
        
        assert self.datalen > 0, 'Invalid triplet tensor of size 0.'
        assert self.strand in ['F', 'RC', 'B'], 'Invalid Option for Type of Strand'

        # Check
        assert len(self.f) > self.triplet_tensor.max(), f'Out of range index. Database size: {len(self.f)}. Max index: {self.triplet_tensor.max()}'
        
        # Make labels
        # sim_label_array  = np.ones(self.datalen//2)
        # dis_label_array  = np.zeros(self.datalen//2)
        
        sim_label_array  = np.ones(triplet_sim_tensor.shape[1])
        dis_label_array  = np.zeros(triplet_dis_tensor.shape[1])
        
        self.label_array = np.concatenate((sim_label_array, dis_label_array))
        assert len(self.label_array) == self.datalen, f'Size of labels does not match that of data: {len(self.label_array)} vs. {self.datalen}.'
        
        self.channel , _, _ = self.triplet_tensor.shape
        self.epoch_num = np.random.choice(list(range(0, self.channel)))
        self.set_matrix_and_labels()
        
    def set_matrix_and_labels(self):
        '''
        Shuffle input along with labels
        '''
        permutation         = np.random.permutation(self.datalen)
        self.triplet_tensor = self.triplet_tensor[:, permutation, :]
        self.label_array    = self.label_array[permutation]
        self.matrix         = self.triplet_tensor[self.epoch_num, :, :]
        
        self.epoch_num = (self.epoch_num + 1) % self.channel
        
    def __getitem__(self, index):
        # Determine batch start and end
        batch_start = index*self.batch_size
        batch_end   = (index+1)*self.batch_size
        if batch_end > self.datalen:
            batch_end = self.datalen
        batch_size = batch_end - batch_start
    
        # Allocate tensors to hold f and y for the batch
        _, num_col = self.f.shape
        x_tensor = np.zeros((batch_size, 1, num_col, 3))
        y_tensor = np.zeros((batch_size, 1))
        
        # If any of these flags is true, then use the forward strand. Otherwise, use the reverse complement
        if self.strand == 'F':
            is_f_1 = True
            is_f_2 = True
            is_f_3 = True
        elif self.strand == 'RC':
            is_f_1 = False
            is_f_2 = False
            is_f_3 = False
        else: 
            is_f_1 = random.choice([True, False])
            is_f_2 = random.choice([True, False])
            is_f_3 = random.choice([True, False])
                
        
        
        # Collect sequences into the tensors
        if is_f_1:
            x_tensor[:, :, :, 0] = np.expand_dims(self.f[self.matrix[batch_start:batch_end, 0], :], axis=1)
        else:
            x_tensor[:, :, :, 0] = np.expand_dims(self.rc[self.matrix[batch_start:batch_end, 0], :], axis=1)
            
        if is_f_2:
            x_tensor[:, :, :, 1] = np.expand_dims(self.f[self.matrix[batch_start:batch_end, 1], :], axis=1)
        else:
            x_tensor[:, :, :, 1] = np.expand_dims(self.rc[self.matrix[batch_start:batch_end, 1], :], axis=1)
                
        if is_f_3:
            x_tensor[:, :, :, 2] = np.expand_dims(self.f[self.matrix[batch_start:batch_end, 2], :], axis=1)
        else:
            x_tensor[:, :, :, 2] = np.expand_dims(self.rc[self.matrix[batch_start:batch_end, 2], :], axis=1)
            
        y_tensor[:] = np.expand_dims(self.label_array[batch_start:batch_end], axis = 1)
        
        if self.is_generator:
            return x_tensor, x_tensor, 
        
        if self.need_index:
            return x_tensor, y_tensor, self.matrix[batch_start:batch_end, :]
        else:
            return x_tensor, y_tensor
    
    def __len__(self):
        '''
        Denotes the number of batches per epoch
        '''
        if self.datalen % self.batch_size == 0:
            return self.datalen // self.batch_size
        else:
            return 1 + (self.datalen // self.batch_size)
    
    def on_epoch_end(self):
        self.set_matrix_and_labels()


class SimDisPairsLoader(keras.utils.Sequence):
    def __init__(self, f, rc, triplet_sim_file, triplet_dis_file, batch_size=32, is_generator=False):
        '''
        Loads a batch of similar and dissimilar triplets
        '''
        assert os.path.exists(triplet_sim_file), f'This similar triplet file {triplet_sim_file} does not exist.'
        assert os.path.exists(triplet_dis_file), f'This dissimilar triplet file {triplet_dis_file} does not exist.'
        assert len(f) == len(rc), f'Length disagreament between forward sequences and reverse complements {len(f)} vs {len(rc)}.'
        
        # Initialization
        self.f  = f
        self.rc = rc
        self.triplet_sim_file = triplet_sim_file
        self.triplet_dis_file = triplet_dis_file
        self.batch_size       = batch_size
        self.is_generator     = is_generator

        # Make input data
        triplet_sim_tensor = np.load(triplet_sim_file)
        triplet_dis_tensor = np.load(triplet_dis_file)
        
        #desired_length = triplet_dis_tensor.shape[1]
        #triplet_sim_tensor_shortened = triplet_sim_tensor[:, :desired_length, :]
        
        self.triplet_tensor = np.concatenate((triplet_sim_tensor, triplet_dis_tensor), axis=1)
        _, self.datalen, _ = self.triplet_tensor.shape
        
        assert self.datalen > 0, 'Invalid triplet tensor of size 0.'

        # Check
        assert len(self.f) > self.triplet_tensor.max(), f'Out of range index. Database size: {len(self.f)}. Max index: {self.triplet_tensor.max()}'
        
        # Make labels
        # sim_label_array  = np.ones(self.datalen//2)
        # dis_label_array  = np.zeros(self.datalen//2)
        
        sim_label_array  = np.ones(triplet_sim_tensor.shape[1])
        dis_label_array  = np.zeros(triplet_dis_tensor.shape[1])
        
        self.label_array = np.concatenate((sim_label_array, dis_label_array))
        assert len(self.label_array) == self.datalen, f'Size of labels does not match that of data: {len(self.label_array)} vs. {self.datalen}.'
        
        self.channel , _, _ = self.triplet_tensor.shape
        self.epoch_num = np.random.choice(list(range(0, self.channel)))
        self.set_matrix_and_labels()
        
    def set_matrix_and_labels(self):
        '''
        Shuffle input along with labels
        '''
        permutation         = np.random.permutation(self.datalen)
        self.triplet_tensor = self.triplet_tensor[:, permutation, :]
        self.label_array    = self.label_array[permutation]
        self.matrix         = self.triplet_tensor[self.epoch_num, :, :]
        
        self.epoch_num = (self.epoch_num + 1) % self.channel
        
    def __getitem__(self, index):
        # Determine batch start and end
        batch_start = index*self.batch_size
        batch_end   = (index+1)*self.batch_size
        if batch_end > self.datalen:
            batch_end = self.datalen
        batch_size = batch_end - batch_start
    
        # Allocate tensors to hold f and y for the batch
        _, num_col = self.f.shape
        x_tensor = np.zeros((batch_size, 1, num_col, 2))          ######################################################3to2
        y_tensor = np.zeros((batch_size, 1))
        
        # If any of these flags is true, then use the forward strand. Otherwise, use the reverse complement
        is_f_1 = random.choice([True, False])
        is_f_2 = random.choice([True, False])
        is_f_3 = random.choice([True, False])
                
        # Collect sequences into the tensors
        if is_f_1:
            x_tensor[:, :, :, 0] = np.expand_dims(self.f[self.matrix[batch_start:batch_end, 0], :], axis=1)
        else:
            x_tensor[:, :, :, 0] = np.expand_dims(self.rc[self.matrix[batch_start:batch_end, 0], :], axis=1)
                           
        if is_f_3:
            x_tensor[:, :, :, 1] = np.expand_dims(self.f[self.matrix[batch_start:batch_end, 2], :], axis=1)
        else:
            x_tensor[:, :, :, 1] = np.expand_dims(self.rc[self.matrix[batch_start:batch_end, 2], :], axis=1)
            
        y_tensor[:] = np.expand_dims(self.label_array[batch_start:batch_end], axis = 1)
        
        if self.is_generator:
            return x_tensor, x_tensor
        
        return x_tensor, y_tensor
    
    def __len__(self):
        '''
        Denotes the number of batches per epoch
        '''
        if self.datalen % self.batch_size == 0:
            return self.datalen // self.batch_size
        else:
            return 1 + (self.datalen // self.batch_size)
    
    def on_epoch_end(self):
        self.set_matrix_and_labels()

   
class SingleSequenceLoader(keras.utils.Sequence):
    def __init__(self, f, rc, enhancer_file, control_file, batch_size=32, is_generator=False, need_index=False, can_shuffle=True, strand='B'):
        '''
        Loads a batch of similar and dissimilar triplets
        '''
        assert os.path.exists(enhancer_file), f'This enhancer file {enhancer_file} does not exist.'
        assert os.path.exists(control_file), f'This control file {control_file} does not exist.'
        assert len(f) == len(rc), f'Length disagreament between forward sequences and reverse complements {len(f)} vs. {len(rc)}.'
        
        # Initialization
        self.f  = f
        self.rc = rc
        self.enhancer_file    = enhancer_file
        self.control_file     = control_file
        self.batch_size       = batch_size
        self.is_generator     = is_generator
        self.need_index       = need_index
        self.can_shuffle      = can_shuffle
        self.strand           = strand

        # Make input data
        enhancer_tensor    = np.load(enhancer_file)
        control_tensor     = np.load(control_file)
        self.single_tensor = np.concatenate((enhancer_tensor, control_tensor))
        self.datalen       = len(self.single_tensor)
        
        assert self.datalen > 0, 'Invalid triplet tensor of size 0.'
        assert self.strand in ['F', 'RC', 'B'], 'Invalid option for type of strand. Valid options are F, RC, or B.'
        assert len(self.f) > self.single_tensor.max(), f'Out of range index. Database size: {len(self.f)}. Max index: {self.single_tensor.max()}'
        
        # Make labels
        enhancers_num = len(enhancer_tensor)
        control_num   = len(control_tensor)
        
        enhancer_label_array  = np.ones(enhancers_num)
        control_label_array  = np.zeros(control_num)
        self.label_array = np.concatenate((enhancer_label_array, control_label_array))
        assert len(self.label_array) == len(self.single_tensor), f'The number of labels does not match the number of samples: {len(self.label_array)} vs. {len(self.single_tensor)}'
        
        if self.can_shuffle:
            self.set_matrix_and_labels()
        
    def set_matrix_and_labels(self):
        '''
        Shuffle input along with labels
        '''
        permutation        = np.random.permutation(self.datalen)
        self.single_tensor = self.single_tensor[permutation]
        self.label_array   = self.label_array[permutation]
        
        # self.matrix        = self.single_tensor[self.epoch_num]
        # self.epoch_num = (self.epoch_num + 1) % self.channel
        #         print('Single Tensor',self.single_tensor.shape)
        #         print('Channel', self.channel)
        #         print('Label array',self.label_array.shape)
        #         print('datalen',self.datalen)
        
    def __getitem__(self, index):
        # Determine batch start and end
        batch_start = index*self.batch_size
        batch_end   = (index+1)*self.batch_size
        if batch_end > self.datalen:
            batch_end = self.datalen
        batch_size = batch_end - batch_start
    
        # Allocate tensors to hold f and y for the batch
        _, num_col = self.f.shape
        x_tensor   = np.zeros((batch_size, 1, num_col, 1))
        y_tensor   = np.zeros((batch_size, 1))
        
        # If any of these flags is true, then use the forward strand. Otherwise, use the reverse complement
            
        if self.strand == 'F':
            is_f_1 = True
        elif self.strand == 'RC':
            is_f_1 = False
        else: 
            is_f_1 = random.choice([True, False])
    
        #is_f_1 = random.choice([True, False])
        
        #is_f_1 = True
        #is_f_2 = True
        #is_f_3 = True
        
        # Collect sequences into the tensors
        if is_f_1:
            x_tensor[:, :, :, 0] = np.expand_dims(self.f[self.single_tensor[batch_start:batch_end], :], axis=1)
        else:
            x_tensor[:, :, :, 0] = np.expand_dims(self.rc[self.single_tensor[batch_start:batch_end], :], axis=1)
            
        y_tensor[:] = np.expand_dims(self.label_array[batch_start:batch_end], axis=1)
        
        if self.is_generator:
            return x_tensor, x_tensor
            
        if self.need_index:
            return x_tensor, y_tensor, self.single_tensor[batch_start:batch_end], self.label_array[batch_start:batch_end]
        
        return x_tensor, y_tensor
    
    def __len__(self):
        '''
        Denotes the number of batches per epoch
        '''
        if self.datalen % self.batch_size == 0:
            return self.datalen // self.batch_size
        else:
            return 1 + (self.datalen // self.batch_size)
    
    def on_epoch_end(self):
        self.set_matrix_and_labels()
        
class CombinedSimDisTripletsLoader(keras.utils.Sequence):
    def __init__(self, f, rc, triplet_sim_list, triplet_dis_list, batch_size=32, is_generator=False, need_index=False, strand='B'):
        '''
        An instance of this class loads batches from multiple data sets
        f: A matrix representing forward strands of sequence
        rc: A matrix representing reverse completes of sequences
        triplet_sim_list and triplet_dis_list: Lists including equal number of numpy arrays of similar and dissimilar triplets
        '''
        assert len(triplet_sim_list) == len(triplet_dis_list), f'Length of triplet similar and triplet dissimilar are not the same: {len(triplet_sim_list)} vs. {len(triplet_dis_list)}'
        assert batch_size % len(triplet_sim_list) == 0, f'A batch size must be a multiple of the number of the component loaders'
        
        # A list of the component loaders
        self.n = len(triplet_sim_list)
        self.loader_list = []
        self.batch_size = batch_size //  self.n
        self.need_index = need_index
        self.strand     = strand
        
        
        assert self.strand in ['F', 'RC', 'B'], 'Invalid Option for Type of Strand'
        
        for sim_triplet_file, dis_triplet_file in zip(triplet_sim_list, triplet_dis_list):
            self.loader_list.append(SimDisTripletsLoader(f, rc, sim_triplet_file, dis_triplet_file, self.batch_size, is_generator, self.need_index, self.strand))
        
        assert len(self.loader_list) == self.n, f'Lenght mismatch between loaders and files: {len(self.loader_list)} vs. {self.n}'
        assert self.strand in ['F', 'RC', 'B'], 'Invalid Option for Type of Strand'
                
    def __getitem__(self, index):
        r1_list = []
        r2_list = []

        for loader in self.loader_list:
            r1, r2 = loader.__getitem__(index)            
            r1_list.append(r1)
            r2_list.append(r2)

        return np.concatenate(r1_list), np.concatenate(r2_list)

    def __len__(self):
        assert len(self.loader_list) > 0, 'The loader list is empty'
        return len(self.loader_list[0])

class CombinedSimDisPairsLoader(keras.utils.Sequence):
    def __init__(self, f, rc, triplet_sim_list, triplet_dis_list, batch_size=32, is_generator=False):
        '''
        An instance of this class loads batches from multiple data sets
        f: A matrix representing forward strands of sequence
        rc: A matrix representing reverse completes of sequences
        triplet_sim_list and triplet_dis_list: Lists including equal number of numpy arrays of similar and dissimilar triplets
        '''
        assert len(triplet_sim_list) == len(triplet_dis_list), f'Length of triplet similar and triplet dissimilar are not the same: {len(triplet_sim_list)} vs. {len(triplet_dis_list)}'
        assert batch_size % len(triplet_sim_list) == 0, f'A batch size must be a multiple of the number of the component loaders'
        
        # A list of the component loaders
        self.n = len(triplet_sim_list)
        self.loader_list = []
        self.batch_size = batch_size //  self.n
        print(self.n, self.batch_size)
        
        for sim_triplet_file, dis_triplet_file in zip(triplet_sim_list, triplet_dis_list):
            self.loader_list.append(SimDisPairsLoader(f, rc, sim_triplet_file, dis_triplet_file, self.batch_size, is_generator))
        
        assert len(self.loader_list) == self.n, f'Lenght mismatch between loaders and files: {len(self.loader_list)} vs. {self.n}'
                
    def __getitem__(self, index):
        r1_list = []
        r2_list = []

        for loader in self.loader_list:
            r1, r2 = loader.__getitem__(index)            
            r1_list.append(r1)
            r2_list.append(r2)

        return np.concatenate(r1_list), np.concatenate(r2_list)

    def __len__(self):
        assert len(self.loader_list) > 0, 'The loader list is empty'
        return len(self.loader_list[0])


class CombinedControlEnhancerLoader(keras.utils.Sequence):
    def __init__(self, f, rc, triplet_sim_list, triplet_dis_list, batch_size=32, is_generator=False, need_index=False, can_shuffle=True, strand='B'):
        '''
        An instance of this class loads batches from multiple data sets
        f: A matrix representing forward strands of sequence
        rc: A matrix representing reverse completes of sequences
        triplet_sim_list and triplet_dis_list: Lists including equal number of numpy arrays of similar and dissimilar triplets
        '''
        assert len(triplet_sim_list) == len(triplet_dis_list), f'Length of triplet similar and triplet dissimilar are not the same: {len(triplet_sim_list)} vs. {len(triplet_dis_list)}'
        assert batch_size % len(triplet_sim_list) == 0, f'A batch size must be a multiple of the number of the component loaders'

        self.can_shuffle = can_shuffle
        self.strand      = strand
        self.need_index  = need_index

        assert self.strand in ['F', 'RC', 'B'], 'Invalid Option for Type of Strand'
        
        # A list of the component loaders
        self.n = len(triplet_sim_list)
        self.loader_list = []
        self.batch_size = batch_size //  self.n
        (self.batch_size)
        
        for sim_triplet_file, dis_triplet_file in zip(triplet_sim_list, triplet_dis_list):
            self.loader_list.append(SingleSequenceLoader(f, rc, sim_triplet_file, dis_triplet_file, self.batch_size, is_generator, self.need_index, self.can_shuffle, self.strand))
        
        assert len(self.loader_list) == self.n, f'Lenght mismatch between loaders and files: {len(self.loader_list)} vs. {self.n}'
                
    def __getitem__(self, index):
        r1_list = []
        r2_list = []
        idx_list = []
        label_list = []
    
        for loader in self.loader_list:
            result = loader.__getitem__(index)
    
            if self.need_index:
                r1, r2, idx, lbl = result
                idx_list.append(idx)
                label_list.append(lbl)
            else:
                r1, r2 = result
    
            r1_list.append(r1)
            r2_list.append(r2)
    
        if self.need_index:
            return (
                np.concatenate(r1_list),
                np.concatenate(r2_list),
                np.concatenate(idx_list),
                np.concatenate(label_list),
            )
        else:
            return np.concatenate(r1_list), np.concatenate(r2_list)

    def __len__(self):
        assert len(self.loader_list) > 0, 'The loader list is empty'
        return len(self.loader_list[0])

class SimDisTripletsTwoClassLoader(keras.utils.Sequence):
    def __init__(self, f, rc, triplet_sim_file, triplet_dis_file, batch_size=32, is_generator=False, need_index=False, strand='B'):
        '''
        Loads a batch of similar and dissimilar triplets
        '''
        assert os.path.exists(triplet_sim_file), f'This similar triplet file {triplet_sim_file} does not exist.'
        assert os.path.exists(triplet_dis_file), f'This dissimilar triplet file {triplet_dis_file} does not exist.'
        assert len(f) == len(rc), f'Length disagreament between forward sequences and reverse complements {len(f)} vs {len(rc)}.'
        
        
        # Initialization
        self.f  = f
        self.rc = rc
        self.triplet_sim_file = triplet_sim_file
        self.triplet_dis_file = triplet_dis_file
        self.batch_size       = batch_size
        self.is_generator     = is_generator
        self.need_index       = need_index
        self.strand           = strand

        # Make input data
        triplet_sim_tensor = np.load(triplet_sim_file)
        triplet_dis_tensor = np.load(triplet_dis_file)
        
        #desired_length = triplet_dis_tensor.shape[1]
        #triplet_sim_tensor_shortened = triplet_sim_tensor[:, :desired_length, :]
        
        self.triplet_tensor = np.concatenate((triplet_sim_tensor, triplet_dis_tensor), axis=1)
        _, self.datalen, _ = self.triplet_tensor.shape
        
        assert self.datalen > 0, 'Invalid triplet tensor of size 0.'
        assert self.strand in ['F', 'RC', 'B'], 'Invalid Option for Type of Strand'

        # Check
        assert len(self.f) > self.triplet_tensor.max(), f'Out of range index. Database size: {len(self.f)}. Max index: {self.triplet_tensor.max()}'
        
        # Make labels
        # sim_label_array  = np.ones(self.datalen//2)
        # dis_label_array  = np.zeros(self.datalen//2)
        
        sim_label_array  = np.zeros((triplet_sim_tensor.shape[1], 2))
        sim_label_array[:, 0] = 1 
        dis_label_array  = np.zeros((triplet_dis_tensor.shape[1], 2))
        dis_label_array[:, 1] = 1 
        
        self.label_array = np.concatenate((sim_label_array, dis_label_array))
        assert len(self.label_array) == self.datalen, f'Size of labels does not match that of data: {len(self.label_array)} vs. {self.datalen}.'
        
        self.channel , _, _ = self.triplet_tensor.shape
        self.epoch_num = np.random.choice(list(range(0, self.channel)))
        self.set_matrix_and_labels()
    def set_matrix_and_labels(self):
        '''
        Shuffle input along with labels
        '''
        permutation         = np.random.permutation(self.datalen)
        self.triplet_tensor = self.triplet_tensor[:, permutation, :]
        self.label_array    = self.label_array[permutation]
        self.matrix         = self.triplet_tensor[self.epoch_num, :, :]
        
        self.epoch_num = (self.epoch_num + 1) % self.channel
        
    def __getitem__(self, index):
        # Determine batch start and end
        batch_start = index*self.batch_size
        batch_end   = (index+1)*self.batch_size
        if batch_end > self.datalen:
            batch_end = self.datalen
        batch_size = batch_end - batch_start
    
        # Allocate tensors to hold f and y for the batch
        _, num_col = self.f.shape
        x_tensor = np.zeros((batch_size, 1, num_col, 3))
        y_tensor = np.zeros((batch_size, 2))
        
        # If any of these flags is true, then use the forward strand. Otherwise, use the reverse complement
        
        if self.strand == 'F':
            is_f_1 = True
            is_f_2 = True
            is_f_3 = True
        elif self.strand == 'RC':
            is_f_1 = False
            is_f_2 = False
            is_f_3 = False
        else: 
            is_f_1 = random.choice([True, False])
            is_f_2 = random.choice([True, False])
            is_f_3 = random.choice([True, False])       
        
        # Collect sequences into the tensors
        if is_f_1:
            x_tensor[:, :, :, 0] = np.expand_dims(self.f[self.matrix[batch_start:batch_end, 0], :], axis=1)
        else:
            x_tensor[:, :, :, 0] = np.expand_dims(self.rc[self.matrix[batch_start:batch_end, 0], :], axis=1)
            
        if is_f_2:
            x_tensor[:, :, :, 1] = np.expand_dims(self.f[self.matrix[batch_start:batch_end, 1], :], axis=1)
        else:
            x_tensor[:, :, :, 1] = np.expand_dims(self.rc[self.matrix[batch_start:batch_end, 1], :], axis=1)
                
        if is_f_3:
            x_tensor[:, :, :, 2] = np.expand_dims(self.f[self.matrix[batch_start:batch_end, 2], :], axis=1)
        else:
            x_tensor[:, :, :, 2] = np.expand_dims(self.rc[self.matrix[batch_start:batch_end, 2], :], axis=1)
            
        #y_tensor[:] = np.expand_dims(self.label_array[batch_start:batch_end], axis = 1)
        y_tensor[:] = self.label_array[batch_start:batch_end]
        
        if self.is_generator:
            return x_tensor, x_tensor, 
        
        if self.need_index:
            return x_tensor, y_tensor, self.matrix[batch_start:batch_end, :]
        else:
            return x_tensor, y_tensor
    
    def __len__(self):
        '''
        Denotes the number of batches per epoch
        '''
        if self.datalen % self.batch_size == 0:
            return self.datalen // self.batch_size
        else:
            return 1 + (self.datalen // self.batch_size)
    
    def on_epoch_end(self):
        self.set_matrix_and_labels()

class CombinedSimDisTripletsTwoClassLoader(keras.utils.Sequence):
    def __init__(self, f, rc, triplet_sim_list, triplet_dis_list, batch_size=32, is_generator=False, need_index=False, strand='B'):
        '''
        An instance of this class loads batches from multiple data sets
        f: A matrix representing forward strands of sequence
        rc: A matrix representing reverse completes of sequences
        triplet_sim_list and triplet_dis_list: Lists including equal number of numpy arrays of similar and dissimilar triplets
        '''
        assert len(triplet_sim_list) == len(triplet_dis_list), f'Length of triplet similar and triplet dissimilar are not the same: {len(triplet_sim_list)} vs. {len(triplet_dis_list)}'
        assert batch_size % len(triplet_sim_list) == 0, f'A batch size must be a multiple of the number of the component loaders'
        
        # A list of the component loaders
        self.n = len(triplet_sim_list)
        self.loader_list = []
        self.batch_size = batch_size //  self.n
        self.need_index = need_index
        self.strand     = strand
        
        
        assert self.strand in ['F', 'RC', 'B'], 'Invalid Option for Type of Strand'
        
        for sim_triplet_file, dis_triplet_file in zip(triplet_sim_list, triplet_dis_list):
            self.loader_list.append(SimDisTripletsTwoClassLoader(f, rc, sim_triplet_file, dis_triplet_file, self.batch_size, is_generator, self.need_index, self.strand))
        
        assert len(self.loader_list) == self.n, f'Lenght mismatch between loaders and files: {len(self.loader_list)} vs. {self.n}'
        assert self.strand in ['F', 'RC', 'B'], 'Invalid Option for Type of Strand'
                
    def __getitem__(self, index):
        r1_list = []
        r2_list = []

        for loader in self.loader_list:
            r1, r2 = loader.__getitem__(index)            
            r1_list.append(r1)
            r2_list.append(r2)

        return np.concatenate(r1_list), np.concatenate(r2_list)

    def __len__(self):
        assert len(self.loader_list) > 0, 'The loader list is empty'
        return len(self.loader_list[0])


class EnhancerControlCAMLoader(keras.utils.Sequence):
    def __init__(self, f, rc, triplet_sim_file, triplet_dis_file, batch_size=32, is_generator=False, need_index=False, strand='B'):
        '''
        Loads a batch of similar and dissimilar triplets
        '''
        assert os.path.exists(triplet_sim_file), f'This similar triplet file {triplet_sim_file} does not exist.'
        assert os.path.exists(triplet_dis_file), f'This dissimilar triplet file {triplet_dis_file} does not exist.'
        assert len(f) == len(rc), f'Length disagreament between forward sequences and reverse complements {len(f)} vs {len(rc)}.'
        
        
        # Initialization
        self.f  = f
        self.rc = rc
        self.triplet_sim_file = triplet_sim_file
        self.triplet_dis_file = triplet_dis_file
        self.batch_size       = batch_size
        self.is_generator     = is_generator
        self.need_index       = need_index
        self.strand           = strand

        # Make input data
        triplet_sim_tensor = np.load(triplet_sim_file)
        triplet_dis_tensor = np.load(triplet_dis_file)
        
        #desired_length = triplet_dis_tensor.shape[1]
        #triplet_sim_tensor_shortened = triplet_sim_tensor[:, :desired_length, :]
        
        self.triplet_tensor = np.concatenate((triplet_sim_tensor, triplet_dis_tensor), axis=1)
        _, self.datalen, _ = self.triplet_tensor.shape
        
        assert self.datalen > 0, 'Invalid triplet tensor of size 0.'
        assert self.strand in ['F', 'RC', 'B'], 'Invalid Option for Type of Strand'

        # Check
        assert len(self.f) > self.triplet_tensor.max(), f'Out of range index. Database size: {len(self.f)}. Max index: {self.triplet_tensor.max()}'
        
        # Make labels
        # sim_label_array  = np.ones(self.datalen//2)
        # dis_label_array  = np.zeros(self.datalen//2)
        
        sim_label_array  = np.zeros((triplet_sim_tensor.shape[1], 2))
        sim_label_array[:, 0] = 1 
        dis_label_array  = np.zeros((triplet_dis_tensor.shape[1], 2))
        dis_label_array[:, 1] = 1 
        
        self.label_array = np.concatenate((sim_label_array, dis_label_array))
        assert len(self.label_array) == self.datalen, f'Size of labels does not match that of data: {len(self.label_array)} vs. {self.datalen}.'
        
        self.channel , _, _ = self.triplet_tensor.shape
        self.epoch_num = np.random.choice(list(range(0, self.channel)))
        self.set_matrix_and_labels()
    def set_matrix_and_labels(self):
        '''
        Shuffle input along with labels
        '''
        permutation         = np.random.permutation(self.datalen)
        self.triplet_tensor = self.triplet_tensor[:, permutation, :]
        self.label_array    = self.label_array[permutation]
        self.matrix         = self.triplet_tensor[self.epoch_num, :, :]
        
        self.epoch_num = (self.epoch_num + 1) % self.channel
        
    def __getitem__(self, index):
        # Determine batch start and end
        batch_start = index*self.batch_size
        batch_end   = (index+1)*self.batch_size
        if batch_end > self.datalen:
            batch_end = self.datalen
        batch_size = batch_end - batch_start
    
        # Allocate tensors to hold f and y for the batch
        _, num_col = self.f.shape
        x_tensor = np.zeros((batch_size, 1, num_col, 3))
        y_tensor = np.zeros((batch_size, 2))
        
        # If any of these flags is true, then use the forward strand. Otherwise, use the reverse complement
        
        if self.strand == 'F':
            is_f_1 = True
            is_f_2 = True
            is_f_3 = True
        elif self.strand == 'RC':
            is_f_1 = False
            is_f_2 = False
            is_f_3 = False
        else: 
            is_f_1 = random.choice([True, False])
            is_f_2 = random.choice([True, False])
            is_f_3 = random.choice([True, False])       
        
        # Collect sequences into the tensors
        if is_f_1:
            x_tensor[:, :, :, 0] = np.expand_dims(self.f[self.matrix[batch_start:batch_end, 0], :], axis=1)
        else:
            x_tensor[:, :, :, 0] = np.expand_dims(self.rc[self.matrix[batch_start:batch_end, 0], :], axis=1)
            
        if is_f_2:
            x_tensor[:, :, :, 1] = np.expand_dims(self.f[self.matrix[batch_start:batch_end, 1], :], axis=1)
        else:
            x_tensor[:, :, :, 1] = np.expand_dims(self.rc[self.matrix[batch_start:batch_end, 1], :], axis=1)
                
        if is_f_3:
            x_tensor[:, :, :, 2] = np.expand_dims(self.f[self.matrix[batch_start:batch_end, 2], :], axis=1)
        else:
            x_tensor[:, :, :, 2] = np.expand_dims(self.rc[self.matrix[batch_start:batch_end, 2], :], axis=1)
            
        #y_tensor[:] = np.expand_dims(self.label_array[batch_start:batch_end], axis = 1)
        y_tensor[:] = self.label_array[batch_start:batch_end]
        
        if self.is_generator:
            return x_tensor, x_tensor, 
        
        if self.need_index:
            return x_tensor, y_tensor, self.matrix[batch_start:batch_end, :]
        else:
            return x_tensor, y_tensor
    
    def __len__(self):
        '''
        Denotes the number of batches per epoch
        '''
        if self.datalen % self.batch_size == 0:
            return self.datalen // self.batch_size
        else:
            return 1 + (self.datalen // self.batch_size)
    
    def on_epoch_end(self):
        self.set_matrix_and_labels()
        
# class ForwardCombinedSimDisTripletsLoader(keras.utils.Sequence):
#     def __init__(self, f, triplet_sim_list, triplet_dis_list, batch_size=32, is_generator=False):
#         '''
#         An instance of this class loads batches from multiple data sets
#         f: A matrix representing forward strands of sequence
#         rc: A matrix representing reverse completes of sequences
#         triplet_sim_list and triplet_dis_list: Lists including equal number of numpy arrays of similar and dissimilar triplets
#         '''
#         assert len(triplet_sim_list) == len(triplet_dis_list), f'Length of triplet similar and triplet dissimilar are not the same: {len(triplet_sim_list)} vs. {len(triplet_dis_list)}'
#         assert batch_size % len(triplet_sim_list) == 0, f'A batch size must be a multiple of the number of the component loaders'
        
#         # A list of the component loaders
#         self.n = len(triplet_sim_list)
#         self.loader_list = []
#         self.batch_size = batch_size //  self.n
        
#         for sim_triplet_file, dis_triplet_file in zip(triplet_sim_list, triplet_dis_list):
#             self.loader_list.append(ForwardSimDisTripletsLoader(f, sim_triplet_file, dis_triplet_file, self.batch_size, is_generator))
        
#         assert len(self.loader_list) == self.n, f'Lenght mismatch between loaders and files: {len(self.loader_list)} vs. {self.n}'
                
#     def __getitem__(self, index):
#         r1_list = []
#         r2_list = []

#         for loader in self.loader_list:
#             r1, r2 = loader.__getitem__(index)            
#             r1_list.append(r1)
#             r2_list.append(r2)

#         return np.concatenate(r1_list), np.concatenate(r2_list)

#     def __len__(self):
#         assert len(self.loader_list) > 0, 'The loader list is empty'
#         return len(self.loader_list[0])