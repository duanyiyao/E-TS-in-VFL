class Distribute_table:


    def __init__(self, data_loader, dims):

        self.data_owners = ["client"+str(i) for i in range(len(dims))]
        self.data_size = dims
        self.data_loader = data_loader
        self.data_pointer = []
        self.labels = []

        # iterate over each batch of dataloader for, 1) spliting image 2) sending to VirtualWorker
        for images, labels in self.data_loader:

            curr_data_dict = {}

            # calculate width and height according to the no. of workers for UNIFORM distribution
            
            self.labels.append(labels)

            # iterate over each worker for distribution of current batch of the self.data_loader
            for i, owner in enumerate(self.data_owners):
                if i == 0:
                # split the image and send it to VirtualWorker (which is supposed to be a dataowner or client)
                    image_part_ptr = images[:, :self.data_size[i]]
                    curr_data_dict[owner] = image_part_ptr
                else:
                    image_part_ptr = images[:, self.data_size[i-1]: self.data_size[i]]
                    curr_data_dict[owner] = image_part_ptr
           
            self.data_pointer.append(curr_data_dict)
            
    def __iter__(self):
        
        for data_ptr, label in zip(self.data_pointer[:-1], self.labels[:-1]):
            yield (data_ptr, label)
            
    def __len__(self):
        
        return len(self.data_loader)-1