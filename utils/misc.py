class NestedTensor:

    def __init__(self, tensors, mask):
        self.tensors = tensors 
        self.mask = mask 

    def to(self, device):
        cast_tensors = self.tensors.to(device)
        if self.mask is not None:
            cast_mask = self.mask.to(device)
        else:
            cast_mask = None 

        return NestedTensor(cast_tensors, cast_mask)

    def decompose(self):
        return self.tensors, self.mask 

    def __repr__(self):
        return str(self.tensors)


