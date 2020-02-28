
def iterate_batched(tensor, batch_size=None):
    batch_size = batch_size or tensor.shape[0]
    if batch_size >= tensor.shape[0]:
        yield tensor
    else:
        num_groups = int(tensor.shape[0] / batch_size)
        for g in range(num_groups):
            yield tensor[g::num_groups]
