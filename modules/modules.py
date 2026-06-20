# Calculates first order si
def first_order_si(data, labels):
    device = data.device
    labels = labels.reshape((-1,1))
    mat = torch.cdist(data, data)
    _, indices = torch.topk(mat, k=2, dim=0, largest=False)
    homolabels = (labels[indices[1]] == labels)
    si = homolabels.sum() / data.shape[0]
    return si

# Calculates high order si (order=2)
def high_order_si(data, labels):
    device = data.device
    labels = labels.reshape((-1,1))
    mat = torch.cdist(data, data)
    _, indices = torch.topk(mat, k=3, dim=0, largest=False)
    homolabels = (labels[indices[1]] == labels) * (labels[indices[2]] == labels)
    si = homolabels.sum() / data.shape[0]
    return si

# Calculates high order soft si (order=2)
def high_order_soft_si(data, labels):
    device = data.device
    labels = labels.reshape((-1,1))
    mat = torch.cdist(data, data)
    _, indices = torch.topk(mat, k=3, dim=0, largest=False)
    homolabels = (labels[indices[1]] == labels).sum(1) + (labels[indices[2]] == labels).sum(1)
    si = homolabels.sum() / 2 / data.shape[0]
    return si

# Calculates center based si
def center_based_si(data, labels):
    device = data.device
    labels = labels.reshape((-1,1))
    kinds_of_label = torch.unique(labels)
    centers = torch.zeros((kinds_of_label.shape[0], data.shape[1]), device=device)
    for i, label in enumerate(kinds_of_label):
        centers[i] = data[labels[:, 0] == label].mean(0)
    mat = torch.cdist(data, centers)
    _, indices = torch.topk(mat, k=1, dim=1, largest=False)
    homolabels = (kinds_of_label[indices] == labels)
    si = homolabels.sum() / data.shape[0]
    return si

# Calculates anti si (order=2)
def anti_si(data, labels):
    device = data.device
    labels = labels.reshape((-1,1))
    mat = torch.cdist(data, data)
    _, indices = torch.topk(mat, k=3, dim=0, largest=False)
    homolabels = (labels[indices[1]] != labels) * (labels[indices[2]] != labels)
    si = homolabels.sum() / data.shape[0]
    return si