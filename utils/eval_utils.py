import torch
from torchmetrics.functional.retrieval import retrieval_average_precision
from pytorch_metric_learning.distances import LpDistance
from torch import linalg as LA
from torch.utils.data import Dataset, DataLoader
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def compute_sim(query, data,metric="cosine"):
    ''' Takes as inpute two torch tensors, if cosine is false, will return the scalar product'''
    if(metric=="product"):
        aff = torch.mm(query, data.t()) #Scalar product
        return aff
    elif(metric=="cosine"):
        aff = torch.mm(query, data.t())
        norm_a = LA.vector_norm(query,dim=-1).unsqueeze(-1)
        norm_b = LA.vector_norm(data, dim=1).unsqueeze(-1)
        x_norm_mul = torch.mm(norm_a, norm_b.t())
        sim_mat = aff / x_norm_mul
        return sim_mat
    elif(metric=="l2"):
        dist = LpDistance()
        sim_mat = dist(query, data)
        return sim_mat
    else:
        raise(NotImplementedError())

def AP(query, query_lab, vec_mat, labels, metric = "cosine", top = None):
  ap = []
  with torch.no_grad():
    indices =  list(range(len(query)))
    for i in indices:
      q = query[i]
      #q =  q.repeat(len(labels), 1)
      sim = compute_sim(q.unsqueeze(0), vec_mat, metric).squeeze()
      #labels = torch.tensor(labels)
      relevances = (labels == query_lab[i]).to(device)
      relevances = relevances[indices != i]
      sim = sim[indices != i]
      ap.append(retrieval_average_precision(sim, relevances, top_k=16).item())
  return ap

def retrieval_performances(test_set, model, map_score, iteration, batch_size = 100, current_classes=[], network="icarl"):
    model.eval()
    model.to(device)
    all_feats, all_labels = [], []
    with torch.no_grad():
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        if network == "icarl":
            extaractor = model.feature_extractor
        else:
            model.fc = torch.nn.Identity()
            extaractor = model
        for patterns, labels in test_loader:
            patterns = patterns.to(device)
            labels = labels.to(device)
            feats = extaractor(patterns.to(device))
            all_feats.append(feats.cpu())
            all_labels.append(labels)
    feats = torch.cat(all_feats)
    # print("feats shape", feats.shape)
    labels = torch.cat(all_labels).int()
    # print("labels shape", labels.shape)
    print("retrieval performances:")
    ap  = torch.tensor(AP(feats, labels, feats, labels))
    if current_classes != []:
        print("current classes type", type(current_classes))
        current_classes_idx = torch.isin(labels, current_classes)
        print("feats_current shape", current_classes_idx.sum(), " / ", len(ap))   
        map_score[iteration, 1] = torch.mean(ap[current_classes_idx])*100
        print("   MAP current classes:", map_score[iteration, 1])

    map_score[iteration, 0] = torch.mean(ap)*100

    print("   MAP cumul classes:", map_score[iteration, 0])
    return map_score