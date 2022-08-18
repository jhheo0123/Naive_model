import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#### 다시 업로드 해보기
class NaiveNet(nn.Module):
    def __init__(self, vocab_size, ddi_adj, emb_dim=64, device = torch.device('cpu:0')):
        super(NaiveNet,self).__init__()
        K = len(vocab_size)
        self.k = K
        self.vocab_size = vocab_size
        self.device = device
        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim*2) for i in range(K-1)])
        self.dropout = nn.Dropout(p=0.4)
        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 4, emb_dim*2),
        )
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, vocab_size[2])
        )
        self.init_weights()
        
    def forward(self, input):
        i1_seq = []
        i2_seq = []
        
        def mean_embedding(embedding):
            return embedding.mean(dim=1).unsqueeze(dim=0)
        
        for adm in input:
            i1 = mean_embedding(self.dropout(self.embeddings[0](torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device)))) # (1,1,dim)
            i2 = mean_embedding(self.dropout(self.embeddings[1](torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device))))
            i1_seq.append(i1)
            i2_seq.append(i2)
        
        
        
        

        i1_seq = torch.cat(i1_seq, dim=1) #(1,seq,dim)
        i2_seq = torch.cat(i2_seq, dim=1) #(1,seq,dim)
        
        # 평균 방법으로 합치기
        # i1_seq_mean  = torch.mean(i1_seq,1,True)
        # i2_seq_mean  = torch.mean(i2_seq,1,True)
        
        # sum 방법으로 합치기 
        # i1_seq_sum = torch.sum(i1_seq, dim =1)
        # i2_seq_sum = torch.sum(i2_seq, dim =1)
        
        # concat 방법으로 합치기
        def reduce_embedding(embedding):
            admission_count = embedding.size()[0]
            reduce_layer = nn.Sequential(nn.ReLU(), nn.Linear(admission_count, 1))
            return reduce_layer(embedding.transpose(0,1))
        
        
        # patient_representations = torch.cat([i1_seq_mean, i2_seq_mean], dim=-1).squeeze(dim=0)
        patient_representations = torch.cat([i1_seq, i2_seq], dim=-1).squeeze(dim=0)
        
        reduced_patient_representations = reduce_embedding(patient_representations).transpose(0,1) # concat할 때
        
        queries = self.query(reduced_patient_representations)
        output = self.output(queries)
        
        # if self.training:
        #     neg_pred_prob = F.sigmoid(output)
        #     neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (voc_size, voc_size)
        #     batch_neg = neg_pred_prob.mul(self.tensor_ddi_adj).mean()

        #     return output, batch_neg
        # else:
        
        # return output.unsqueeze(dim=0) # sum을 할 때
        return output
        
    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)

        