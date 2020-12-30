import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.modules import Lstm, CRF, Attention
from src.utils import load_embedding_from_npy, load_embedding_from_pkl
from src.datareader import SLOT_PAD, y2_set, domain_set, domain2slot, y1_set,father_son_slot, father_keys

class CoarseSLUTagger(nn.Module):
    def __init__(self, params, vocab, coarse_fine_map):
        super(CoarseSLUTagger, self).__init__()
        
        self.lstm = Lstm(params, vocab)
        self.num_binslot = params.num_binslot
        self.hidden_dim = params.hidden_dim * 2 if params.bidirection else params.hidden_dim
        self.atten_w = nn.Parameter(torch.Tensor(self.hidden_dim, params.emb_dim))
        
        self.linear = nn.Linear(self.hidden_dim, self.num_binslot)
        self.crf_layer = CRF(self.num_binslot)
        self.domain_coarse_mask = self.gen_emission_mask()
        self.example_emb = load_embedding_from_pkl(params.example_emb_file)
        self.slot_embs = load_embedding_from_pkl(params.slot_emb_file)
        self.coarse_fine_map = coarse_fine_map

        nn.init.xavier_normal_(self.atten_w)

    def forward(self, X, y_dm, iseval=False, lengths=None):
        """
        Input: 
            X: (bsz, seq_len)
        Output:
            prediction: (bsz, seq_len, num_binslot)
            lstm_hidden: (bsz, seq_len, hidden_size)
        """
        bsz, seq_len = X.size()
        lstm_hidden = self.lstm(X)  # (bsz, seq_len, hidden_dim)
        prediction = self.linear(lstm_hidden)

        all_mask = []
        
        for dm_id in y_dm:
            mask_vec = self.domain_coarse_mask[dm_id.item()]
            mask_vec = mask_vec.unsqueeze(0)
            all_mask.append(mask_vec.repeat(seq_len, 1))
        
        all_mask = torch.cat(all_mask, 0)
        all_mask = all_mask.view(bsz, seq_len, -1)
        all_mask = all_mask.float().cuda()

        if iseval == True:

            prediction = prediction + all_mask





        return prediction, lstm_hidden
    
    def crf_decode(self, inputs, lengths):
        """ crf decode
        Input:
            inputs: (bsz, seq_len, num_entity)
            lengths: lengths of x (bsz, )
        Ouput:
            crf_loss: loss of crf
        """
        prediction = self.crf_layer(inputs)
        prediction = [ prediction[i, :length].data.cpu().numpy() for i, length in enumerate(lengths) ]

        return prediction
    
    def crf_loss(self, inputs, lengths, y):
        """ create crf loss
        Input:
            inputs: (bsz, seq_len, num_entity)
            lengths: lengths of x (bsz, )
            y: label of slot value (bsz, seq_len)
        Ouput:
            crf_loss: loss of crf
        """
        padded_y = self.pad_label(lengths, y)

        crf_loss = self.crf_layer.loss(inputs, padded_y)

        return crf_loss

    def encode_slot_name(self, inputs, lengths):
        lengths = torch.tensor(lengths).cuda()-1
        lengths = lengths.unsqueeze(-1)

        lstm_hiddens = self.lstm(inputs)
        lengths = lengths.repeat(1, lstm_hiddens.size(-1))
        lengths = lengths.unsqueeze(1)

        emb = torch.gather(lstm_hiddens, 1, lengths)

        return emb
    
    def get_labelembedding(self, lstm_hiddens, lengths, y_dm):
        res_emb = []
        
        batch_example_emb = []

        for i in range(len(y_dm)):
            one_sent_lstm_hidden = lstm_hiddens[i]
            one_sent_length = lengths[i]
            dm = y_dm[i].item()
            domain_example_embs = self.example_emb[domain_set[dm]]
            domain_example_embs = torch.tensor(domain_example_embs)
            domain_example_embs = domain_example_embs.mean(-1)
            domain_example_embs =  domain_example_embs.float().cuda()
            one_sent_lstm_hidden = one_sent_lstm_hidden[:one_sent_length, :]
            atten_temp = torch.matmul(one_sent_lstm_hidden, self.atten_w)
            atten_temp = torch.matmul(atten_temp, domain_example_embs.transpose(0, 1))
            alpha = torch.softmax(atten_temp, -1)
            sum_emb = torch.matmul(one_sent_lstm_hidden.transpose(0,1), alpha).transpose(0, 1)
            batch_example_emb.append(sum_emb)

        # print(batch_example_emb[0].size())
        # print(self.coarse_fine_map)

        # print(self.slot_embs)

        for i in range(len(batch_example_emb)):
            temp_dict = {}
            for fa in father_keys:
                temp_dict[fa] = []
                dm = domain_set[y_dm[i].item()]
                for j in range(len(self.coarse_fine_map[dm][fa])):
                    slot_name = self.coarse_fine_map[dm][fa][j]
                    slot_id_in_domain = domain2slot[dm].index(slot_name)
                    slot_example_emb = batch_example_emb[i][slot_id_in_domain]

                    
                    slot_name_emb = self.slot_embs[dm][slot_id_in_domain]
                    slot_name_emb = torch.tensor(slot_name_emb).float().cuda()

                    slot_coarse_emb = torch.zeros(len(father_keys)).float().cuda()
                    fa_id = father_keys.index(fa)
                    slot_coarse_emb[fa_id] = 1

                    # emb = torch.cat([slot_example_emb, slot_name_emb, slot_coarse_emb], 0)


                    # temp_dict[fa].append(emb)

                    temp_dict[fa].append(slot_name_emb)


            res_emb.append(temp_dict)

        return res_emb 

    def pad_label(self, lengths, y):
        bsz = len(lengths)
        max_len = torch.max(lengths)
        padded_y = torch.LongTensor(bsz, max_len).fill_(SLOT_PAD)
        for i in range(bsz):
            length = lengths[i]
            y_i = y[i]
            padded_y[i, 0:length] = torch.LongTensor(y_i)

        padded_y = padded_y.cuda()
        return padded_y

    def pad_label2(self, lengths, y):
        y = [yy.cpu() for yy in y]
        lengths = lengths.cpu()
        bsz = len(lengths)
        max_len = torch.max(lengths)
        padded_y = torch.LongTensor(bsz, max_len).fill_(SLOT_PAD)
        for i in range(bsz):
            length = lengths[i]
            y_i = y[i]
            padded_y[i, 0:length] = torch.LongTensor(y_i)

        padded_y = padded_y.cuda()
        return padded_y

    def gen_emission_mask(self):
        mask = {}
        son_to_father = {}
        for k,v in father_son_slot.items():
            for w in v:
                son_to_father[w] = k
        


        for i in range(len(domain_set)):
            temp = [-1000000] * len(y1_set)
            temp[0] = 0
            for slot in domain2slot[domain_set[i]]:
                fa = son_to_father[slot]
                B_idx = y1_set.index('B-'+fa)
                I_idx = y1_set.index('I-'+fa)
                temp[B_idx] = 0
                temp[I_idx] = 0
            mask[i]= torch.tensor(temp).cuda()
        
        return mask
                    
class FinePredictor(nn.Module):
    def __init__(self, params, coarse_fine_map):
        super(FinePredictor, self).__init__()
        self.input_dim = params.hidden_dim * 2 if params.bidirection else params.hidden_dim
        self.enc_type = params.enc_type
        if self.enc_type == "trs":
            self.trs_enc = TransformerEncoder(input_size=self.input_dim, hidden_size=params.trs_hidden_dim, num_layers=params.trs_layers, num_heads=params.num_heads, dim_key=params.dim_key, dim_value=params.dim_value, filter_size=params.filter_size)
        elif self.enc_type == "lstm":
            self.lstm_enc = nn.LSTM(self.input_dim, params.trs_hidden_dim//2, num_layers=params.trs_layers, bidirectional=True, batch_first=True)
        
        self.slot_embs = load_embedding_from_pkl(params.slot_emb_file)
        self.slot_embs_list = self.get_emb_for_coarse_fine_map(domain2slot, self.slot_embs, coarse_fine_map)
        self.coarse_fine_map = coarse_fine_map


        self.similarity_W = nn.Parameter(torch.Tensor(400, params.emb_dim))
        nn.init.xavier_normal_(self.similarity_W)
        
    def get_emb_for_coarse_fine_map(self, domain2slot, slot_embs, coarse_fine_map):
        
        slot2emb = {}

        for k, v in domain2slot.items():
            for i, w in enumerate(v):
                slot2emb[w] = slot_embs[k][i]


        res_emb = {}
        for domain, coarse_dict in coarse_fine_map.items():
            res_emb[domain] = {}
            for father, son in coarse_dict.items():
                res_emb[domain][father] = []
                for s in son:
                    res_emb[domain][father].append(slot2emb[s])
                # res_emb[domain][father] = torch.tensor(res_emb[domain][father]).cuda()
        
        return res_emb

    def forward(self, domains, cur_coarse, hidden_layers, y_label_embedding, coarse_B_index,coarse_I_index, binary_preditions=None, binary_golds=None, final_golds=None):
        binary_labels = binary_golds if binary_golds is not None else binary_preditions
        feature_list = []
        gold_slotname_list = []
        coarse_fine_map = self.coarse_fine_map
        bsz = hidden_layers.size()[0]
        for i in range(bsz):
            dm_id = domains[i]
            domain_name = domain_set[dm_id]
            # slot_list_based_domain = domain2slot[domain_name]
            # seq_len x hidden_size
            hidden_i = hidden_layers[i]

            feature_each_sample = []
            
            if final_golds is not None:
                final_gold_each_sample = final_golds[i]
            gold_slotname_each_sample = []

            bin_label = binary_labels[i]
            bin_label = torch.LongTensor(bin_label)
            B_list = bin_label == coarse_B_index
            I_list = bin_label == coarse_I_index
            nonzero_B = torch.nonzero(B_list)
            num_slotname = nonzero_B.size()[0]
            
            if num_slotname == 0:
                feature_list.append(feature_each_sample)
                continue

            for j in range(num_slotname):
                if j == 0 and j < num_slotname - 1:
                    prev_index = nonzero_B[j]
                    continue

                curr_index = nonzero_B[j]
                if not (j == 0 and j == num_slotname-1):
                    nonzero_I = torch.nonzero(I_list[prev_index: curr_index])

                    if len(nonzero_I) != 0:
                        nonzero_I = (nonzero_I + prev_index).squeeze(1) # squeeze to one dimension
                        indices = torch.cat((prev_index, nonzero_I), dim=0)
                        hiddens_based_slotname = hidden_i[indices.unsqueeze(0)]   # (1, subseq_len, hidden_dim)
                    else:
                        # length of slot name is only 1
                        hiddens_based_slotname = hidden_i[prev_index.unsqueeze(0)]  # (1, 1, hidden_dim)

                    if self.enc_type == "trs":
                        slot_feats = self.trs_enc(hiddens_based_slotname)   # (1, subseq_len, hidden_dim)
                        slot_feats = torch.sum(slot_feats, dim=1) # (1, hidden_dim)
                    elif self.enc_type == "lstm":
                        slot_feats, (_, _) = self.lstm_enc(hiddens_based_slotname)   # (1, subseq_len, hidden_dim)
                        slot_feats = torch.sum(slot_feats, dim=1) # (1, hidden_dim)
                    else:
                        slot_feats = torch.sum(hiddens_based_slotname, dim=1) # (1, hidden_dim)

                    feature_each_sample.append(slot_feats.squeeze(0))
                    if final_golds is not None:
                        slot_name = y2_set[final_gold_each_sample[prev_index]].split("-")[1]

                        gold_slotname_each_sample.append(coarse_fine_map[domain_name][cur_coarse].index(slot_name))


                        # gold_slotname_each_sample.append(coarse_fine_map[domain_name][cur_coarse].index(slot_name))


                        # if slot_name in 
                    # if final_golds is not None:
                    #     slot_name = y2_set[final_gold_each_sample[prev_index]].split("-")[1]
                    #     gold_slotname_each_sample.append(slot_list_based_domain.index(slot_name))

                if j == num_slotname - 1:
                    nonzero_I = torch.nonzero(I_list[curr_index:])
                    if len(nonzero_I) != 0:
                        nonzero_I = (nonzero_I + curr_index).squeeze(1)  # squeeze to one dimension
                        indices = torch.cat((curr_index, nonzero_I), dim=0)
                        hiddens_based_slotname = hidden_i[indices.unsqueeze(0)]   # (1, subseq_len, hidden_dim)
                    else:
                        # length of slot name is only 1
                        hiddens_based_slotname = hidden_i[curr_index.unsqueeze(0)]  # (1, 1, hidden_dim)

                    if self.enc_type == "trs":
                        slot_feats = self.trs_enc(hiddens_based_slotname)   # (1, subseq_len, hidden_dim)
                        slot_feats = torch.sum(slot_feats, dim=1)  # (1, hidden_dim)
                    elif self.enc_type == "lstm":
                        slot_feats, (_, _) = self.lstm_enc(hiddens_based_slotname)   # (1, subseq_len, hidden_dim)
                        slot_feats = torch.sum(slot_feats, dim=1)  # (1, hidden_dim)
                    else:
                        slot_feats = torch.sum(hiddens_based_slotname, dim=1) # (1, hidden_dim)
                    # slot_feats = torch.sum(slot_feats, dim=1)
                    feature_each_sample.append(slot_feats.squeeze(0))   

                    if final_golds is not None:
                        slot_name = y2_set[final_gold_each_sample[curr_index]].split("-")[1]
                        gold_slotname_each_sample.append(coarse_fine_map[domain_name][cur_coarse].index(slot_name))
                        
                else:
                    prev_index = curr_index

            feature_each_sample = torch.stack(feature_each_sample)  # (num_slotname, hidden_dim)
            feature_list.append(feature_each_sample)
            if final_golds is not None:
                gold_slotname_each_sample = torch.LongTensor(gold_slotname_each_sample)   # (num_slotname)
                gold_slotname_list.append(gold_slotname_each_sample)
       
        pred_slotname_list = []
        for i in range(bsz):
            dm_id = domains[i]
            domain_name = domain_set[dm_id]
       
            if len(feature_list[i]) != 0 and len(self.slot_embs_list[domain_name][cur_coarse]) != 0:
       
                temp_label_embedding = [t.unsqueeze(1) for t in y_label_embedding[i][cur_coarse]["inputs"]]
                temp_label_embedding = torch.cat(temp_label_embedding, 1)
       
                slot_embs_based_domain = temp_label_embedding
           
                feature_each_sample = feature_list[i]

                temp = torch.matmul(slot_embs_based_domain.transpose(0, 1), self.similarity_W)
                temp = torch.matmul(temp, feature_each_sample.transpose(0,1)).transpose(0, 1)
                pred_slotname_each_sample = temp

            else:
                pred_slotname_each_sample = None
            
            pred_slotname_list.append(pred_slotname_each_sample)


        if final_golds is not None:
            return pred_slotname_list, gold_slotname_list
        else:
            
            return pred_slotname_list
                
class SentRepreGenerator(nn.Module):
    def __init__(self, params, vocab):
        super(SentRepreGenerator, self).__init__()

        self.hidden_size = params.hidden_dim * 2 if params.bidirection else params.hidden_dim

        self.template_encoder = Lstm(params, vocab)

        self.input_atten_layer = Attention(attention_size=self.hidden_size)
        self.template_attn_layer = Attention(attention_size=self.hidden_size)

    def forward(self, templates, tem_lengths, hidden_layers, x_lengths):
        """
        Inputs:
            templates: (bsz, 3, max_template_length)
            tem_lengths: (bsz,)
            hidden_layers: (bsz, max_length, hidden_size)
            x_lengths: (bsz,)
        Outputs:
            template_sent_repre: (bsz, 3, hidden_size)
            input_sent_repre: (bsz, hidden_size)
        """
        # generate templates sentence representation
        template0 = templates[:, 0, :]
        template1 = templates[:, 1, :]
        template2 = templates[:, 2, :]

        template0_hiddens = self.template_encoder(template0)
        template1_hiddens = self.template_encoder(template1)
        template2_hiddens = self.template_encoder(template2)

        template0_repre, _ = self.template_attn_layer(template0_hiddens, tem_lengths)
        template1_repre, _ = self.template_attn_layer(template1_hiddens, tem_lengths)
        template2_repre, _ = self.template_attn_layer(template2_hiddens, tem_lengths)

        templates_repre = torch.stack((template0_repre, template1_repre, template2_repre), dim=1)  # (bsz, 3, hidden_size)

        # generate input sentence representations
        input_repre, _ = self.input_atten_layer(hidden_layers, x_lengths)

        return templates_repre, input_repre
