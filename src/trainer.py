from src.dataloader import get_dataloader
from src.datareader import y1_set, father_son_slot, domain2slot, father_keys, y0_set
from src.utils import init_experiment
from src.model import *
from config import get_params
from tqdm import tqdm
from preprocess.gen_embeddings_for_slu import *
import os
from tqdm import tqdm
import numpy as np
import logging

logger = logging.getLogger()

from conll2002_metrics import *
import conll
import conlleval



class SLUTrainer(object):
    def __init__(self, params, coarse_tagger, fine_tagger, sent_repre_generator=None):
        self.params = params
        self.coarse_tagger = coarse_tagger
        self.fine_tagger = fine_tagger
        self.lr = params.lr
        self.lr_decay = params.lr_decay
        self.use_label_encoder = params.tr
        self.num_domain = params.num_domain
        self.patience = params.patience
        self.model_saved_path = os.path.join(self.params.dump_path, "best_model.pth")
        self.opti_saved_path = os.path.join(self.params.dump_path, "opti.pth")
        if self.use_label_encoder:
            self.sent_repre_generator = sent_repre_generator
            self.loss_fn_mse = nn.MSELoss()
            model_parameters = [
                {"params": self.coarse_tagger.parameters()},
                {"params": self.fine_tagger.parameters()},
                {"params": self.sent_repre_generator.parameters()}
            ]
        else:
            model_parameters = [
                {"params": self.coarse_tagger.parameters()},
                {"params": self.fine_tagger.parameters()}
            ]
        # Adam optimizer
        self.optimizer = torch.optim.Adam(model_parameters, lr=self.lr)
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.early_stop = params.early_stop
        self.no_improvement_num = 0
        self.trivial_num = 0
        self.best_coarse_f1 = 0
        self.best_f1 = 0

        self.stop_training_flag = False


    def train_step(self, X, lengths, y_bin, y_final, y_dm, templates=None, tem_lengths=None, epoch=None):
        self.coarse_tagger.train()
        self.fine_tagger.train()
        
        if self.use_label_encoder:
            self.sent_repre_generator.train()
        
        bin_preds, lstm_hiddens = self.coarse_tagger(X, y_dm)
        


        loss_bin = self.coarse_tagger.crf_loss(bin_preds, lengths, y_bin)

        self.optimizer.zero_grad()
        loss_bin.backward(retain_graph=True)
        self.optimizer.step()


        y_label_embedding = self.coarse_tagger.get_labelembedding(lstm_hiddens, lengths, y_dm)


        for k in father_keys:

            all_pred_list = []
            all_gold_list = []
            v = father_son_slot[k]
            coarse_B_index = y1_set.index('B-'+k)
            coarse_I_index = y1_set.index('I-'+k)
            pred_fine_list, gold_fine_list = self.fine_tagger(y_dm,  k, lstm_hiddens, y_label_embedding, coarse_B_index=coarse_B_index, coarse_I_index=coarse_I_index,binary_golds=y_bin, final_golds = y_final)
            pred_fine_list = [temp for temp in pred_fine_list if temp is not None]

            all_pred_list.extend(pred_fine_list)
            all_gold_list.extend(gold_fine_list)

        # loss_slotname = torch.tensor(0).cuda()
            for pred_slotname_each_sample, gold_slotname_each_sample in zip(all_pred_list, all_gold_list):
                assert pred_slotname_each_sample.size()[0] == gold_slotname_each_sample.size()[0]
                # loss_slotname = loss_slotname + self.loss_fn(pred_slotname_each_sample, gold_slotname_each_sample.cuda())
                self.optimizer.zero_grad()
                # print(pred_slotname_each_sample)
                # print(gold_slotname_each_sample)
                loss_slotname = self.loss_fn(pred_slotname_each_sample, gold_slotname_each_sample.cuda())
                # print(loss_slotname)
                loss_slotname.backward(retain_graph=True)
                self.optimizer.step()
        # print('-'*20)
        if self.use_label_encoder:
            templates_repre, input_repre = self.sent_repre_generator(templates, tem_lengths, lstm_hiddens, lengths)

            input_repre = input_repre.detach()
            template0_loss = self.loss_fn_mse(templates_repre[:, 0, :], input_repre)
            template1_loss = -1 * self.loss_fn_mse(templates_repre[:, 1, :], input_repre)
            template2_loss = -1 * self.loss_fn_mse(templates_repre[:, 2, :], input_repre)
            input_repre.requires_grad = True

            self.optimizer.zero_grad()
            template0_loss.backward(retain_graph=True)
            template1_loss.backward(retain_graph=True)
            template2_loss.backward(retain_graph=True)
            self.optimizer.step()

            if epoch > 3:
                templates_repre = templates_repre.detach()
                input_loss0 = self.loss_fn_mse(input_repre, templates_repre[:, 0, :])
                input_loss1 = -1 * self.loss_fn_mse(input_repre, templates_repre[:, 1, :])
                input_loss2 = -1 * self.loss_fn_mse(input_repre, templates_repre[:, 2, :])
                templates_repre.requires_grad = True

                self.optimizer.zero_grad()
                input_loss0.backward(retain_graph=True)
                input_loss1.backward(retain_graph=True)
                input_loss2.backward(retain_graph=True)
                self.optimizer.step()


        if self.use_label_encoder:
            return loss_bin.item(), loss_slotname.item(), template0_loss.item(), template1_loss.item()
        else:
            return loss_bin.item(), loss_slotname.item()

    def evaluate(self, dataloader, istestset=False):
        self.coarse_tagger.eval()
        self.fine_tagger.eval()



        binary_preds, binary_golds = [], []
        final_preds, final_golds = [], []

        pbar = tqdm(enumerate(dataloader), total=len(dataloader))

        for i, (X, lengths,y_0,  y_bin, y_final, y_dm) in pbar:
            binary_golds.extend(y_bin)
            final_golds.extend(y_final)

            X, lengths = X.cuda(), lengths.cuda()
            bin_preds_batch, lstm_hidden = self.coarse_tagger(X, y_dm, iseval=True)

            y_label_embedding = self.coarse_tagger.get_labelembedding(lstm_hidden, lengths, y_dm)


            bin_preds_batch = self.coarse_tagger.crf_decode(bin_preds_batch, lengths)
           
            binary_preds.extend(bin_preds_batch)
           
            

            all_pred_list = []
            for k in father_keys:
                coarse_B_index = y1_set.index('B-'+k)
                coarse_I_index = y1_set.index('I-'+k)
                # print(k)
                pred_fine_list = self.fine_tagger(y_dm,  k, lstm_hidden, y_label_embedding, coarse_B_index=coarse_B_index, coarse_I_index=coarse_I_index, binary_preditions=bin_preds_batch)
                # print(pred_fine_list)
                # print('-'*10)
                
                # pred_fine_list = [temp for temp in pred_fine_list if temp is not None]

                all_pred_list.append(pred_fine_list)
                

            # coarse_preds_batch = self.fine_tagger(y_dm, lstm_hiddens, binary_preditions=bin_preds_batch, binary_golds=None, final_golds=None)
            
            final_preds_batch = self.combine_binary_and_slotname_preds(y_dm, bin_preds_batch, all_pred_list, self.fine_tagger.coarse_fine_map)
            final_preds.extend(final_preds_batch)


        binary_preds = np.concatenate(binary_preds, axis=0)
        binary_preds = list(binary_preds)
        binary_golds = np.concatenate(binary_golds, axis=0)
        binary_golds = list(binary_golds)

        # final predictions
        final_preds_temp = []
        for i in final_preds:
            final_preds_temp.extend(i)
        final_preds = final_preds_temp 
        final_golds = np.concatenate(final_golds, axis=0)
        final_golds = list(final_golds)
        bin_lines, final_lines = [], []
        _bin_preds = []
        _bin_golds = []
        _final_preds = []
        _final_golds = []





        for bin_pred, bin_gold, final_pred, final_gold in zip(binary_preds, binary_golds, final_preds, final_golds):

            bin_slot_pred = y1_set[bin_pred]
            bin_slot_gold = y1_set[bin_gold]
            
            final_slot_pred = final_pred
            final_slot_gold = y2_set[final_gold]

            _bin_preds.append(bin_slot_pred)
            _bin_golds.append(bin_slot_gold)
            _final_preds.append(final_slot_pred)
            _final_golds.append(final_slot_gold)

            bin_lines.append("w" + " " + bin_slot_pred + " " + bin_slot_gold)
            final_lines.append("w" + " " + final_slot_pred + " " + final_slot_gold)


        # print(_bin_preds)
        # sset = set()
        ##########将sc中 son 全换成 father， 同时将其他father全为O。 按理说 father,son F1 应该相同
        '''
        for i in range(len(_bin_preds)):
            if _bin_preds[i] not in['O', 'B-special_name','I-special_name','B-common_name','I-common_name']:
                # print(_bin_preds[i])
                # print(_final_preds[i])
                # print(_final_golds[i])
                # print('-'*10)
                _bin_preds[i] = 'O'
                _final_preds[i] = 'O'

            
        # print(sset)
        
        for i in range(len(_final_preds)):
            _final_preds[i] = _final_preds[i].replace("object_name", "special_name").replace("object_type","common_name")

        for i in range(len(_final_preds)):
            if _final_preds[i] != _bin_preds[i]:
                print(_final_preds[i])
                print(_bin_preds[i])
                print('-'*20)
        '''
        ################################
        bin_result = conll2002_measure(bin_lines, True)
        bin_f1 = bin_result["fb1"]
        
        final_result = conll2002_measure(final_lines, True)
        final_f1 = final_result["fb1"]

        # conll.evaluate(_bin_golds, _bin_preds)
        conlleval.evaluate(_bin_golds, _bin_preds,logger)
        logger.info('-'*20)
        # conll.evaluate(_final_golds, _final_preds)
        conlleval.evaluate(_final_golds, _final_preds,logger)

        # print(_bin_preds[:100])
        # print(_bin_golds[:20])
        # print(_final_preds[:100])
        # print(_final_golds[:20])

     













        if istestset == False:  # dev set
            if final_f1 > self.best_f1:
            # if bin_f1 > self.best_coarse_f1 or final_f1 > self.best_fine_f1:
                self.best_f1 = final_f1
                # self.best_coarse_f1 = bin_f1
                # self.best_fine_f1 = final_f1
                self.no_improvement_num = 0
                logger.info("Found better model!!")
                self.save_model()
            else:
                self.no_improvement_num += 1
                logger.info("No better model found (%d/%d)" % (self.no_improvement_num, self.early_stop))
                if self.no_improvement_num == self.early_stop:
                    logger.info("hit patience %d" % self.no_improvement_num)
                    self.trivial_num += 1
                    logger.info('hit # %d trial' % self.trivial_num)
                    if self.trivial_num == self.patience:
                        logger.info('early stop!')
                        exit(0)
                    
                    # lr = [self.optimizer.param_groups[i]['lr'] * self.lr_decay for i in range(len(self.optimizer.param_groups))]
                    lr = self.optimizer.param_groups[0]['lr'] * self.lr_decay
                    
                    logger.info("load previously best model and decay learning rate to %f" % lr)
                    
                    reloaded = torch.load(self.model_saved_path)
                    self.coarse_tagger.load_state_dict(reloaded["coarse_tagger"])
                    self.fine_tagger.load_state_dict(reloaded["fine_tagger"])
                    if self.use_label_encoder:
                        self.sent_repre_generator.load_state_dict(reloaded["sent_repre_generator"])
                        self.sent_repre_generator = self.sent_repre_generator.cuda()

                   
                    self.coarse_tagger = self.coarse_tagger.cuda()
                    self.fine_tagger = self.fine_tagger.cuda()


                    logger.info('restore parameters of the optimizers')

                    self.optimizer.load_state_dict(torch.load(self.opti_saved_path))

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr

                    self.no_improvement_num = 0


        
        return bin_f1, final_f1, False

    def combine_binary_and_slotname_preds(self, dm_id_batch, binary_preds_batch, fine_preds_batch, coarse_fine_map):
        # print('\n')
        # print(binary_preds_batch)
        # print('\n')
        # print(fine_preds_batch)
        # print('\n')
        final_preds = []
        for i in range(len(binary_preds_batch)):
            temp = ['O'] * len(binary_preds_batch[i])
            final_preds.append(temp)

        # print(final_preds)
        # print('\n')
        for fa_id, father in enumerate(father_keys):
            for bsz_id in range(len(final_preds)):
                if fine_preds_batch[fa_id][bsz_id] is None:
                    continue
                else:
                    i = 0
                    cur_entity_id = 0
                    while i < len(final_preds[bsz_id]):
                        cur_id = binary_preds_batch[bsz_id][i]
                        
                        if y1_set[cur_id][0] == 'B' and y1_set[cur_id][2:] == father:
                            # print(fine_preds_batch[fa_id][bsz_id])
                            # print(fine_preds_batch[fa_id][bsz_id].argmax(-1))
                            
                            cur_entity_best = fine_preds_batch[fa_id][bsz_id].argmax(-1)[cur_entity_id]
                            cur_entity_id += 1

                            id_1 = dm_id_batch[bsz_id].item()
                            id_2 = fa_id 
                            id_3 = cur_entity_best.item()
                            id_1 = domain_set[id_1]
                            id_2 = father_keys[id_2]
                            # print(id_1)
                            # print(id_2)
                            # print(id_3)

                            final_preds[bsz_id][i] = 'B-' + coarse_fine_map[id_1][id_2][id_3]
                        elif y1_set[cur_id][0] == 'I' and y1_set[cur_id][2:] == father:
                            if i-1 >= 0 and (final_preds[bsz_id][i-1][0] == 'B' or final_preds[bsz_id][i-1][0] == 'I'):
                                final_preds[bsz_id][i] = 'I-' + final_preds[bsz_id][i-1][2:]

                        
                        i += 1

                # print('\n')
                # print(final_preds)
                # print('\n')
                # print(fa_id)
                # print(bsz_id)
                # print('\n')
        return final_preds
        
    def save_model(self):
        """
        save the best model
        """
        model_saved_path = os.path.join(self.params.dump_path, "best_model.pth")
        if self.use_label_encoder:
            torch.save({
                "coarse_tagger": self.coarse_tagger.state_dict(),
                "fine_tagger": self.fine_tagger.state_dict(),
                "sent_repre_generator": self.sent_repre_generator.state_dict()
            }, model_saved_path)
        else:
            torch.save({
                "coarse_tagger": self.coarse_tagger.state_dict(),
                "fine_tagger": self.fine_tagger.state_dict(),
            }, model_saved_path)
        logger.info("Best model has been saved to %s" % model_saved_path)

        opti_saved_path = os.path.join(self.params.dump_path, "opti.pth")
        torch.save(self.optimizer.state_dict(), opti_saved_path)
        logger.info("Best model opti has been saved to %s" % opti_saved_path)

        self.model_saved_path = model_saved_path
        self.opti_saved_path = opti_saved_path

def get_coarse_labels_for_domains():
    dm_coarse= {}
    for domain, slot_list in domain2slot.items():
        dm_coarse[domain] = {}
        for father, son in father_son_slot.items():
            dm_coarse[domain][father] = []
            for s in son:
                if s in slot_list:
                    dm_coarse[domain][father].append(s)
    
    return dm_coarse