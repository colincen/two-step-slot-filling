from src.dataloader import get_dataloader
from src.datareader import y1_set, father_son_slot, domain2slot
from src.utils import init_experiment
from src.model import *
from src.trainer import SLUTrainer
from config import get_params
from tqdm import tqdm
from preprocess.gen_embeddings_for_slu import *
# gen_slot_embs_based_on_each_domain('/data/sh/glove.6B.300d.txt')
# combine_word_with_char_embs_for_slot("/data/sh/coachdata/snips/emb/slot_embs_based_on_each_domain.dict")
# combine_word_with_char_embs_for_vocab("/data/sh/coachdata/snips/emb/slu_embs.npy")
# add_slot_embs_to_slu_embs("/data/sh/coachdata/snips/emb/slot_word_char_embs_based_on_each_domain.dict", "/data/sh/coachdata/snips/emb/slu_word_char_embs.npy")

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
            

def main(params):

    logger = init_experiment(params, logger_filename=params.logger_filename)

    dataloader_tr, dataloader_val, dataloader_test, vocab = get_dataloader(params.tgt_dm, params.batch_size, params.tr, params.n_samples)

    coarse_slutagger = CoarseSLUTagger(params, vocab)

    coarse_slutagger = coarse_slutagger.cuda()
    dm_coarse = get_coarse_labels_for_domains()

    fine_predictor = FinePredictor(params, dm_coarse)
    fine_predictor = fine_predictor.cuda()
    

    # if params.tr:
    sent_repre_generator = SentRepreGenerator(params, vocab)
    sent_repre_generator = sent_repre_generator.cuda()

    slu_trainer = SLUTrainer(params, coarse_slutagger, fine_predictor, sent_repre_generator=sent_repre_generator)

    for e in range(params.epoch):
        logger.info("============== epoch {} ==============".format(e+1))
        loss_bin_list, loss_slotname_list = [], []
        if params.tr:
            loss_tem0_list, loss_tem1_list = [], []
        pbar = tqdm(enumerate(dataloader_tr), total=len(dataloader_tr))
        if params.tr:
            for i, (X, lengths, y_bin, y_final, y_dm, templates, tem_lengths) in pbar:
                X, lengths, templates, tem_lengths = X.cuda(), lengths.cuda(), templates.cuda(), tem_lengths.cuda()
                loss_bin, loss_slotname, loss_tem0, loss_tem1 = slu_trainer.train_step(X, lengths, y_bin, y_final, y_dm, templates=templates, tem_lengths=tem_lengths, epoch=e)
                loss_bin_list.append(loss_bin)
                loss_slotname_list.append(loss_slotname)
                loss_tem0_list.append(loss_tem0)
                loss_tem1_list.append(loss_tem1)

                pbar.set_description("(Epoch {}) LOSS BIN:{:.4f} LOSS SLOT:{:.4f} LOSS TEM0:{:.4f} LOSS TEM1:{:.4f}".format((e+1), np.mean(loss_bin_list), np.mean(loss_slotname_list), np.mean(loss_tem0_list), np.mean(loss_tem1_list)))
        else:
            for i, (X, lengths, y_bin, y_final, y_dm) in pbar:
                X, lengths = X.cuda(), lengths.cuda()
                loss_bin, loss_slotname = slu_trainer.train_step(X, lengths, y_bin, y_final, y_dm)
                loss_bin_list.append(loss_bin)
                loss_slotname_list.append(loss_slotname)
                pbar.set_description("(Epoch {}) LOSS BIN:{:.4f} LOSS SLOT:{:.4f}".format((e+1), np.mean(loss_bin_list), np.mean(loss_slotname_list)))
        
        if params.tr:
            logger.info("Finish training epoch {}. LOSS BIN:{:.4f} LOSS SLOT:{:.4f} LOSS TEM0:{:.4f} LOSS TEM1:{:.4f}".format((e+1), np.mean(loss_bin_list), np.mean(loss_slotname_list), np.mean(loss_tem0_list), np.mean(loss_tem1_list)))
        else:
            logger.info("Finish training epoch {}. LOSS BIN:{:.4f} LOSS SLOT:{:.4f}".format((e+1), np.mean(loss_bin_list), np.mean(loss_slotname_list)))

        logger.info("============== Evaluate Epoch {} ==============".format(e+1))
        bin_f1, final_f1, stop_training_flag = slu_trainer.evaluate(dataloader_val, istestset=False)
        logger.info("Eval on dev set. Binary Slot-F1: {:.4f}. Final Slot-F1: {:.4f}.".format(bin_f1, final_f1))

        bin_f1, final_f1, stop_training_flag = slu_trainer.evaluate(dataloader_test, istestset=True)
        logger.info("Eval on test set. Binary Slot-F1: {:.4f}. Final Slot-F1: {:.4f}.".format(bin_f1, final_f1))

        if stop_training_flag == True:
            break


    # optimizer = torch.optim.AdamW(coarse_slutagger.parameters(),lr=0.0001)

   

    # pbar = tqdm(enumerate(dataloader_tr), total=len(dataloader_tr))
    # for i, (X, lengths, y_bin, y_final, y_dm, templates, tem_lengths) in pbar:
    #     X, lengths, templates, tem_lengths = X.cuda(), lengths.cuda(), templates.cuda(), tem_lengths.cuda()


    #     bin_preds, lstm_hiddens = coarse_slutagger(X)
    #     loss_bin = coarse_slutagger.crf_loss(bin_preds, lengths, y_bin)
    #     optimizer.zero_grad()
    #     loss_bin.backward(retain_graph=True)
    #     # optimizer.step()
    #     loss_fn = nn.CrossEntropyLoss()
    #     # slot_list = []
    #     # for k, v in domain2slot.items():
    #     #     slot_list.extend(v)
    #     all_pred_list = []
    #     all_gold_list = []
    #     for k,v in father_son_slot.items():
    #         coarse_B_index = y1_set.index('B-'+k)
    #         coarse_I_index = y1_set.index('I-'+k)
    #         pred_fine_list, gold_fine_list = fine_predictor(y_dm,  k, lstm_hiddens, coarse_B_index=coarse_B_index, coarse_I_index=coarse_I_index,binary_golds=y_bin, final_golds = y_final)
    #         pred_fine_list = [temp for temp in pred_fine_list if temp is not None]

    #         all_pred_list.extend(pred_fine_list)
    #         all_gold_list.extend(gold_fine_list)

    #     loss_slotname = torch.tensor(0).cuda()
    #     for pred_slotname_each_sample, gold_slotname_each_sample in zip(all_pred_list, all_gold_list):
    #         assert pred_slotname_each_sample.size()[0] == gold_slotname_each_sample.size()[0]
    #         loss_slotname = loss_slotname + loss_fn(pred_slotname_each_sample, gold_slotname_each_sample.cuda())
    #         # optimizer.zero_grad()
        
    #     loss_slotname.backward(retain_graph=True)
    #     # optimizer.step()


    #     templates_repre, input_repre = sent_repre_generator(templates, tem_lengths, lstm_hiddens, lengths)

    #     input_repre = input_repre.detach()
    #     template0_loss = loss_fn_mse(templates_repre[:, 0, :], input_repre)
    #     template1_loss = -1 * loss_fn_mse(templates_repre[:, 1, :], input_repre)
    #     template2_loss = -1 * loss_fn_mse(templates_repre[:, 2, :], input_repre)
    #     input_repre.requires_grad = True

    #     # self.optimizer.zero_grad()
    #     template0_loss.backward(retain_graph=True)
    #     template1_loss.backward(retain_graph=True)
    #     template2_loss.backward(retain_graph=True)
    #     optimizer.step()


    #     print('ok')

        # pbar.set_description("LOSS BIN:{:.4f} LOSS SLOT:{:.4f} ".format(loss_bin.detach().cpu().numpy(), loss_slotname.detach().cpu().numpy()))


        # print(bin_preds.size())
        # print(lstm_hiddens.size())
            # loss_bin, loss_slotname, loss_tem0, loss_tem1 = slu_trainer.train_step(X, 
            # lengths, y_bin, y_final, y_dm, templates=templates, tem_lengths=tem_lengths, epoch=e)



if __name__ == "__main__":
    params = get_params()
    main(params)