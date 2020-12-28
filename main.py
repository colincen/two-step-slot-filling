from src.dataloader import get_dataloader
from src.datareader import y1_set, father_son_slot, domain2slot
from src.utils import init_experiment
from src.model import *
from src.trainer import SLUTrainer
from config import get_params
from tqdm import tqdm
from preprocess.gen_embeddings_for_slu import *
# gen_slot_embs_based_on_each_domain('/home/sh/data/glove.6B.300d.txt')
# combine_word_with_char_embs_for_slot("/home/sh/data/coachdata/snips/emb/slot_embs_based_on_each_domain.dict")
# combine_word_with_char_embs_for_vocab("/home/sh/data/coachdata/snips/emb/slu_embs.npy")
# add_slot_embs_to_slu_embs("/home/sh/data/coachdata/snips/emb/slot_word_char_embs_based_on_each_domain.dict", "/home/sh/data/coachdata/snips/emb/slu_word_char_embs.npy")

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

    dm_coarse = get_coarse_labels_for_domains()
    
    coarse_slutagger = CoarseSLUTagger(params, vocab, dm_coarse)

    coarse_slutagger = coarse_slutagger.cuda()
    

    fine_predictor = FinePredictor(params, dm_coarse)
    fine_predictor = fine_predictor.cuda()
    

    # if params.tr:
    sent_repre_generator = SentRepreGenerator(params, vocab)
    sent_repre_generator = sent_repre_generator.cuda()

    slu_trainer = SLUTrainer(params,vocab, coarse_slutagger, fine_predictor, sent_repre_generator=sent_repre_generator)

    for e in range(params.epoch):
        loss_c_list = []
        pbar = tqdm(enumerate(dataloader_tr), total=len(dataloader_tr))
        logger.info("============== epoch {} ==============".format(e+1))

        loss_bin_list, loss_slotname_list = [], []
        if params.tr:
            loss_tem0_list, loss_tem1_list = [], []
        
        # record = int(len(dataloader_tr) / 4)
        if params.tr:
            for i, (X, lengths, y_0, y_bin, y_final, y_dm, templates, tem_lengths) in pbar:
                X, lengths, templates, tem_lengths = X.cuda(), lengths.cuda(), templates.cuda(), tem_lengths.cuda()
                loss_bin, loss_slotname, loss_tem0, loss_tem1 = slu_trainer.train_step(X, lengths, y_bin, y_final, y_dm, templates=templates, tem_lengths=tem_lengths, epoch=e)
                loss_bin_list.append(loss_bin)
                loss_slotname_list.append(loss_slotname)
                loss_tem0_list.append(loss_tem0)
                loss_tem1_list.append(loss_tem1)

                pbar.set_description("(Epoch {}) LOSS BIN:{:.4f} LOSS SLOT:{:.4f} LOSS TEM0:{:.4f} LOSS TEM1:{:.4f}".format((e+1), np.mean(loss_bin_list), np.mean(loss_slotname_list), np.mean(loss_tem0_list), np.mean(loss_tem1_list)))
        else:
            for i, (X, lengths, y_0, y_bin, y_final, y_dm) in pbar:       
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



if __name__ == "__main__":
    params = get_params()
    main(params)