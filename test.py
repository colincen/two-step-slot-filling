from src.dataloader import get_dataloader
from src.datareader import y1_set, father_son_slot, domain2slot
from src.utils import init_experiment
from src.model import *
from src.trainer import SLUTrainer
from config import get_params
from tqdm import tqdm
from preprocess.gen_embeddings_for_slu import *
import os

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
def test_coach(params):
    
    logger = init_experiment(params, logger_filename='test')
    # get dataloader
    dataloader_tr, dataloader_val, dataloader_test, vocab = get_dataloader(params.tgt_dm, params.batch_size, params.tr, params.n_samples)
    # _, _, dataloader_test, _ = get_dataloader(params.tgt_dm, params.batch_size, params.tr, params.n_samples)

    print(params.model_path)
    model_path = params.model_path
    opti_path = './experiments/coach_patience/atp_0/opti.pth'
    
    assert os.path.isfile(model_path)
    
    reloaded = torch.load(model_path)
    coarse_slutagger = CoarseSLUTagger(params, vocab)

    coarse_slutagger = coarse_slutagger.cuda()
    dm_coarse = get_coarse_labels_for_domains()

    fine_tagger = FinePredictor(params, dm_coarse)
    fine_tagger = fine_tagger.cuda()

    coarse_slutagger.load_state_dict(reloaded["coarse_tagger"])

    fine_tagger.load_state_dict(reloaded["fine_tagger"])
    coarse_tagger = coarse_slutagger.cuda()
    # fine_tagger.cuda()


    # model_parameters = [
    #             {"params": coarse_tagger.parameters()},
    #             {"params": fine_tagger.parameters()}
    #         ]

    # optimizer = torch.optim.Adam(model_parameters, lr=self.lr)

    # optimizer.load_state_dict(torch.load(opti_path))

    slu_trainer = SLUTrainer(params, coarse_tagger, fine_tagger)
    slu_trainer.optimizer.load_state_dict(torch.load(opti_path))

    for e in range(params.epoch):
        logger.info("============== epoch {} ==============".format(e+1))
        loss_bin_list, loss_slotname_list = [], []
        if params.tr:
            loss_tem0_list, loss_tem1_list = [], []
        pbar = tqdm(enumerate(dataloader_tr), total=len(dataloader_tr))
        # record = int(len(dataloader_tr) / 4)
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
                if i == 2: 
                    break
                # if i %record == 0 and i > 0:
                #     logger.info("============== Evaluate Epoch {} {}==============".format(e+1, i))
                #     bin_f1, final_f1, stop_training_flag = slu_trainer.evaluate(dataloader_val, istestset=False)
                #     logger.info("Eval on dev set. Binary Slot-F1: {:.4f}. Final Slot-F1: {:.4f}.".format(bin_f1, final_f1))

                #     bin_f1, final_f1, stop_training_flag = slu_trainer.evaluate(dataloader_test, istestset=True)
                #     logger.info("Eval on test set. Binary Slot-F1: {:.4f}. Final Slot-F1: {:.4f}.".format(bin_f1, final_f1))                    
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


    # _, f1_score, _ = slu_trainer.evaluate(dataloader_test, istestset=True)
    # print("Eval on test set. Final Slot F1 Score: {:.4f}.".format(f1_score))



if __name__ == "__main__":
    params = get_params()
    test_coach(params)