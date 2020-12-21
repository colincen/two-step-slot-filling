export CUDA_VISIBLE_DEVICES=0

python main.py --exp_name coach_encode_slotname  --exp_id atp_0 --bidirection --freeze_emb  --tgt_dm AddToPlaylist 

# python main.py --exp_name coach_tr --tr --exp_id atp_0 --bidirection --freeze_emb  --tgt_dm AddToPlaylist --emb_file  /home/sh/data/coachdata/snips/emb/slu_word_char_embs_with_slotembs.npy
# python main.py --exp_name coach_no_tr_coarse_loss --exp_id br_0 --bidirection --freeze_emb  --tgt_dm BookRestaurant
# python main.py --exp_name coach_no_tr --exp_id rb_0 --bidirection --freeze_emb  --tgt_dm RateBook
# python main.py --exp_name coach_no_tr --exp_id pm_0 --bidirection --freeze_emb  --tgt_dm PlayMusic
# python main.py --exp_name coach_no_tr --exp_id sc_0 --bidirection --freeze_emb  --tgt_dm SearchCreativeWork
# python main.py --exp_name coach_no_tr --exp_id gw_0 --bidirection --freeze_emb  --tgt_dm GetWeather
# python main.py --exp_name coach_no_tr --exp_id sse_0 --bidirection --freeze_emb  --tgt_dm SearchScreeningEvent

