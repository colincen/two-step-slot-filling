export CUDA_VISIBLE_DEVICES=0
python main.py --exp_name coach_no_tr --exp_id rb_0 --bidirection --freeze_emb  --tgt_dm RateBook
python main.py --exp_name coach_no_tr --exp_id pm_0 --bidirection --freeze_emb  --tgt_dm PlayMusic
python main.py --exp_name coach_no_tr --exp_id sc_0 --bidirection --freeze_emb  --tgt_dm SearchCreativeWork
# python main.py --exp_name coach_no_tr --exp_id gw_0 --bidirection --freeze_emb  --tgt_dm GetWeather
# python main.py --exp_name coach_no_tr --exp_id sse_0 --bidirection --freeze_emb  --tgt_dm SearchScreeningEvent