export CUDA_VISIBLE_DEVICES=0
python test.py --exp_name coach_patience --exp_id atp_0 --bidirection --freeze_emb  --tgt_dm AddToPlaylist --model_path='./experiments/coach_patience/atp_0/best_model.pth'