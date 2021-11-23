echo "Generating channel edits..."
python main.py --cp_rank 0 --tucker_ranks "4,4,4,512" --model_name pggan_celebahq1024 --penalty_lam 0.001 --resume_iters 1000 --mode edit_modewise --n_to_edit 10 --attribute_to_edit blonde

echo "Generating spatial edits..."
python main.py --cp_rank 0 --tucker_ranks "4,4,4,512" --model_name pggan_celebahq1024 --penalty_lam 0.001 --resume_iters 1000 --mode edit_modewise --n_to_edit 10 --attribute_to_edit yaw
python main.py --cp_rank 0 --tucker_ranks "4,4,4,512" --model_name pggan_celebahq1024 --penalty_lam 0.001 --resume_iters 1000 --mode edit_modewise --n_to_edit 10 --attribute_to_edit pitch