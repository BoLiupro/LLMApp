# config.py
from argparse import ArgumentParser

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True
    if v.lower() in ("no", "false", "f", "0", "n"):
        return False
    raise ValueError("Boolean argument expected.")

def str_list(v):
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    # 支持逗号/空格分隔
    parts = []
    for token in str(v).replace(",", " ").split():
        if token:
            parts.append(token)
    return parts

def make_args():
    parser = ArgumentParser(description="Selector + Predictor unified config")

    # ======================= General =======================
    parser.add_argument('--csv_file', type=str, nargs='+',
                        default=["data/processed_data/nanchang/app_usage_records"],
                        help="path(s) to csv dataset: dir / file / glob pattern; can pass multiple")
    parser.add_argument('--device', type=str, default="cuda", help="cuda or cpu")
    parser.add_argument('--seed', type=int, default=42, help="random seed")

    # ======================= Dataloader =======================
    parser.add_argument('--selector_batch_size', type=int, default=128, help="batch size")
    parser.add_argument('--predictor_batch_size', type=int, default=32, help="batch size")
    parser.add_argument('--num_workers', type=int, default=4, help="DataLoader workers")
    parser.add_argument('--pin_memory', type=str2bool, default=True, help="pin memory for CUDA")
    parser.add_argument('--val_ratio', type=float, default=0.05, help="validation split ratio")
    parser.add_argument('--keep_in_memory', type=str2bool, default=True,
                        help="True: concat all CSVs to memory (faster). False: stream by file (lower RAM).")
    parser.add_argument('--pad_id', type=int, default=0, help="padding id for discrete fields")
    parser.add_argument('--stride', type=int, default=None,
                        help="sliding window step; None -> use seq_len window (no overlap by default)")

    # ======================= Sequence lengths =======================
    # Predictor: 总窗口 = long_seq_len + short_seq_len + 1（最后1位为label）
    # short 为倒数 short_seq_len（不含 label），long 与 short 允许重叠
    parser.add_argument('--long_seq_len', type=int, default=64, help="user long-term length")
    parser.add_argument('--short_seq_len', type=int, default=12, help="user short-term length")

    # ======================= Optim / Schedules =======================
    parser.add_argument('--selector_epochs', type=int, default=10, help="selector default epochs")
    parser.add_argument('--predictor_epochs', type=int, default=3, help="predictor default epochs")
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.0, help="AdamW weight decay")
    parser.add_argument('--log_every', type=int, default=10, help="print train losses every N steps")

    # ======================= Vocab / Sizes =======================
    parser.add_argument('--app_vocab_size', type=int, default=640, help="number of apps") # sh305
    parser.add_argument('--location_vocab_size', type=int, default=63200, help="number of locations")# sh12576
    parser.add_argument('--num_app_cate', type=int, default=23, help="number of app categories") # sh20 nc23
    parser.add_argument('--num_traffic_bin', type=int, default=10, help="number of traffic bins")

    # ======================= Selector / Imputer =======================
    parser.add_argument('--emb_dim', type=int, default=128, help="selector embedding dim")
    parser.add_argument('--hidden_dim', type=int, default=256, help="selector hidden dim")
    parser.add_argument('--n_layers', type=int, default=4, help="imputer layers")
    parser.add_argument('--key_pattern_len', type=int, default=10, help="selector output length")
    parser.add_argument('--mlm_prob', type=float, default=0.5, help="masking prob for imputer")
    parser.add_argument('--mask_on', type=str, default="masked_only",
                        choices=["masked_only", "unmasked_only"],
                        help="where to compute CE loss relative to selector mask")

    # ======================= Resources for prompts =======================
    parser.add_argument('--app_category_csv', type=str, default='data/processed_data/nanchang/category.csv',
                        help="CSV with columns: app_type_id,en,cn")
    parser.add_argument('--location_csv', type=str, default='data/processed_data/nanchang/location.csv',
                        help="CSV with POI counts per base_id (poi_*)")
    parser.add_argument('--traffic_bin_labels', type=str_list,
                        default=["ultra-low","very-low","low","lower-mid","mid","upper-mid","high","very-high","extreme","burst"],
                        help="comma/space separated labels to override 10 traffic bins")

    # ======================= Predictor (shared LLM + LoRA) =======================
    parser.add_argument('--model_path', type=str, default='/datadisk/gemma-2-2b',
                        help="shared LLM path/name for both encoding and classification")
    parser.add_argument('--freeze_backbone', type=str2bool, default=True,
                        help="freeze base LLM weights; train LoRA + projector/heads only")
    parser.add_argument('--lora_r', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--lora_target_modules', type=str_list, default=None,
                        help="e.g. q_proj,k_proj,v_proj,o_proj; None -> sensible defaults")
    parser.add_argument('--tokens_per_elem', type=int, default=2, help="soft-prefix tokens per element embedding")
    parser.add_argument('--max_elem_text_len', type=int, default=96, help="tokenizer max len for element prompts")
    parser.add_argument('--inst_max_len', type=int, default=64, help="tokenizer max len for instruction prompts")

    # ======================= Predictor training behavior =======================
    parser.add_argument('--train', type=str2bool, default=False, help="whether to train the model")
    parser.add_argument('--ablation', type=str2bool, default=False, help="whether to use ablation version")
    parser.add_argument('--use_nextcate_in_phaseB', type=str2bool, default=True,
                        help="whether to use next_cate prediction in Phase-B of PredictorTrainer")
    parser.add_argument('--use_hlong_in_phaseB', type=str2bool, default=True,
                        help="whether to use hlong (long seq) in Phase-B of PredictorTrainer")
    parser.add_argument('--selector_topk', type=int, default=5, help="#key indices from selector (on long)")
    parser.add_argument('--cate_aux_lambda', type=float, default=2.0,
                        help="loss weight for auxiliary next_cate prediction (Phase-A)")
    parser.add_argument('--traf_aux_lambda', type=float, default=1.0,
                        help="loss weight for auxiliary traffic prediction")  # +++
    parser.add_argument('--eval_topk', type=int, nargs='+', default=[1,5,10],
                        help="K list for Acc@K/MRR@K/NDCG@K on app_id")

    # ======================= Checkpoints =======================
    parser.add_argument('--selector_save_path', type=str, default='checkpt/nanchang/bestSelector.pt',
                        help="where selector checkpoint is saved")
    parser.add_argument('--predictor_save_path', type=str, default='checkpt/nanchang/bestPredictor.pt',
                        help="where predictor checkpoint is saved")
    parser.add_argument('--continue_the_best', type=str, default='',
                        help="continue training from the best checkpoint path")

    # ======================= Misc =======================
    parser.add_argument('--ignore_unknown_args', action='store_true',
                        help='Ignore unknown command line arguments')

    args, unknown = parser.parse_known_args()

    if len(unknown) != 0 and not args.ignore_unknown_args:
        print("Some unrecognised arguments {}".format(unknown))
        raise SystemExit

    return args
