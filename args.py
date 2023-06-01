import argparse


def _add_common_args(arg_parser):
    arg_parser.add_argument('--config', type=str)

    arg_parser.add_argument("--local_rank", type=int, default=-1, help="")
    arg_parser.add_argument("--world_size", type=int, default=-1, help="")

    # Input
    arg_parser.add_argument('--types_path', type=str, help="Path to type specifications")

    # Preprocessing
    arg_parser.add_argument('--tokenizer_path', type=str, help="Path to tokenizer")
    arg_parser.add_argument('--lowercase', action='store_true', default=False,
                            help="If true, input is lowercased during preprocessing")
    arg_parser.add_argument('--sampling_processes', type=int, default=4,
                            help="Number of sampling processes. 0 = no multiprocessing for sampling")

    # Logging
    arg_parser.add_argument('--label', type=str, help="Label of run. Used as the directory name of logs/models")
    arg_parser.add_argument('--log_path', type=str, help="Path do directory where training/evaluation logs are stored")
    arg_parser.add_argument('--store_predictions', action='store_true', default=False,
                            help="If true, store predictions on disc (in log directory)")
    arg_parser.add_argument('--store_examples', action='store_true', default=False,
                            help="If true, store evaluation examples on disc (in log directory)")
    arg_parser.add_argument('--example_count', type=int, default=None,
                            help="Count of evaluation example to store (if store_examples == True)")
    arg_parser.add_argument('--debug', action='store_true', default=False, help="Debugging mode on/off")
    arg_parser.add_argument('--save_code', action='store_true', default=False)

    # Model / Training / Evaluation
    
    arg_parser.add_argument('--lstm_layers', type=int, default=0, help="")
    arg_parser.add_argument('--span_attn_layers', type=int, default=0, help="")
    arg_parser.add_argument('--wo_self_attn', action='store_true', default=False)
    arg_parser.add_argument('--wo_cross_attn', action='store_true', default=False)
    
    
    arg_parser.add_argument('--split_epoch', type=int, default=0, help="")
    arg_parser.add_argument('--stage_one_lr_scale', type=float, default=1.0, help="")
    

    arg_parser.add_argument('--prop_drop', type=float, default=0.1, help="Probability of dropout used in diffusionner")
    arg_parser.add_argument('--soi_pooling', type=str, default="lrconcat")
    arg_parser.add_argument('--pos_type', type=str, default="sine")
    
    
    arg_parser.add_argument('--num_proposals', type=int, default=100, help="")
    arg_parser.add_argument('--sampling_timesteps', type=int, default=10, help="")
    arg_parser.add_argument('--beta_schedule', type=str, default="cosine", help="")
    arg_parser.add_argument('--timesteps', type=int, default=100, help="")
    arg_parser.add_argument('--step_embed_type', type=str, default="add", help="")
    arg_parser.add_argument('--sample_dist_type', type=str, default="normal", help="")

    
    arg_parser.add_argument('--scale', type=float, default=3.0, help="")
    arg_parser.add_argument('--extand_noise_spans', type=str, default="repeat", help="")
    arg_parser.add_argument('--span_renewal', action='store_true', default=False)
    arg_parser.add_argument('--step_ensemble', action='store_true', default=False)
    
    
    arg_parser.add_argument('--device_id', type=int, default=-1, help="gpu device id")
    arg_parser.add_argument('--model_path', type=str, help="Path to directory that contains model checkpoints")
    arg_parser.add_argument('--model_type', type=str, default="diffusionner", help="Type of model")
    arg_parser.add_argument('--cpu', action='store_true', default=False,
                            help="If true, train/evaluate on CPU even if a CUDA device is available")
    arg_parser.add_argument('--eval_batch_size', type=int, default=1, help="Evaluation batch size")
    arg_parser.add_argument('--pool_type', type=str, default = "max")

    arg_parser.add_argument('--no_overlapping', action='store_true', default=False)
    arg_parser.add_argument('--no_partial_overlapping', action='store_true', default=False)
    arg_parser.add_argument('--no_duplicate', action='store_true', default=False)

    arg_parser.add_argument('--boundary_threshold', type=float, default=0.0, help="")
    arg_parser.add_argument('--cls_threshold', type=float, default=0.0, help="")
    arg_parser.add_argument('--entity_threshold', type=float, default=0.0, help="")

    
    
    # Misc
    arg_parser.add_argument('--seed', type=int, default=-1, help="Seed")
    arg_parser.add_argument('--cache_path', type=str, default=None,
                            help="Path to cache transformer models (for HuggingFace transformers library)")



def train_argparser():
    arg_parser = argparse.ArgumentParser()

    # Input
    arg_parser.add_argument('--train_path', type=str, help="Path to train dataset")
    arg_parser.add_argument('--valid_path', type=str, help="Path to validation dataset")

    # Logging
    arg_parser.add_argument('--save_path', type=str, help="Path to directory where model checkpoints are stored")
    arg_parser.add_argument('--save_path_include_iteration', action='store_true', default=False)

    
    arg_parser.add_argument('--init_eval', action='store_true', default=False,
                            help="If true, evaluate validation set before training")
    arg_parser.add_argument('--save_optimizer', action='store_true', default=False,
                            help="Save optimizer alongside model")
    arg_parser.add_argument('--train_log_iter', type=int, default=1, help="Log training process every x iterations")
    arg_parser.add_argument('--final_eval', action='store_true', default=False,
                            help="Evaluate the model only after training, not at every epoch")

    # Model / Training
    arg_parser.add_argument('--train_batch_size', type=int, default=2, help="Training batch size")
    arg_parser.add_argument('--epochs', type=int, default=20, help="Number of epochs")
    arg_parser.add_argument('--lr', type=float, default=5e-5, help="Learning rate")
    arg_parser.add_argument('--lr_warmup', type=float, default=0.1,
                            help="Proportion of total train iterations to warmup in linear increase/decrease schedule")
    arg_parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay to apply")
    arg_parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Maximum gradient norm")

    
    arg_parser.add_argument('--match_solver', type=str, help="", default="hungarian")
    arg_parser.add_argument('--type_loss', type=str, help="", default="celoss")
    
    
    arg_parser.add_argument('--nil_weight', type=float, default=-1)
    arg_parser.add_argument('--match_boundary_weight', type=float, default=10.0)
    arg_parser.add_argument('--match_class_weight', type=float, default=2.0)
    arg_parser.add_argument('--loss_boundary_weight', type=float, default=2.0)
    arg_parser.add_argument('--loss_class_weight', type=float, default=2.0)
    arg_parser.add_argument('--match_boundary_type',  type=str, default="f1")

    
    arg_parser.add_argument('--repeat_gt_entities', type=int, default=-1, help="")
    arg_parser.add_argument('--eval_every_epochs', type=int, default=1, help="")
    arg_parser.add_argument('--eval_test',action='store_true', default=False)
    

    _add_common_args(arg_parser)

    return arg_parser


def eval_argparser():
    arg_parser = argparse.ArgumentParser()

    # Input
    arg_parser.add_argument('--dataset_path', type=str, help="Path to dataset")

    _add_common_args(arg_parser)

    return arg_parser
