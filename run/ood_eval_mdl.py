import math
import pandas as pd
import sys
from argparse import Namespace
from typing import Tuple
from unittest import result
import time

import logging
from tqdm import tqdm

from utils.status import ProgressBar

from run.evaluation import *

from accelerate import Accelerator

try:
    import wandb
except ImportError:
    wandb = None

def ood_eval(model, dataset, accelerator, args: Namespace, ood_ori_dataset):
    """
    The training process, including evaluations and loggers.
    
    Args:
        model: the model to be trained
        dataset: the dataset at hand
        args: the arguments of the current execution
    """

    if accelerator.is_local_main_process:
        print(args)
        save_folder = f'checkpoints/{args.dataset}/{args.model}/{args.model}/{args.log_path}'
        create_if_not_exists(save_folder)
        logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s', level=logging.INFO, filename=save_folder+'/log.txt')
        if not args.nowand:
            assert wandb is not None, "Wandb not installed, please install it or run without wandb"
            if not args.wandb_name:
                wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
            else:
                wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_name, config=vars(args))
            args.wandb_url = wandb.run.get_url()
        print(file=sys.stderr)

    test_loader = dataset.test_dataloader
    ood_ori_test_loader = ood_ori_dataset.test_dataloader
    # i am going to change all model./model.net to model.model
    model.tokenizer = dataset.tokenizer
    model.model.target_ids = dataset.target_ids.squeeze(-1)
    model.model.model.target_ids = dataset.target_ids.squeeze(-1)


    model.model, test_loader, ood_ori_test_loader = accelerator.prepare(
        model.model,  test_loader, ood_ori_test_loader
    )
    start_time = time.time()

    model.model.eval() # it was model.module.eval() before....
    print('args.model_wrapper')
    print(args.modelwrapper)
    print('args.model_wrapper.startwith(lightblob)')
    print(args.modelwrapper.startswith('lightblob'))
    if args.modelwrapper.startswith('blob') or args.modelwrapper.startswith('lightblob'):
        start_time = time.time()
        ######  I am changing the following, previously it was without for loop, ....
        for ns in [0, 5, 10]:
            # model.model.eval_n_samples = args.eval_n_samples_final
            args.bayes_eval_n_samples = ns
            # test_acc, test_ece, test_nll, test_brier, ood_acc, ood_auc = evaluate_ood_detection(model, dataset, ood_ori_dataset, test_loader, ood_ori_test_loader, accelerator, args)
            test_acc, test_ece, test_nll, test_brier, ood_acc, ood_ece, ood_nll = evaluate_ood_detection(model, dataset, ood_ori_dataset, test_loader, ood_ori_test_loader, accelerator, args, nsamp =ns) # adding the number of samples...
            if accelerator.is_local_main_process:
                wandb.log({'test_acc'+str(ns): test_acc, 'test_ece'+str(ns): test_ece, 'test_nll'+str(ns): test_nll, 'test_brier'+str(ns):test_brier, "ood_acc"+str(ns): ood_acc, 'ood_ece'+str(ns): ood_ece, 'ood_nll'+str(ns): ood_nll})#"ood_auc": ood_auc})
                # logging.info(f'test_acc: {test_acc}, test_ece: {test_ece}, test_nll: {test_nll}, test_brier: {test_brier}, ood_acc: {ood_acc}, ood_auc: {ood_auc}')
                logging.info(f'test_acc{ns}: {test_acc}, test_ece{ns}: {test_ece}, test_nll{ns}: {test_nll}, test_brier{ns}: {test_brier}, ood_acc{ns}: {ood_acc}, ood_ece{ns}: {ood_ece}, ood_nll{ns}: {ood_nll}') #ood_auc: {ood_auc}')
                end_time = time.time()
                time_seconds = end_time - start_time
                time_minutes = time_seconds / 60
                print(time_minutes)
        
    else:
        for ns in [0]:#, 5, 10]: # there was no for in else function....
            # test_acc, test_ece, test_nll, test_brier, ood_acc, ood_auc = evaluate_ood_detection(model, dataset, ood_ori_dataset, test_loader, ood_ori_test_loader, accelerator, args)
            test_acc, test_ece, test_nll, test_brier, ood_acc, ood_ece, ood_nll = evaluate_ood_detection(model, dataset, ood_ori_dataset, test_loader, ood_ori_test_loader, accelerator, args, nsamp =ns) # adding the number of samples...
            if accelerator.is_local_main_process:
                # wandb.log({'test_acc': test_acc, 'test_ece': test_ece, 'test_nll': test_nll, 'test_brier':test_brier, "ood_acc": ood_acc, "ood_auc": ood_auc})
                # logging.info(f'test_acc: {test_acc}, test_ece: {test_ece}, test_nll: {test_nll}, test_brier: {test_brier}, ood_acc: {ood_acc}, ood_auc: {ood_auc}')
                wandb.log({'test_acc'+str(ns): test_acc, 'test_ece'+str(ns): test_ece, 'test_nll'+str(ns): test_nll, 'test_brier'+str(ns):test_brier, "ood_acc"+str(ns): ood_acc, 'ood_ece'+str(ns): ood_ece, 'ood_nll'+str(ns): ood_nll})#"ood_auc": ood_auc})
                logging.info(f'test_acc{ns}: {test_acc}, test_ece{ns}: {test_ece}, test_nll{ns}: {test_nll}, test_brier{ns}: {test_brier}, ood_acc{ns}: {ood_acc}, ood_ece{ns}: {ood_ece}, ood_nll{ns}: {ood_nll}') #ood_auc: {ood_auc}')

    # OOD should be done in the same way as the in-distribution dataset using checkpoint trained by the in-distribution dataset.

    # checkpointing the backbone model.
    if args.checkpoint: # by default the checkpoints folder is checkpoints
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            save_folder = f'checkpoints/{args.dataset}/{args.model}/{args.model}/{args.checkpoint_dic_name}'
            create_if_not_exists(save_folder)
            accelerator.unwrap_model(model.net).model.save_pretrained(save_folder, save_function=accelerator.save)

    if not args.nowand:
        if accelerator.is_local_main_process:
            wandb.finish()

