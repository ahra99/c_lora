import numpy as np
import torch

from torch.nn import functional as F
from torchmetrics import Accuracy, CalibrationError
from utils import create_if_not_exists
import os

import pickle


def append_dictionary(dic1, dic2):
    """
    Extend dictionary dic1 with dic2.
    """
    for key in dic2.keys():
        if key in dic1.keys():
            dic1[key].append(dic2[key])
        else:
            dic1[key] = [dic2[key]]

def accuracy_topk(output, target, k=1):
    """Computes the topk accuracy"""
    batch_size = target.size(0)

    _, pred = torch.topk(output, k=k, dim=1, largest=True, sorted=True)

    res_total = 0
    for curr_k in range(k):
      curr_ind = pred[:,curr_k]
      num_eq = torch.eq(curr_ind, target).sum()
      acc = num_eq/len(output)
      res_total += acc
    return res_total*100

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def evaluate_all(model, dataloader, accelerator, args, sample=True, num_classes=None):
    """
    Evaluates the **acc, ece, nll** of the model for given dataset.

    Args:
        model: the model to be evaluated
        dataloader: the dataset to evaluate the model on
        kwargs: optional arguments
    Returns:
        acc: accuracy of the model evaluated on the dataloader
    """
    status = model.net.training
    model.net.eval()

    nlls = AverageMeter()
    if num_classes is None:
        num_classes = args.outdim
    metric_kwargs = {"task": "multiclass", "num_classes": num_classes}
    acc_metric = Accuracy(**metric_kwargs).to(accelerator.device)
    ece_metric = CalibrationError(**metric_kwargs, n_bins = args.num_bins).to(accelerator.device)
    briers = AverageMeter()

    samples_seen = 0
    for step, batch in enumerate(dataloader):
        with torch.no_grad() and torch.inference_mode():
            if args.dataset_type == 'mcdataset':
                _, labels, _ = batch
                logits = model(batch, sample = not args.bayes_inference_notsample).detach()
            else:
                logits = model(batch, sample = not args.bayes_inference_notsample).detach()
                labels = batch["labels"]
            logits, labels = accelerator.gather([logits, labels])
            if accelerator.num_processes > 1:
                if step == len(dataloader) - 1:
                    labels = labels[: len(dataloader.dataset) - samples_seen]
                    logits = logits[: len(dataloader.dataset) - samples_seen]
                else:
                    samples_seen += labels.shape[0]
            # loss_func = 
            # loss_func = torch.nn.CrossEntropyLoss(reduction="mean")
            # nll = loss_func(logits, labels)
            # nlls.update(nll)

            if (not args.bayes_inference_notsample and (args.modelwrapper.startswith('blob') or args.modelwrapper.startswith('light')) ) or args.model.startswith('deepensemble') or args.model.startswith('mcdropout'):
                probs = torch.softmax(logits, dim=-1).mean(dim=1)
                std = torch.softmax(logits, dim=-1).std(dim=1).mean()
            else:
                probs = torch.softmax(logits, dim=-1)
                std = 0

            acc_metric(probs, labels)
            ece_metric(probs, labels)
            loss_func = torch.nn.NLLLoss(reduction="mean")
            nll = loss_func(torch.log(probs), labels)
            nlls.update(nll)

            brier = (probs - F.one_hot(labels, num_classes=logits.size(-1))).pow(2).sum(dim=-1).mean()
            briers.update(brier)
            
    acc = acc_metric.compute().item()
    ece = ece_metric.compute().item()
    nll = nlls.avg
    brier = briers.avg
    model.net.train(status)
        
    return acc, ece, nll, std, brier

def logit_entropy(probs):
    return (-torch.sum(probs * torch.log(probs), dim=1)).cpu().numpy()

def max_softmax(probs):
    return (1 - probs.max(dim=1)[0]).cpu().numpy()

def logit_std(probs):
    return (probs.std(dim=1)).cpu().numpy()

def evaluate_ood_detection(model, dataset, ood_ori_dataset, dataloader, ood_ori_dataloader, accelerator, args, nsamp = 1, sample=True, num_classes=None):
    """
    Evaluates the **acc, ece, nll** of the model for given dataset.

    Args:
        model: the model to be evaluated
        dataloader: the dataset to evaluate the model on
        kwargs: optional arguments
    Returns:
        acc: accuracy of the model evaluated on the dataloader
    """
    # i am changing all model.net/model. to model.model ..............
    print('model.model.training')
    status = model.model.training
    print(status)
    model.model.eval()
    loss = F.nll_loss

    model.tokenizer = ood_ori_dataset.tokenizer
    # model.net.module.target_ids = ood_ori_dataset.target_ids.squeeze(-1) # it was model.net.module. ....
    model.model.target_ids = ood_ori_dataset.target_ids.squeeze(-1) # it was model.net.module.model. ....

    if num_classes is None:
        # num_classes = args.outdim
        num_classes = args.ood_ori_outdim

    nll_ood = AverageMeter()
    metric_kwargs = {"task": "multiclass", "num_classes": num_classes}
    acc_ood = Accuracy(**metric_kwargs).to(accelerator.device)
    ece_ood = CalibrationError(**metric_kwargs, n_bins = args.num_bins).to(accelerator.device)
    
    # print(' bayes-eval-n-samples ---> evaluation . py ')
    # print(args.bayes_eval_n_samples)
    # print('ns')
    ns = nsamp #args.bayes_eval_n_samples
    samples_seen = 0
    id_prob_list = np.array([])
    for step, batch in enumerate(ood_ori_dataloader):
        with torch.no_grad() and torch.inference_mode():
            
            # I am going to change the following part according to the evaluate function in lightblob wrapper
            
            # if args.dataset_type == 'mcdataset':
            #     _, labels, _ = batch
            #     logits = model.model.forward_logits(batch, sample = not args.bayes_inference_notsample).detach() # it was model only
            # else:
            #     logits = model.model.forward_logits(batch, sample = not args.bayes_inference_notsample).detach() # it was model only
            #     labels = batch["labels"]
            # logits, labels = accelerator.gather([logits, labels])
            # if accelerator.num_processes > 1:
            #     if step == len(ood_ori_dataloader) - 1:
            #         labels = labels[: len(ood_ori_dataloader.dataset) - samples_seen]
            #         logits = logits[: len(ood_ori_dataloader.dataset) - samples_seen]
            #     else:
            #         samples_seen += labels.shape[0]
            
            #### the following 4 lines were already commented out.......
            # loss_func = 
            # loss_func = torch.nn.CrossEntropyLoss(reduction="mean")
            # nll = loss_func(logits, labels)
            # nlls.update(nll)

            if ns ==0:
                # print('inside if with ns'+str(ns))
                logits = model.model.forward_logits(batch, sample=False, n_samples = ns).detach()
                # print('logits.shape with ns == 0')
                # print(logits.shape)
            if ns!=0:
                logits = model.model.forward_logits(batch, sample = not args.bayes_inference_notsample, n_samples = ns).detach()
                # print('inside if with ns'+str(ns))
                # print('logits.shape with ns != 0')
                # print(logits.shape)

            if args.dataset_type == 'mcdataset':
                _, labels, _ = batch
            else:
                labels = batch['labels']
            
            logits, labels = accelerator.gather([logits, labels])
            if accelerator.num_processes > 1:
                if step == len(ood_ori_dataloader)-1:
                    lables = lables[: len(ood_ori_dataloader.dataset) - samples_seen]
                    logits = logits[: len(ood_ori_dataloader.dataset) - samples_seen]
                else: 
                    samples_seen += lables.shape[0]
            

            # if not args.bayes_inference_notsample and (args.modelwrapper.startswith('blob') or args.modelwrapper.startswith('light')):
            if (args.modelwrapper.startswith('blob') or args.modelwrapper.startswith('light')):
                # previously, there was no if in here, it was only the 4 lines inside if ....
                if ns !=0:
                    pre_mean_probs = torch.softmax(logits, dim=-1)
                    probs = torch.softmax(logits, dim=-1).mean(dim=1)
                    std = torch.softmax(logits, dim=-1).std(dim=1).mean()
                    logits = logits.mean(dim=1)
                if ns==0:
                    probs = torch.softmax(logits, dim=-1)
                    std = 0
            else:
                # probs = torch.softmax(logits, dim=-1)
                probs = torch.softmax(logits, dim=-1).squeeze(1)
                std = 0
            # print('num-classes:')
            # print(num_classes)
            # print('probs.shape --> ood data')
            # print(probs.shape)
            # print('labels.shape --> ood data')
            # print(labels.shape)
            # print('labels')
            # print(labels)
            acc_ood(probs, labels)
            ece_ood(probs, labels)
            nll = loss(torch.log(probs), labels, reduction = 'mean')
            if torch.isinf(nll):
                nll = loss(torch.log(probs+1e-6), lables, reduction = 'mean')
            # if args.ood_detection_method == "max-softmax":
            #     id_probs = max_softmax(probs)
            # elif args.ood_detection_method == "logits-std":
            #     id_probs = logit_std(logits)
            # elif args.ood_detection_method == "logits-entropy":
            #     id_probs = logit_entropy(probs)
            # else:
            #     raise NotImplementedError(f"OOD detection method {args.ood_detection_method} not implemented.")
            
            nll_ood.update(nll)
             

            # print('probs  (ood_ori_data):')
            # print(probs.shape)
            # print('labels (ood_ori_data):')
            # print(labels)
            
            id_probs = max_softmax(probs)
            id_prob_list = np.append(id_prob_list, id_probs)
    
    acc_ood_value = acc_ood.compute().item()
    ece_ood_value = ece_ood.compute().item()
    nll_ood_value = nll_ood.avg


    id_label_list = np.zeros_like(id_prob_list)

    model.tokenizer = dataset.tokenizer
    # model.net.module.target_ids = dataset.target_ids.squeeze(-1) # the same as 146
    model.model.target_ids = dataset.target_ids.squeeze(-1) # the same as 147

    ood_prob_list = np.array([])
    samples_seen = 0
    nlls = AverageMeter()
    # commenting the following out....
    # if num_classes is None:
    #     num_classes = args.outdim
    
    # putting the following out of if
    num_classes = args.outdim
    
    metric_kwargs = {"task": "multiclass", "num_classes": num_classes}
    acc_metric = Accuracy(**metric_kwargs).to(accelerator.device)
    ece_metric = CalibrationError(**metric_kwargs, n_bins = args.num_bins).to(accelerator.device)
    briers = AverageMeter()
    for step, batch in enumerate(dataloader):
        with torch.no_grad() and torch.inference_mode():
            # I am going to change this according to evalute function in lightblob wrapper...

            # if args.dataset_type == 'mcdataset':
            #     _, labels, _ = batch
            #     logits = model.model.forward_logits(batch, sample = not args.bayes_inference_notsample).detach() # it was model only
            # else:
            #     logits = model.model.forward_logits(batch, sample = not args.bayes_inference_notsample).detach() # it was model only
            #     labels = batch["labels"]
            # logits, labels = accelerator.gather([logits, labels])
            # if accelerator.num_processes > 1:
            #     if step == len(dataloader) - 1:
            #         labels = labels[: len(dataloader.dataset) - samples_seen]
            #         logits = logits[: len(dataloader.dataset) - samples_seen]
            #     else:
            #         samples_seen += labels.shape[0]

            if ns ==0:
                logits = model.model.forward_logits(batch, sample=False, n_samples = ns).detach()
            if ns!=0:
                logits = model.model.forward_logits(batch, sample = not args.bayes_inference_notsample, n_samples = ns).detach()
            
            if args.dataset_type == 'mcdataset':
                _, labels, _ = batch
            else:
                labels = batch['labels']
            
            logits, labels = accelerator.gather([logits, labels])
            if accelerator.num_processes > 1:
                if step == len(ood_ori_dataloader)-1:
                    lables = lables[: len(dataloader.dataset) - samples_seen]
                    logits = logits[: len(dataloader.dataset) - samples_seen]
                else: 
                    samples_seen += lables.shape[0]
            

            # if not args.bayes_inference_notsample and (args.modelwrapper.startswith('blob') or args.modelwrapper.startswith('light')):
            if (args.modelwrapper.startswith('blob') or args.modelwrapper.startswith('light')):
                # print('inside IF')
                # previously, only the 4 lines in if was in this part....
                if ns != 0:
                    pre_mean_probs = torch.softmax(logits, dim=-1)
                    probs = torch.softmax(logits, dim=-1).mean(dim=1)
                    std = torch.softmax(logits, dim=-1).std(dim=1).mean()
                    logits = logits.mean(dim=1)
                if ns==0:
                    probs = torch.softmax(logits, dim=-1)
                    std = 0
            else:
                # print('inside ELSE')
                # print('model')
                # print(args.model)
                # # print('model.model')
                # # print(args.model.model)
                # print('model_wrapper')
                # print(args.modelwrapper)
                # probs = torch.softmax(logits, dim=-1)
                probs = torch.softmax(logits, dim=-1).squeeze(1)
                std = 0
            # print('label - evaluation.py code')
            # print(labels)
            # print('probs shape - evaluation.py code')
            # print(probs.shape)
            acc_metric(probs, labels)
            ece_metric(probs, labels)
            loss_func = torch.nn.NLLLoss(reduction="mean")
            nll = loss_func(torch.log(probs), labels)
            nlls.update(nll)

            brier = (probs - F.one_hot(labels, num_classes=logits.size(-1))).pow(2).sum(dim=-1).mean()
            briers.update(brier)

            # if args.ood_detection_method == "max-softmax":
                # ood_probs = max_softmax(probs)
            # elif args.ood_detection_method == "logits-std":
            #     ood_probs = logit_std(logits)
            # elif args.ood_detection_method == "logits-entropy":
            #     ood_probs = logit_entropy(probs)
            # else:
            #     raise NotImplementedError(f"OOD detection method {args.ood_detection_method} not implemented.")
            ood_probs = max_softmax(probs)
            ood_prob_list = np.append(ood_prob_list, ood_probs)

    acc = acc_metric.compute().item()
    ece = ece_metric.compute().item()
    nll = nlls.avg
    brier = briers.avg
            
    ood_label_list = np.ones_like(ood_prob_list)
    labels = np.concatenate((id_label_list, ood_label_list))
    probs = np.concatenate((id_prob_list, ood_prob_list))

    ##### commenting the following on Apr 1 2025 by AMIR....

    # # log the scores
    # ood_detection_method = "max-softmax"
    # create_if_not_exists('log-ood-detection')
    # # with open(os.path.join('log-ood-detection', f'{args.model}-{args.modelwrapper}-{args.dataset}-{ood_detection_method}-seed{args.seed}.pkl'), 'wb') as f:
    # with open(os.path.join('log-ood-detection', f'{args.modelwrapper}-{args.dataset}-{ood_detection_method}-seed{args.seed}.pkl'), 'wb') as f:
    #     to_dump = {"labels": labels, "scores": probs}
    #     pickle.dump(to_dump, f)

    # from sklearn.metrics import roc_curve, auc
    # fpr, tpr, thresholds = roc_curve(labels, probs)
    # roc_auc = auc(fpr, tpr)

    # best_threshold_index = np.argmax(tpr - fpr)
    # best_threshold = thresholds[best_threshold_index]

    # print("Best Threshold:", best_threshold)

    # predictions = [1 if prob >= best_threshold else 0 for prob in probs]
    # total_samples = len(labels)
    # correct_predictions = sum(1 for pred, label in zip(predictions, labels) if pred == label)
    # acc_ood = correct_predictions / total_samples
            
    model.model.train(status)
    # I changed the output of this function to the following.....    
    return acc, ece, nll, brier, acc_ood_value, ece_ood_value, nll_ood_value
    #acc_ood, roc_auc


