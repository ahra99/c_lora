import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import SGD
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union
import math
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_

from .wrapperbase_ece_prf import WrapperBaseeceprf, get_linear_schedule_with_warmup, get_cnst_schedule_with_warmup
from utils.args import add_management_args, add_experiment_args, ArgumentParser
from run.evaluation import *
import time
# from run.temperature_scaling import ModelWithTemperature ################ Trying to use Temperature scaling ...

from transformers import PreTrainedModel

from peft.config import PeftConfig
from peft.tuners.lora import LoraLayer, Linear
from peft.tuners.lora.bnb import Linear8bitLt


torch.autograd.set_detect_anomaly(True)








## Model Specific Argument Parsing
def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Bayesian By Backprop, BLoB.')
    add_management_args(parser)
    add_experiment_args(parser)
    # BLoB-specific arguments.
    parser.add_argument('--bayes-train-n-samples', type=int, default=1)
    parser.add_argument('--bayes-eval-n-samples', type=int, default=1,
                        help="Number of samples to use during evaluation when training.")
    parser.add_argument('--bayes-eval-n-samples-final', type=int, default=10,
                        help="Number of samples to use during evaluation.")
    
    parser.add_argument('--bayes-eps', type=float, default=0.05)         
    parser.add_argument('--bayes-gamma', type=float, default=8)
    parser.add_argument('--bayes-kllr', type=float, default=0.02)
    parser.add_argument('--bayes-kllr-std', type=float, default=0.02)   
    parser.add_argument('--bayes-momentum', type=float, default=0.9)    
    parser.add_argument('--bayes-beta', type=float, default=0.2)
    parser.add_argument('--bayes-inference-notsample', action='store_true',
                        help='Whether to sample during inference.')
    parser.add_argument('--bayes-kl-reweighting', type = int, default=1)  
    parser.add_argument('--bayes-opt2-wd', type = float, default = 0.0005) 
    # parser.add_argument('--bayes-kl-reweighting', action='store_true',
    #                     help='Whether to use reweighting.')
    parser.add_argument('--wgs-nl-scale', type = float, default = 2)
    parser.add_argument('--wgm-nl-scale', type = float, default = 4)
    parser.add_argument('--obqa-nl-scale', type = float, default = 2)
    parser.add_argument('--boolq-nl-scale', type = float, default = 2)

    return parser



@dataclass
class LightBLoBConfig:
    bayes_eps: float = field(metadata={"help": "Bayes epsilon"})
    bayes_gamma: float = field(metadata={"help": "Bayes gamma"})
    bayes_beta: float = field(metadata={"help": "Bayes beta"})
   
def update_lora_layer(self, adapter_name):
    for adapter_name in self._active_adapter:
        if adapter_name not in self.lora_A.keys():
            continue

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

def initialize_orthogonal_matrix(n_rows, n_cols):
    """
    Initialize an orthogonal matrix of shape (n_rows, n_cols) using the QR decomposition of a random matrix.
    """
    matrix = torch.randn(n_rows, n_cols)
    q, _ = torch.linalg.qr(matrix, mode='reduced')
    return q
    

def lightblob_linear_forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
   

    previous_dtype = x.dtype
    
    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        result = self.base_layer(x, *args, **kwargs)
    elif self.merged:
        result = self.base_layer(x, *args, **kwargs)
    else:
        result = self.base_layer(x, *args, **kwargs)
        for active_adapter in self.active_adapters:
            

            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            lora_E = self.lora_E[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            
            x = x.to(lora_A.weight.dtype)
            
            bsize, fsize = x.shape[0], x.shape[1]
            oA = lora_A(dropout(x)) 
            
            E = lora_E(oA)
           
           


            
            Em, Eg = E[:,:,:self.r[active_adapter]**2], E[:,:,self.r[active_adapter]**2:]   
            
            
            if (self.bayes_eps < 1) and (self.bayes_eps > 0):
                Eg = torch.sigmoid(Eg)
            if (self.bayes_eps < 2) and (self.bayes_eps > 1):
                Eg = torch.tanh(Eg)
            if (self.bayes_eps >2):
                Eg = torch.clamp(Eg, min = -1, max = 1)

            

            self.E_m[active_adapter] = Em 
            self.E_g[active_adapter] = Eg 

            Emm = Em.reshape(bsize, fsize, self.r[active_adapter], self.r[active_adapter])

            my_output = torch.matmul(Emm, oA.unsqueeze(-1)).squeeze(-1) 
            
           
            result = result + lora_B(my_output) * scaling
            
          
    if self.blobsample:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            
            if self.bayes_eps < 0:
                E_sigma = torch.log1p(torch.exp(Eg))
            else:
                E_sigma = Eg ** 2 

            scaling = self.scaling[active_adapter]
            dropout = self.lora_dropout[active_adapter]

            x = x.to(lora_A.weight.dtype)
            if x.dim() == 2:
                r_E = torch.ones((x.size(0), self.r[active_adapter]), device=x.device, dtype=x.dtype)#.uniform_(-1, 1).sign()
                r_E.uniform_(-1, 1)
                r_E = r_E.sign()
                s_E = torch.ones((x.size(0), self.r[active_adapter]), device=x.device, dtype=x.dtype)#.uniform_(-1, 1).sign()
                s_E.uniform_(-1, 1)
                s_E = s_E.sign()
            else:
                r_E = torch.ones((x.size(0), x.size(1), self.r[active_adapter]), device=x.device, dtype=x.dtype)#.uniform_(-1, 1).sign()
                r_E.uniform_(-1, 1)
                r_E = r_E.sign()
                s_E = torch.ones((x.size(0), x.size(1), self.r[active_adapter]), device=x.device, dtype=x.dtype)#.uniform_(-1, 1).sign()
                s_E.uniform_(-1, 1)#.sign()
                s_E = s_E.sign()


            x = dropout(x)

            bsize, fsize = x.shape[0], x.shape[1]

            lora_noise_E = E_sigma * torch.randn_like(Eg)
            lora_noise_E = lora_noise_E.contiguous().view(bsize, fsize, self.r[active_adapter], self.r[active_adapter])
           
            oAn = ((x @ self.lora_A[active_adapter].weight.transpose(0, 1)) * r_E)
            my_noise = torch.matmul(lora_noise_E, oAn.unsqueeze(-1)).squeeze(-1)
            
            if torch.isnan(my_noise).any():
                print('my noise:')
                print(my_noise)

            noise = ((my_noise * s_E) @ self.lora_B[active_adapter].weight.transpose(0,1))

            if torch.isnan(noise).any():
                print('noise: ')
                print(noise)


            result = result + noise * scaling 

        result = result.to(previous_dtype)
    
    return result
    
    
def lightblob_8bitlinear_forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
    
    
    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        result = self.base_layer(x, *args, **kwargs)
    elif self.merged:
        result = self.base_layer(x, *args, **kwargs)
    else:
        
        result = self.base_layer(x, *args, **kwargs)
        for active_adapter in self.active_adapters:
            

            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            lora_E = self.lora_E[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            
            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                expected_dtype = result.dtype
                compute_dtype = lora_A.weight.dtype
                if x.dtype != compute_dtype:
                    x = x.to(compute_dtype)
            bsize, fsize = x.shape[0], x.shape[1]

            oA = lora_A(dropout(x))

            E = lora_E(oA)

    


            Em, Eg = E[:,:,:self.r[active_adapter]**2], E[:,:,self.r[active_adapter]**2:]   
            

            if (self.bayes_eps < 1) and (self.bayes_eps > 0):
                Eg = torch.sigmoid(Eg)
            if (self.bayes_eps < 2) and (self.bayes_eps > 1):
                Eg = torch.tanh(Eg)
            if (self.bayes_eps >2):
                Eg = torch.clamp(Eg, min = -1, max = 1)

            



            self.E_m[active_adapter] = Em
            self.E_g[active_adapter] = Eg
           
            Emm = Em.reshape(bsize, fsize, self.r[active_adapter], self.r[active_adapter])
            
            my_output = torch.matmul(Emm, oA.unsqueeze(-1)).squeeze(-1) 

            output = lora_B(my_output)

            if requires_conversion:
                output = output.to(expected_dtype)

            output = output * scaling
            result = result + output
            
    if self.blobsample:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            
            if self.bayes_eps < 0:
                E_sigma = torch.log1p(torch.exp(Eg)) 
            else:
                E_sigma = Eg**2 
            scaling = self.scaling[active_adapter]
            dropout = self.lora_dropout[active_adapter]

            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                expected_dtype = result.dtype
                compute_dtype = lora_A.weight.dtype
                if x.dtype != compute_dtype:
                    x = x.to(compute_dtype)

            
            if x.dim() == 2:
                r_E = torch.ones((x.size(0), self.r[active_adapter]), device=x.device, dtype=x.dtype)
                r_E.uniform_(-1, 1)
                r_E = r_E.sign()
                s_E = torch.ones((x.size(0), self.r[active_adapter]), device=x.device, dtype=x.dtype)
                s_E.uniform_(-1, 1)
                s_E = s_E.sign()
            else:
                r_E = torch.ones((x.size(0), x.size(1), self.r[active_adapter]), device=x.device, dtype=x.dtype)
                r_E.uniform_(-1, 1)
                r_E = r_E.sign()
                s_E = torch.ones((x.size(0), x.size(1), self.r[active_adapter]), device=x.device, dtype=x.dtype)
                s_E.uniform_(-1, 1)#.sign()
                s_E = s_E.sign()



            x = dropout(x)
            bsize, fsize = x.shape[0], x.shape[1]

            lora_noise_E = E_sigma * torch.randn_like(Eg)
            lora_noise_E = lora_noise_E.contiguous().view(bsize, fsize, self.r[active_adapter], self.r[active_adapter])
           
            oAn = ((x @ self.lora_A[active_adapter].weight.transpose(0, 1)) * r_E)
            my_noise = torch.matmul(lora_noise_E, oAn.unsqueeze(-1)).squeeze(-1) 
            

            if torch.isnan(my_noise).any():
                print('my noise:')
                print(my_noise)

            noise = (my_noise * s_E) @ self.lora_B[active_adapter].weight.transpose(0, 1)
            

            if torch.isnan(noise).any():
                print('noise:')
                print(noise)

            if requires_conversion:
                noise = noise.to(expected_dtype)

            result = result + noise * scaling
     
    return result

def div_posterior_prior(self) -> torch.Tensor:
    def kl_div_stable(mu_q, sigma_q, mu_p, sigma_p):
        eps = 1e-6
        kl = (math.log(sigma_p+eps) - torch.log(sigma_q.to(torch.float64)+eps) + 
              (sigma_q.to(torch.float64)**2 + (mu_q.to(torch.float64) - mu_p)**2) / (2 * (sigma_p**2)+eps) - 0.5)
        return kl.sum()
    kl = 0
    for active_adapter in self.active_adapters:
        if self.bayes_eps < 0:
            sigma_weight = torch.log1p(torch.exp(self.E_g[active_adapter]))
        else:
            sigma_weight = self.E_g[active_adapter] ** 2
            
        kl += kl_div_stable(
        self.E_m[active_adapter], 
        sigma_weight,
        0, self.bayes_beta)
    return kl

def sample(self, status = True):
    if self.training is True and status is False:
        raise ValueError("blobsample should be set to True only during training.")
    self.blobsample = status
    


class contextual_E(nn.Module):
    def __init__(self, in_feat = 8, out_feat = 128, device=None, dtype=None):
        super(contextual_E, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat

        self.e1 = nn.Linear(self.in_feat, 64, device=device, dtype=dtype)
        self.e2 = nn.Linear(64, self.out_feat, device=device, dtype=dtype)
        self.relu = nn.ReLU()

    def forward(self, x):
        
        o = self.e1(x)
        o = self.relu(o)
        o = self.e2(o)

        return o 


class clora(WrapperBaseeceprf):
    
    def __init__(self, model: PreTrainedModel, peft_config: PeftConfig, args, accelerator, adapter_name: str = "default"):
        super().__init__(model, peft_config, args, accelerator, adapter_name)
        
        # self.my_kl_track = []
        self.lightblobconfig = LightBLoBConfig(bayes_eps = self.args.bayes_eps, bayes_gamma = self.args.bayes_gamma, bayes_beta = self.args.bayes_beta)
        self._modify_lora_layers(self.base_model)
        if args.load_checkpoint:
            self.load_adapter(args.load_path, adapter_name)
        
        self.i = 1 
        self.M = 0 

        self.train_n_samples = self.args.bayes_train_n_samples
        self.eval_n_samples = self.args.bayes_eval_n_samples
        
        if self.args.bayes_kl_reweighting == 1:
            self.kl_reweighting = True
        else: 
            self.kl_reweighting = False


        if self.args.max_train_steps == 0 :
            num_training_steps = self.args.num_samples * self.args.n_epochs // self.args.batch_size
        else:
            num_training_steps = self.args.max_train_steps
        warmup_steps = num_training_steps * self.args.warmup_ratio


        params= [param for pname, param in self.named_parameters() if ('lora_E' in pname)]
        self.opt2 = SGD(
                [{'params': params}],
                lr=args.bayes_kllr
            )
        
        
        
        self.scheduler2 = get_linear_schedule_with_warmup(self.opt2, warmup_steps, num_training_steps)
        
        self.counter = 0
        
    def _modify_lora_layers(self, module):
        """
        Recursively go through the model and modify LoraLayer instances.
        """
        for name, child in module.named_children():
            if isinstance(child, LoraLayer) and isinstance(child, Linear):
                setattr(child, 'update_layer', update_lora_layer.__get__(child, child.__class__))
                child.update_layer(child._active_adapter)
                self._wrap_lora_layer(child)
                # modify existing methods
                setattr(child, 'forward', lightblob_linear_forward.__get__(child, child.__class__))
                # add new methods
                setattr(child, 'div_posterior_prior', div_posterior_prior.__get__(child, child.__class__))
                setattr(child, 'sample', sample.__get__(child, child.__class__))
            if isinstance(child, LoraLayer) and isinstance(child, Linear8bitLt):
                setattr(child, 'update_layer', update_lora_layer.__get__(child, child.__class__))
                child.update_layer(child._active_adapter)
                self._wrap_lora_layer(child)
                # modify existing methods
                setattr(child, 'forward', lightblob_8bitlinear_forward.__get__(child, child.__class__))
                # add new methods
                setattr(child, 'div_posterior_prior', div_posterior_prior.__get__(child, child.__class__))
                setattr(child, 'sample', sample.__get__(child, child.__class__))
            else:
                self._modify_lora_layers(child)
                
    def _wrap_lora_layer(self, lora_layer):
        lora_layer.lora_E = nn.ModuleDict({})
        
        lora_layer.bayes_eps = self.lightblobconfig.bayes_eps
        lora_layer.bayes_gamma = self.lightblobconfig.bayes_gamma
        lora_layer.bayes_beta = self.lightblobconfig.bayes_beta
        lora_layer.blobsample = True
    

        lora_layer.E_m = {}
        lora_layer.E_g = {}

        # Loop through active adapters to set parameters
        for adapter_name in lora_layer._active_adapter:
            device = lora_layer.lora_A[adapter_name].weight.device
            dtype = lora_layer.lora_A[adapter_name].weight.dtype

          
            lora_layer.lora_E[adapter_name] = contextual_E(in_feat = lora_layer.r[adapter_name], out_feat = lora_layer.r[adapter_name]**2 * 2, device=device, dtype =dtype)
           

            
            if adapter_name in lora_layer.lora_A.keys():
                
                lora_layer._move_adapter_to_device_of_base_layer(adapter_name)
                lora_layer.set_adapter(lora_layer.active_adapters)
            

        return

   
    def div_posterior_prior(self, module):
        kl = 0
        for name, child in module.named_children():
            if isinstance(child, LoraLayer):
                kl_ = child.div_posterior_prior()
                # if not math.isnan(kl_):
                kl += kl_
            else:
                kl += self.div_posterior_prior(child)
        return kl
    
    def sample(self, module, status = True):
        """
        Set the sampling status of the model.
        """
        for name, child in module.named_children():
            if isinstance(child, LoraLayer):
                child.sample(status)
            else:
                self.sample(child, status)
                           
    def forward_logits(self, batch, sample=True, n_samples=1, **kwargs) -> torch.Tensor:
        if self.args.dataset_type == 'mcdataset':
            inputs, _, _ = batch
            if not sample:
                self.sample(self.base_model, False)
                output = self.base_model(**inputs)
                logits = output.logits[:, -1, self.target_ids]
                self.sample(self.base_model, True)
                return logits
            else:
                logits_list = []
                for _ in range(n_samples):
                    output = self.base_model(**inputs)
                    logits = output.logits[:, -1, self.target_ids]
                    logits_list.append(logits)
                return torch.stack(logits_list, dim = 1)
        else:
            if not sample:
                self.sample(self.base_model, False)
                res = self.base_model(**batch).logits
                self.sample(self.base_model, True)
                return res
            else:
                res = []
                for _ in range(n_samples):
                    res.append(self.base_model(**batch).logits)    
                return torch.stack(res, dim = 1)
                
    def fit(self, train_loader, eval_loader):
        
        stop_criteria = torch.inf 
        # self.counter = 0

        nll_losses = AverageMeter()
        kl_losses = AverageMeter()
        elbo_losses = AverageMeter()
        accs = AverageMeter()   
        samples_seen = 0
        
        with tqdm(total=len(train_loader), desc=f"Epoch {self.args.epoch+1}/{self.args.n_epochs}", leave=False) as pbar:
            for i, batch in enumerate(train_loader): 
                if self.args.dataset_type == 'mcdataset':
                    _, golds, _ = batch
                elif self.args.dataset_type == 'bertds':
                    golds = batch['labels']
                else:
                    raise NotImplementedError(f"Dataset type {self.args.dataset_type} not implemented.")

                                
                logits = self.forward_logits(batch, sample=True, n_samples=self.train_n_samples).mean(1)
                output = torch.log_softmax(logits, dim=1)

                
                if torch.isnan(output).any():
                    print('nan in ouptuts')

                nll = self.loss(output, golds, reduction='mean')

                
                if self.args.dataset == 'winogrande_s':
                    nll = self.args.wgs_nl_scale * nll
                if self.args.dataset == 'winogrande_m':
                    nll = self.args.wgm_nl_scale * nll
                if self.args.dataset == 'obqa':
                    nll = self.args.obqa_nl_scale * nll
                if self.args.dataset == 'boolq':
                    nll = self.args.boolq_nl_scale * nll
                
                
                self.accelerator.backward(nll, retain_graph = True) 

                kl_divs = []
                for _ in range(self.train_n_samples):
                    if hasattr(self.base_model, 'module'):
                        kl_divs.append(self.div_posterior_prior(self.base_model.module))
                    else:
                        kl_divs.append(self.div_posterior_prior(self.base_model))
                
                kl = torch.mean(torch.stack(kl_divs), dim=0) * 1/65 # avg kl
               

                if self.kl_reweighting:
                    if self.i % self.M == 0:
                        i = self.M
                    else:
                        i = self.i % self.M
                    self.pi = 2**i/(2**(self.M+1)-1)
                    self.i+=1
                else:
                    self.pi = 1 / (self.M) 
                kl_div =  kl * self.pi *0.000001 
            
                
                
                self.accelerator.backward(kl_div)


                
                
                self.opt.step()
                self.opt.zero_grad()
                self.scheduler.step()
                
                self.opt2.step()
                self.opt2.zero_grad()
                self.scheduler2.step()
                
               
                acc = accuracy_topk(output.data, golds)

                loss, acc, nll_loss, kl = (kl+nll).detach().cpu().numpy(), acc.item(), nll.detach().cpu().numpy(), kl_div.detach().cpu().numpy()

                if self.args.dataset_type == 'mcdataset':
                    _, classes, _ = batch
                    references = self.accelerator.gather(classes)
                else:
                    references = self.accelerator.gather(batch["labels"])
                if self.accelerator.num_processes > 1:
                    if i == len(train_loader) - 1:
                        references = references[: len(train_loader.dataset) - samples_seen]
                    else:
                        samples_seen += references.shape[0]
                len_batch = references.shape[0]
                kl_losses.update(kl, len_batch)
                nll_losses.update(nll_loss, len_batch)
                elbo_losses.update(loss, len_batch)
                accs.update(acc, len_batch)
                
                assert not math.isnan(nll_loss)
                assert not math.isnan(kl)
                if self.accelerator.is_local_main_process:
                    if self.wandb_logger is not None:
                        self.wandb_logger.log({
                                'train_acc': accs.avg, 
                                'train_nll_loss': nll_losses.avg, 
                                'kl_loss': kl_losses.avg, 
                                'lr': self.opt.param_groups[0]['lr'],
                                'kllr': self.opt2.param_groups[0]['lr'],      
                                'pi'+str(self.args.bayes_kl_reweighting): self.pi,
                            })
                    
                
                self.step += self.accelerator.num_processes
                pbar.update(1)
                if self.step >= self.args.eval_per_steps:
                    print('accs.avg: ', accs.avg)
                    self.step -= self.args.eval_per_steps
                    v_acc,v_ecc,v_nll,_ = self.evaluate(self.val_loader)
                    perf_check = (1-v_acc) * (v_ecc * 100) 
                    if (perf_check < stop_criteria):
                        print(f'INSIDE perf check < stop criteria ----- counter: {self.counter}')
                        if self.args.dataset != 'boolq':
                            v_thresh = 0.6
                        else:
                            v_thresh = 0.7
                        if v_acc > v_thresh:
                            start_time = time.time()
                            print(f'INSIDE v_acc > {v_thresh} --->  v_acc: ', v_acc)
                            
                            for sample_num in [0]:
                                self.eval_n_samples = sample_num
                                self.evaluate(eval_loader, val_stat = '-best-prf-model') 
                            time_secs = time.time()-start_time
                            print(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Inference time in secs {time_secs}, and in mins {time_secs/60} >>>>>>>>>>>>>>>>>>>>>>>>>>>>')

                            self.eval_n_samples = 1 
                            stop_criteria = perf_check 

                            
                            if self.args.dataset == 'obqa':
                                
                                self.accelerator.wait_for_everyone()
                                
                                

                                if self.accelerator.is_main_process:
                                    
                                    save_folder = f'bstm_obqa/{self.counter}/{sample_num}/{self.args.modelwrapper}/{self.args.model}/{self.args.dataset}/{self.args.checkpoint_dic_name}/{self.args.seed}'
                                    create_if_not_exists(save_folder)

                                    whole_model = self
                                    

                                    whole_model.save_pretrained(save_folder, save_function = self.accelerator.save)
                                    

                                    self.counter += 1
                                    

                    
    def evaluate(self, eval_loader, val_stat = None):
        valid_best_performing = val_stat if val_stat != None else ""  
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

        print('self.eval_n_samples:', self.eval_n_samples)
        self.eval()
        status = self.training
        nlls = AverageMeter()
        metric_kwargs = {"task": "multiclass", "num_classes": self.num_classes}
        acc_metric = Accuracy(**metric_kwargs).to(self.accelerator.device)
        ece_metric = CalibrationError(**metric_kwargs, n_bins = self.args.num_bins).to(self.accelerator.device)
        briers = AverageMeter()


        
        nlls_t_scaled = AverageMeter()
        acc_metric_t_scaled = Accuracy(**metric_kwargs).to(self.accelerator.device)
        ece_metric_t_scaled = CalibrationError(**metric_kwargs, n_bins = self.args.num_bins).to(self.accelerator.device)
        
        
        nlls_crct = AverageMeter()
        acc_metric_crct = Accuracy(**metric_kwargs).to(self.accelerator.device)
        ece_metric_crct = CalibrationError(**metric_kwargs, n_bins = self.args.num_bins).to(self.accelerator.device)

        
        nlls_crct_t_scaled = AverageMeter()
        acc_metric_crct_t_scaled = Accuracy(**metric_kwargs).to(self.accelerator.device)
        ece_metric_crct_t_scaled = CalibrationError(**metric_kwargs, n_bins = self.args.num_bins).to(self.accelerator.device)

        
        nlls_crct_prf = AverageMeter()
        acc_metric_crct_prf = Accuracy(**metric_kwargs).to(self.accelerator.device)
        ece_metric_crct_prf = CalibrationError(**metric_kwargs, n_bins = self.args.num_bins).to(self.accelerator.device)

        
        nlls_crct_prf_t_scaled = AverageMeter()
        acc_metric_crct_prf_t_scaled = Accuracy(**metric_kwargs).to(self.accelerator.device)
        ece_metric_crct_prf_t_scaled = CalibrationError(**metric_kwargs, n_bins = self.args.num_bins).to(self.accelerator.device)
        

        
        #############################
        ##### tmp scaling ...........  
        #############################

        logits_list = [] 
        labels_list = [] 

        samples_seen_tmp = 0
        
        for step, batch in enumerate(self.val_loader): 
            with torch.no_grad() and torch.inference_mode():
                
                tmp_logits = self.forward_logits(batch, sample = False, n_samples=1).detach()
                logits_list.append(tmp_logits)
                
                if self.args.dataset_type == 'mcdataset':
                    _, labels_tmp, _ = batch
                else:
                    labels_tmp = batch["labels"]
                tmp_logits, labels_tmp = self.accelerator.gather([tmp_logits, labels_tmp])
                if self.accelerator.num_processes > 1:
                    if step == len(self.val_loader) - 1: 
                        labels_tmp = labels_tmp[: len(self.val_loader.dataset) - samples_seen_tmp] 
                    else:
                        samples_seen_tmp += labels_tmp.shape[0]


                labels_list.append(labels_tmp)


        logits_tmp = torch.cat(logits_list).to(self.accelerator.device)
        labels_tmp = torch.cat(labels_list).to(self.accelerator.device)
        optimizer = optim.LBFGS([self.temperature], lr = 0.01, max_iter = 50)
        nll_criterion_tmp = nn.CrossEntropyLoss().to(self.accelerator.device)
        ece_criterion_tmp = _ECELoss().to(self.accelerator.device)

        def eval_tmp():
            optimizer.zero_grad()
            tmp = self.temperature.unsqueeze(1).expand(logits_tmp.size(0), logits_tmp.size(1)).to(self.accelerator.device)
            loss_tmp = nll_criterion_tmp(logits_tmp/tmp, labels_tmp)
            loss_tmp.backward()
            return loss_tmp
        optimizer.step(eval_tmp)
        
    
        #############################
        ##### tmp scaling ...........  
        #############################


        p_all = torch.tensor([])
        p_all_ns = torch.tensor([])
        p_all_ns_tnsr = torch.tensor([])
        p_all_ns_tnsr_ts = torch.tensor([])

        labels_all_ns = torch.tensor([])

        samples_seen = 0
        # for ns in [0,5,10]:
        ns = self.eval_n_samples
        for step, batch in enumerate(eval_loader):
            with torch.no_grad() and torch.inference_mode():
                
                if ns ==0:
                   logits = self.forward_logits(batch, sample = False, n_samples=ns).detach() 
                if ns != 0:
                   logits = self.forward_logits(batch, sample = not self.args.bayes_inference_notsample, n_samples=ns).detach() 
		        
                if self.args.dataset_type == 'mcdataset':
                    _, labels, _ = batch
                else:
                    labels = batch["labels"]
                logits, labels = self.accelerator.gather([logits, labels])
                if self.accelerator.num_processes > 1:
                    if step == len(eval_loader) - 1:
                        labels = labels[: len(eval_loader.dataset) - samples_seen]
                        logits = logits[: len(eval_loader.dataset) - samples_seen]
                    else:
                        samples_seen += labels.shape[0]
               



                if ns == 0:
                    tmp_mat = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1)).to(self.accelerator.device)
                    tmp_mat = tmp_mat.detach()
                    probs_t_scaled = torch.softmax(logits/tmp_mat, dim=-1)

                    probs = torch.softmax(logits, dim=-1)
                    std = 0
                if ns!= 0:
                    
                    tmp_mat = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1), logits.size(2)).to(self.accelerator.device)
                    tmp_mat = tmp_mat.detach()
                    probs_t_scaled = torch.softmax(logits/tmp_mat, dim = -1).mean(dim=1) 

                    probs = torch.softmax(logits, dim=-1).mean(dim=1)
                    std = torch.softmax(logits, dim=-1).std(dim=1).mean()

                    
                    
                

                acc_metric(probs, labels)
                ece_metric(probs, labels)
                nll = self.loss(torch.log(probs), labels, reduction='mean')
                if torch.isinf(nll):
                    
                    nll = self.loss(torch.log(probs + 1e-6 ), labels, reduction='mean')
                    
                if torch.isnan(nll):
                    if self.accelerator.is_local_main_process:
                        print('nll:', nll)
                        print('probs:', probs)
                        print('logits:', logits)
                        exit()
                nlls.update(nll)
                

                brier = (probs - F.one_hot(labels, num_classes=logits.size(-1))).pow(2).sum(dim=-1).mean()
                briers.update(brier)
                

                
                acc_metric_t_scaled(probs_t_scaled, labels)
                ece_metric_t_scaled(probs_t_scaled, labels)
                nll_t_scaled = self.loss(torch.log(probs_t_scaled), labels, reduction='mean')
                if torch.isinf(nll_t_scaled):
                    nll_t_scaled = self.loss(torch.log(probs_t_scaled + 1e-6 ), labels, reduction='mean')
                    
                if torch.isnan(nll_t_scaled):
                    if self.accelerator.is_local_main_process:
                        print('nll_t_scaled:', nll_t_scaled)
                        print('probs_t_scaled:', probs_t_scaled)
                        print('logits_t_scaled:', logits/tmp_mat)
                        exit()
                nlls_t_scaled.update(nll_t_scaled)

                
                

        val_acc = acc_metric.compute().item()
        val_ece = ece_metric.compute().item()
        val_nll = nlls.avg
        val_brier = briers.avg

        

        
        val_acc_t_scaled = acc_metric_t_scaled.compute().item()
        val_ece_t_scaled = ece_metric_t_scaled.compute().item()
        val_nll_t_scaled = nlls_t_scaled.avg


       
        
        tmp = self.temperature.unsqueeze(1).expand(logits_tmp.size(0), logits_tmp.size(1)).to(self.accelerator.device)
        tmp = tmp.detach()
        print('\ntmp device:::::::::')
        print(tmp.device)
        print('\nlogits device:::::::::')
        print(logits_tmp.device)
        lg_scaled = logits_tmp/tmp

        before_temperature_nll = nll_criterion_tmp(logits_tmp, labels_tmp).item()
        before_temperature_ece = ece_criterion_tmp(logits_tmp, labels_tmp).item()
        after_temperature_ece  = ece_criterion_tmp(lg_scaled, labels_tmp).item()
        after_temperature_nll  = nll_criterion_tmp(lg_scaled, labels_tmp).item()

        new_probs_tscaled = torch.softmax(lg_scaled, dim = 1)
        new_probs = torch.softmax(logits_tmp, dim = 1)
        ece_metric_double_check = CalibrationError(**metric_kwargs, n_bins = self.args.num_bins).to(self.accelerator.device)
        b_tmp_ece_org = ece_metric_double_check(new_probs, labels_tmp).item()
        a_tmp_ece_org = ece_metric_double_check(new_probs_tscaled, labels_tmp).item()

        if self.accelerator.is_local_main_process:
            if self.wandb_logger is not None:
                self.wandb_logger.log( {
                    'before_tmp_nll_validation'+valid_best_performing : before_temperature_nll,
                    'before_tmp_ece_validation'+valid_best_performing : before_temperature_ece,
                    'after_tmp_nll_validation'+valid_best_performing  : after_temperature_nll,
                    'after_tmp_ece_validation'+valid_best_performing  : after_temperature_ece,
                    # 'b_tmp_ece_org0'+valid_best_performing  : b_tmp_ece_org,
                    # 'a_tmp_ece_org0'+valid_best_performing  : a_tmp_ece_org,
                })


        self.train(status)
        
        

        if self.accelerator.is_local_main_process:
            if self.wandb_logger is not None:
                self.wandb_logger.log({
                    'val_acc'+str(ns)+valid_best_performing: val_acc, 
                    'val_ece'+str(ns)+valid_best_performing: val_ece, 
                    'val_nll'+str(ns)+valid_best_performing: val_nll, 
                    'val_acc_t_scaled'+str(ns)+valid_best_performing: val_acc_t_scaled, 
                    'val_ece_t_scaled'+str(ns)+valid_best_performing: val_ece_t_scaled, 
                    'val_nll_t_scaled'+str(ns)+valid_best_performing: val_nll_t_scaled, 
                    # 'std'+str(ns)+valid_best_performing:std,
                    'val_brier'+str(ns)+valid_best_performing: val_brier,
                })
        return val_acc, val_ece, val_nll, val_brier

    def prepare_for_fit_evaluate(self, dataset, wandb_logger=None):
        """
        Prepare the model for training and evaluation.
        """
        self.wandb_logger = wandb_logger
        train_loader, test_loader = dataset.train_dataloader, dataset.test_dataloader
        if self.args.testing_set == 'train_val':
            val_loader = dataset.val_dataloader
            val_loader = self.accelerator.prepare(val_loader)
            self.val_loader = val_loader
        
        
        if self.args.subset_size > 0: 
            val_loader = dataset.valid_dataloader
            val_loader = self.accelerator.prepare(val_loader)
            self.val_loader = val_loader

        if self.args.dataset_type == 'mcdataset':
            self.target_ids = dataset.target_ids.squeeze(-1)
            
        l_train = len(train_loader)
        
        num_update_steps_per_epoch = math.ceil(len(train_loader) / self.args.gradient_accumulation_steps)
        if self.args.max_train_steps == 0:
            self.args.max_train_steps = self.args.n_epochs * num_update_steps_per_epoch
        self.args.n_epochs = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch) if self.args.ood_ori_dataset is None else 0
        if self.args.early_stop_steps > 0:
            self.earlystop_n_epochs = math.ceil(self.args.early_stop_steps / num_update_steps_per_epoch) if self.args.ood_ori_dataset is None else 0
        else:
            self.earlystop_n_epochs = 0
        if self.accelerator.is_local_main_process:
            if self.args.subset_size > 0: 
                print('len(val_loader):', len(val_loader))
            print('len(train_loader):', len(train_loader))
            print('num of epochs:', self.args.n_epochs)
        self.step = 0
        
        self.base_model, self.opt, train_loader, test_loader, self.scheduler, self.scheduler2, self.opt2 = self.accelerator.prepare(self.base_model, self.opt, train_loader, test_loader, self.scheduler, self.scheduler2, self.opt2)
        
        self.train_loader = train_loader
        self.test_loader = test_loader
        if self.args.bayes_kl_reweighting:
            self.M = int(100 * (dataset.num_samples ** (math.pi/self.args.bayes_gamma)) / (l_train / len(train_loader)) / self.args.batch_size)
        else:
            self.M = len(train_loader)
        
        print("M:", self.M)

    


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
