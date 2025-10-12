import torch.nn as nn

from run import get_modelwrapper

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import (
    get_peft_model,
    LoraConfig,
    PeftModel,
    PeftConfig,
)

class CausalLM(nn.Module):
    def __init__(self, args, accelerator=None, **kwargs) -> None:
        super().__init__()
        if accelerator is not None:
            accelerator.wait_for_everyone()
        if args.load_checkpoint:
            print('=====================')
            if args.ood_ori_dataset is not None:
                if ('light' in args.modelwrapper) and (args.bayes_eval_n_samples_final == 0) : 
                    if args.seed == 3:# or args.seed == 6 or args.seed == 9:#1: # since I did some weird way of saving the weights for obqa seed 1 ...... Feb 19 ===> modified on Apr 22 for debugging
                        # args.load_path = f'/scratch/user/amir_hossein_rahmati/BayesPEFT/bayesian-peft/last_models/{args.modelwrapper}/{args.model}/{args.ood_ori_dataset}/{args.checkpoint_dic_name}/3'#/adam'#{args.load_model_path}'
                        args.load_path = f'/scratch/user/amir_hossein_rahmati/BayesPEFT/bayesian-peft/bstm_obqa/5/{args.bayes_eval_n_samples_final}/{args.modelwrapper}/{args.model}/{args.ood_ori_dataset}/{args.checkpoint_dic_name}/{args.seed}'#/adam'#{args.load_model_path}'
                        # args.load_path = f'/scratch/user/amir_hossein_rahmati/BayesPEFT/bayesian-peft/bstm/0/{args.modelwrapper}/{args.model}/{args.ood_ori_dataset}/{args.checkpoint_dic_name}/{args.seed}'#/adam'#{args.load_model_path}'
                        # args.load_path = f'/scratch/user/amir_hossein_rahmati/BayesPEFT/bayesian-peft/bests/5/{args.modelwrapper}/{args.model}/{args.ood_ori_dataset}/{args.checkpoint_dic_name}/{args.seed}'#/adam'#{args.load_model_path}'
                        # args.load_path = f'/scratch/user/amir_hossein_rahmati/BayesPEFT/bayesian-peft/save_best_model_check1/{args.modelwrapper}/{args.model}/{args.ood_ori_dataset}/{args.checkpoint_dic_name}/{args.seed}'#/adam'#{args.load_model_path}'
                        # args.load_path = f'/scratch/user/amir_hossein_rahmati/BayesPEFT/bayesian-peft/checkpoints/{args.modelwrapper}/{args.model}/{args.ood_ori_dataset}/default/adam.pt/'#/adam'#{args.load_model_path}'
                    elif args.seed == 6:
                        args.load_path = f'/scratch/user/amir_hossein_rahmati/BayesPEFT/bayesian-peft/bstm_obqa/8/{args.bayes_eval_n_samples_final}/{args.modelwrapper}/{args.model}/{args.ood_ori_dataset}/{args.checkpoint_dic_name}/{args.seed}'#/adam'#{args.load_model_path}'

                    elif args.seed == 9:
                        args.load_path = f'/scratch/user/amir_hossein_rahmati/BayesPEFT/bayesian-peft/bstm_obqa/3/{args.bayes_eval_n_samples_final}/{args.modelwrapper}/{args.model}/{args.ood_ori_dataset}/{args.checkpoint_dic_name}/{args.seed}'#/adam'#{args.load_model_path}'

                if ('light' in args.modelwrapper) and (args.bayes_eval_n_samples_final == 10) : 
                    if args.seed == 3:# or args.seed == 6 or args.seed == 9:#1: # since I did some weird way of saving the weights for obqa seed 1 ...... Feb 19 ===> modified on Apr 22 for debugging
                        # args.load_path = f'/scratch/user/amir_hossein_rahmati/BayesPEFT/bayesian-peft/last_models/{args.modelwrapper}/{args.model}/{args.ood_ori_dataset}/{args.checkpoint_dic_name}/3'#/adam'#{args.load_model_path}'
                        args.load_path = f'/scratch/user/amir_hossein_rahmati/BayesPEFT/bayesian-peft/bstm_obqa/2/{args.bayes_eval_n_samples_final}/{args.modelwrapper}/{args.model}/{args.ood_ori_dataset}/{args.checkpoint_dic_name}/{args.seed}'#/adam'#{args.load_model_path}'
                        # args.load_path = f'/scratch/user/amir_hossein_rahmati/BayesPEFT/bayesian-peft/bstm/0/{args.modelwrapper}/{args.model}/{args.ood_ori_dataset}/{args.checkpoint_dic_name}/{args.seed}'#/adam'#{args.load_model_path}'
                        # args.load_path = f'/scratch/user/amir_hossein_rahmati/BayesPEFT/bayesian-peft/bests/5/{args.modelwrapper}/{args.model}/{args.ood_ori_dataset}/{args.checkpoint_dic_name}/{args.seed}'#/adam'#{args.load_model_path}'
                        # args.load_path = f'/scratch/user/amir_hossein_rahmati/BayesPEFT/bayesian-peft/save_best_model_check1/{args.modelwrapper}/{args.model}/{args.ood_ori_dataset}/{args.checkpoint_dic_name}/{args.seed}'#/adam'#{args.load_model_path}'
                        # args.load_path = f'/scratch/user/amir_hossein_rahmati/BayesPEFT/bayesian-peft/checkpoints/{args.modelwrapper}/{args.model}/{args.ood_ori_dataset}/default/adam.pt/'#/adam'#{args.load_model_path}'
                    elif args.seed == 6:
                        args.load_path = f'/scratch/user/amir_hossein_rahmati/BayesPEFT/bayesian-peft/bstm_obqa/6/{args.bayes_eval_n_samples_final}/{args.modelwrapper}/{args.model}/{args.ood_ori_dataset}/{args.checkpoint_dic_name}/{args.seed}'#/adam'#{args.load_model_path}'

                    elif args.seed == 9:
                        args.load_path = f'/scratch/user/amir_hossein_rahmati/BayesPEFT/bayesian-peft/bstm_obqa/1/{args.bayes_eval_n_samples_final}/{args.modelwrapper}/{args.model}/{args.ood_ori_dataset}/{args.checkpoint_dic_name}/{args.seed}'#/adam'#{args.load_model_path}'


                elif args.modelwrapper == 'blob':
                    args.load_path = f'/scratch/user/amir_hossein_rahmati/BayesPEFT/bayesian-peft/last_models/{args.modelwrapper}/{args.model}/{args.ood_ori_dataset}/{args.checkpoint_dic_name}/{args.seed}'

                elif ('light' in args.modelwrapper) and (args.bayes_eval_n_samples_final < 0):
                    args.load_path = f'/scratch/user/amir_hossein_rahmati/BayesPEFT/bayesian-peft/last_models/{args.modelwrapper}/{args.model}/{args.ood_ori_dataset}/{args.checkpoint_dic_name}/{args.seed}'

                else: 
                    # /teamspace/studios/this_studio/bayesian-peft/last_models/mle/meta-llama/Llama-2-7b-hf/obqa/default/3/
                    args.load_path = f'/teamspace/studios/this_studio/bayesian-peft/last_models_1/{args.modelwrapper}/{args.model}/{args.ood_ori_dataset}/{args.checkpoint_dic_name}/{args.seed}'#/adam'#{args.load_model_path}'
                    # args.load_path = f'/scratch/user/amir_hossein_rahmati/BayesPEFT/bayesian-peft/checkpoints/{args.modelwrapper}/{args.model}/{args.ood_ori_dataset}/{args.checkpoint_dic_name}/{args.seed}'#/adam'#{args.load_model_path}'
            else:
                args.load_path = f'checkpoints/{args.modelwrapper}/{args.model}/{args.dataset}/{args.load_model_path}'
            print('Loading model from: ', args.load_path)
            peft_config = PeftConfig.from_pretrained(args.load_path, is_trainable=True)
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=args.load_in_8bit
            )
            # /teamspace/studios/this_studio/bayesian-peft/last_models/
            model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, quantization_config=bnb_config, cache_dir='/teamspace/studios/this_studio/bayesian-peft/')
            # model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, quantization_config=bnb_config, cache_dir='/scratch/user/amir_hossein_rahmati/BayesPEFT/bayesian-peft')
            self.model = PeftModel.from_pretrained(model, args.load_path, is_trainable=True)
            modelwrapper = get_modelwrapper(args.modelwrapper)
            self.model = modelwrapper(self.model, peft_config, args, accelerator, adapter_name="default")
            self.model.print_trainable_parameters()

            print('Model loaded successfully')
            print('=====================')
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=args.load_in_8bit
            )
            if args.load_model_path is not None:
                model = AutoModelForCausalLM.from_pretrained(args.load_model_path, quantization_config=bnb_config)#, cache_dir='/scratch/user/amir_hossein_rahmati/BayesPEFT/bayesian-peft')
            else:
                model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb_config)#, cache_dir='/scratch/user/amir_hossein_rahmati/BayesPEFT/bayesian-peft')
            if args.apply_classhead_lora:
                target_modules=["q_proj", "v_proj", "lm_head"]
            elif args.apply_qkv_head_lora:
                target_modules=["q_proj", "v_proj", "k_proj", "lm_head"]
            else:
                target_modules=["q_proj", "v_proj"]
            
            peft_config = LoraConfig(task_type="CAUSAL_LM", inference_mode=False, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, target_modules=target_modules)
            self.model = get_peft_model(model, peft_config)
            modelwrapper = get_modelwrapper(args.modelwrapper)
            self.model = modelwrapper(self.model, peft_config, args, accelerator, adapter_name="default")
            self.model.print_trainable_parameters()

