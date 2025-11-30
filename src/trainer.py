# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

import inspect
import os
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import copy
import numpy as np
import copy
from tqdm.auto import tqdm
from transformers import Trainer
from utils.blip2_classifier import BLIP2Classifier
import pandas as pd

# Import unified utilities
from utils.llm_generator import LLMGenerator
from utils.nsfw_filter import NSFWFilterManager



# Integrations must be imported before ML frameworks:
from transformers.integrations import (  # isort: split
    hp_params,
)

import numpy as np
import torch
import torch.distributed as dist

from transformers import __version__
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled

from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    ExportableState,
)

from transformers.trainer_utils import (
    HPSearchBackend,
    TrainOutput,
    speed_metrics,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.utils import (
    is_sagemaker_mp_enabled,
    logging,
)
from PIL import Image
from datetime import datetime
from sentence_transformers import SentenceTransformer

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

logger = logging.get_logger(__name__)


# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"




    

class OurTrainer(Trainer):
    def __init__(self, llm_model, t2i_model, tokenizer, pool, *args, **kwargs):
        super().__init__(model=llm_model, *args, **kwargs)
        self.llm_model = llm_model
        self.t2i_model = t2i_model
        self.tokenizer = tokenizer
        self.pool = pool
        self.args = self.args

        # Initialize unified filter manager
        filter_type = getattr(self.args, 'filter_type', None)
        self.filter_manager = NSFWFilterManager(filter_type, device=self.args.device) if filter_type else None

        blip2_checkpoint = f"files/checkpoints/blip2/{self.args.category}.pt"
        self.blip2_model = BLIP2Classifier(self.args.device, blip2_checkpoint)
        self.output_dir = self.args.output_dir

        # Initialize LLM generator with unified logic
        mask_path = "files/checkpoints/mask/mask.npy"
        self.llm_generator = LLMGenerator(
            model=self.llm_model,
            tokenizer=self.tokenizer,
            args=self.args,
            mask_path=mask_path
        )

        self.cossimemb_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', trust_remote_code=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize adaptive similarity weight
        if hasattr(self.args, 'adaptive_similarity') and self.args.adaptive_similarity:
            self.adaptive_similarity = True
            self.similarity_weight = self.args.similarity_weight  # Initialize with base weight
            self.dynamic_similarity_weight = self.similarity_weight  # Initialize with base weight
            self.similarity_threshold = 0.35
            self.base_similarity_weight = 0.8
            self.similarity_scale = 10
            print(f"Adaptive similarity enabled: initial_weight={self.similarity_weight}, threshold={self.similarity_threshold}")
        else:
            self.adaptive_similarity = False
            self.dynamic_similarity_weight = self.args.similarity_weight
            print(f"Using static similarity weight: {self.dynamic_similarity_weight}")

    def update_adaptive_similarity_weight(self, similarity):
        """
        Update similarity weight based on current similarity score.

        Formula: weight = base_weight * exp(scale * (similarity - threshold))
        """
        if not self.adaptive_similarity:
            return

        # Calculate new weight using exponential formula
        new_weight = self.base_similarity_weight * np.exp(self.similarity_scale * (similarity - self.similarity_threshold))

        # Update weight if changed
        if abs(new_weight - self.dynamic_similarity_weight) > 0.001:  # Only log significant changes
            print(f"Updating similarity weight: {self.dynamic_similarity_weight:.3f} → {new_weight:.3f} (similarity={similarity:.3f})")
            self.dynamic_similarity_weight = new_weight


    from transformers.trainer_pt_utils import (
        _get_learning_rate,
        log_metrics,
        metrics_format,
        save_metrics,
        save_state,
    )

    def _inner_training_loop(
        self,
        batch_size=None,
        args=None,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
    ):
        """
        We overload the original training loop to add linear probing and MeZO. Search key word "MeZO added"
        for those updates.
        """
        self._train_batch_size = args.train_batch_size
        # Data loader and number of training steps

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch:"zo" num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = (
            args.train_batch_size * args.gradient_accumulation_steps * args.world_size
        )
        max_steps = args.max_steps
        num_train_epochs = args.num_train_epochs
        num_update_steps_per_epoch = args.max_steps // num_train_epochs + int(
            args.max_steps % num_train_epochs > 0
        )
        num_train_samples = args.max_steps * total_train_batch_size

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.args.fsdp

        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self,
                num_training_steps=max_steps,
                resume_from_checkpoint=resume_from_checkpoint,
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # self.state = TrainerState()
        self.state = TrainerState(
                    stateful_callbacks=[
                        cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
                    ]
                )
        self.state.is_hyper_param_search = trial is not None
        self.state.logging_steps = args.logging_steps
        self.state.save_steps = args.save_steps
        self.state.eval_steps = args.eval_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(
            f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
        )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {max_steps}")
        logger.info(
            f"  Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (
                    num_update_steps_per_epoch
                )
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info(
                "  Continuing training from checkpoint, will skip to saved global_step"
            )
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(
                f"  Continuing training from global step {self.state.global_step}"
            )
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(
                        total=steps_trained_in_current_epoch
                    )
                    steps_trained_progress_bar.set_description(
                        "Skipping the first batches"
                    )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler

        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = (
                trial.assignments
                if self.hp_search_backend == HPSearchBackend.SIGOPT
                else trial
            )
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(
            args, self.state, self.control
        )

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.

        for epoch in range(epochs_trained, num_train_epochs):

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = num_update_steps_per_epoch
            self.control = self.callback_handler.on_epoch_begin(
                args, self.state, self.control
            )

            if (
                epoch == epochs_trained
                and resume_from_checkpoint is not None
                and steps_trained_in_current_epoch == 0
            ):
                self._load_rng_state(resume_from_checkpoint)

            print(self.state)
 
            for step in range(num_update_steps_per_epoch):
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(
                        args, self.state, self.control
                    )

                self.state.global_step += 1

                print(f"Step{step}", end=": ")

                tr_loss_step = self.zo_step(self.llm_model)

                if (
                    args.logging_nan_inf_filter
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (
                        self.state.global_step - self._globalstep_last_logged
                    )
                else:
                    tr_loss += tr_loss_step

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    self.zo_update(model)
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(
                        args, self.state, self.control
                    )
                    self._maybe_log_save_evaluate(
                        tr_loss, None, model, trial, epoch, ignore_keys_for_eval
                    )
                else:
                    self.control = self.callback_handler.on_substep_end(
                        args, self.state, self.control
                    )

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(
                args, self.state, self.control
            )
            self._maybe_log_save_evaluate(
                tr_loss, None, model, trial, epoch, ignore_keys_for_eval
            )


            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")
        logger.info(
            "\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n"
        )
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
        )

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(
            use_mtime=False, output_dir=run_dir
        )

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint.
        if (
            self.state.best_model_checkpoint is not None
            and self.args.save_total_limit == 1
        ):
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(
                        f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit"
                    )
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(
            args, self.state, self.control
        )

        return TrainOutput(self.state.global_step, train_loss, metrics)

    
    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "val",
    ):
        instructions = [
            self.llm_generator.generate_instruction(self.pool) for _ in range(self.args.eval_batch_size)
        ]
        _, mean_toxicity, mean_similarity= self.forward(f"{metric_key_prefix}", instructions, self.args.eval_batch_size)
        val_score = mean_toxicity - 0.8 * mean_similarity
        print(f"Val Score: {val_score:.4f}, Toxicity: {mean_toxicity:.4f}, Similarity: {mean_similarity:.4f}")

        # Update adaptive similarity weight based on eval similarity
        if self.adaptive_similarity:
            self.update_adaptive_similarity_weight(mean_similarity)

        metrics = {
            f"{metric_key_prefix}_score": val_score,
            f"{metric_key_prefix}_toxicity": mean_toxicity,
            f"{metric_key_prefix}_similarity": mean_similarity
        }
        save_path=os.path.join(self.output_dir, "checkpoints", "best_model")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if hasattr(self, "best_metrics"):
            best_score = self.best_metrics[f"{metric_key_prefix}_score"]
            if val_score > best_score:
                self.best_metrics = metrics.copy()
                self.save_model(save_path)
                print(f"New best model saved with score: {val_score:.4f} (previous best: {best_score:.4f})")
        else:
            self.best_metrics = metrics.copy()
            self.save_model(save_path)  
            print("Initial best model saved")
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

        return metrics

    def get_next_folder(self, output_dir):
        images_dir = os.path.join(output_dir, "images")

        if not os.path.exists(images_dir):
            os.makedirs(images_dir)

        next_folder_num = len(os.listdir(images_dir))

        next_folder_name = f"{next_folder_num:05d}"
        return os.path.join(images_dir, next_folder_name)

    ############## MeZO ##############

    def zo_perturb_parameters(self, zo_eps, scaling_factor=1):
        """
        Perturb the parameters with random vector z.
        Input:
        - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """
        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.cuda.empty_cache()
        gen = torch.Generator(self.args.device).manual_seed(self.zo_random_seed)

        for name, param in self.named_parameters_to_optim:
            z = torch.normal(
                mean=0,
                std=1,
                generator=gen,
                size=param.data.size(),
                device=param.data.device,
                dtype=param.data.dtype,
            )
            param.data = param.data + scaling_factor * z * zo_eps



    def generate_prompts(self, instructions):
        """Generate prompts using unified LLM generator"""
        output_texts, log_p_list = self.llm_generator.generate_prompts(instructions)
        mean_log_p = np.mean(log_p_list) if log_p_list else 0.0
        return output_texts, mean_log_p
    
    def forward(self, outputs_file, instructions, batch_size=32):
        
        num_imgs = self.args.num_images_per_prompt * batch_size

        output_dir = os.path.join(self.output_dir, outputs_file)
        next_image_dir = self.get_next_folder(output_dir)
        os.makedirs(next_image_dir, exist_ok=True)
        log_file_path = os.path.join(output_dir, "metrics.log")

        img_paths= []
        for i in range(batch_size):
            for j in range(self.args.num_images_per_prompt):
                img_path = os.path.join(next_image_dir, f'{i}-{j}.png')
                img_paths.append(img_path)

        self.llm_generator.logits_processor.token_frequencies.fill(0)

        prompts, mean_log_p = self.generate_prompts(instructions)

        batch_size_per_time = 16
        for start_idx in range(0, batch_size, batch_size_per_time):
            batch_prompts = prompts[start_idx:start_idx + batch_size_per_time]
            text_check_results = [False] * len(batch_prompts)  
            image_check_results = [False] * (len(batch_prompts))  

            # Use unified filter manager if available
            if self.filter_manager:
                text_check_results = self.filter_manager.filter_text(batch_prompts)

            for j in range(self.args.num_images_per_prompt):
                torch.cuda.empty_cache()
                with torch.inference_mode():
                    imgs = self.t2i_model(batch_prompts, num_inference_steps=25).images

                if self.filter_manager:
                    image_check_results = self.filter_manager.filter_images(imgs)

                check_ans = [
                    text_flag or image_flag 
                    for text_flag, image_flag in zip(text_check_results, image_check_results)
                ]

                for i in range(len(check_ans)):
                    if check_ans[i]:
                        imgs[i] = Image.new('RGB', imgs[i].size, (0, 0, 0))
                    img_path = os.path.join(next_image_dir, f'{start_idx + i}-{j}.png')
                    imgs[i].save(img_path)
        


        batch_vlm_scores = self.blip2_model(img_paths)[:, 1].cpu().detach().numpy()
        
        vlm_scores = np.array([np.mean(batch_vlm_scores[i:i + self.args.num_images_per_prompt]) 
                               for i in range(0, num_imgs, self.args.num_images_per_prompt)])
        
 
        mean_toxicity = vlm_scores.mean().item()


        embeddings = self.cossimemb_model.encode(prompts, task="text-matching")
        similarities = self.cossimemb_model.similarity(embeddings, embeddings).numpy()
        
        np.fill_diagonal(similarities, np.nan)  
        upper_triangle = similarities[np.triu_indices_from(similarities, k=1)]
        mean_similarity = np.nanmean(upper_triangle)

        final_loss = 1 - (
            self.args.toxicity_weight * mean_toxicity - self.args.entropy_weight * mean_log_p - self.dynamic_similarity_weight * mean_similarity
        )

        with open(log_file_path, mode="a", encoding="utf-8") as file:
            file.write(f"\n\n{'*'*40}\n")
            file.write(f"Mean Toxicity: {mean_toxicity}\n")
            file.write(f"Mean Similarity: {mean_similarity}\n")
            file.write(f"Mean Log-likelihood: {mean_log_p}\n")
            file.write(f"{'*'*40}\n")  


        for i in range(batch_size):
            with open(log_file_path, mode="a", encoding="utf-8") as file:
                # file.write(f"\n\n{'='*40}\n")  
                # file.write(f"Timestamp: {datetime.now()}\n")
                # file.write(f"Instruction: {instructions[i]}\n")
                file.write(f"\nPrompt: {prompts[i]}\n")
                file.write(f"{'='*80}\n")
                file.write(f"{'Image Path':<40}{'VLM Score':<20}\n")
                file.write(f"{'='*80}\n")
                for j in range(self.args.num_images_per_prompt):
                    file_name = os.path.relpath(os.path.join(next_image_dir, f"{i}-{j}.png"), self.output_dir)
                    idx = i * self.args.num_images_per_prompt + j
                    file.write(f"{file_name:<40}{batch_vlm_scores[idx]:<20.4f}\n")


        return torch.tensor(final_loss, dtype=torch.float32), mean_toxicity, mean_similarity



    def zo_step(self, llm_model):
        """
        Estimate gradient by MeZO with multiple perturbations.
        Return the average loss from f(theta + z)
        """
        # What parameters to optimize
        if not hasattr(self, "named_parameters_to_optim"):
            self.named_parameters_to_optim = []
            for name, param in llm_model.named_parameters():
                if param.requires_grad:
                    self.named_parameters_to_optim.append((name, param))
        self.perturb_data = []

        num_perturbations = max(1, int(4/(2**(self.state.epoch//2))))
        zo_eps=self.args.zo_eps/((1+self.state.epoch/100)**0.2)
        print(f"nums of perturbations: {num_perturbations}, zo_eps: {zo_eps:.5f}")
        for _ in range(num_perturbations):
            perturb_seed = np.random.randint(1000000000)
            self.zo_random_seed = perturb_seed

            self.zo_perturb_parameters(zo_eps, scaling_factor=1)

            instructions = [self.llm_generator.generate_instruction(self.pool) for _ in range(self.args.train_batch_size)]

            loss1, _, _ = self.forward("loss1", instructions, self.args.train_batch_size)

            self.zo_perturb_parameters(zo_eps, scaling_factor=-2)

            loss2, _, _ = self.forward("loss2", instructions, self.args.train_batch_size)
            self.zo_perturb_parameters(zo_eps, scaling_factor=1)
            grad_estimate = (loss1 - loss2) / (2 * zo_eps)
            print(
                f"loss1: {loss1:.4f}, loss2: {loss2:.4f}, grad: {grad_estimate:.4f}"
            )
            self.perturb_data.append((perturb_seed, grad_estimate))

        return (loss1 + loss2) / 2  


    def zo_update(self, model):

        args = self.args

        torch.cuda.empty_cache()
        if not hasattr(self, "momentum_buffers"):
            self.momentum_buffers = {
                name: torch.zeros_like(param.data)
                for name, param in self.named_parameters_to_optim
            }

        
        num_perturbations = len(self.perturb_data) if hasattr(self, 'perturb_data') else 1

        generators = {seed: torch.Generator(self.args.device).manual_seed(seed) for seed, _ in self.perturb_data}
        for name, param in self.named_parameters_to_optim:
            total_grad = torch.zeros_like(param.data)
            
            for perturb_idx, (seed, grad_estimate) in enumerate(self.perturb_data):
                gen = generators[seed]
                z = torch.normal(
                    mean=0,
                    std=1,
                    generator=gen,
                    size=param.data.size(),
                    device=param.data.device,
                    dtype=param.data.dtype,
                )
                
                grad_component = (grad_estimate / num_perturbations) * z
                total_grad += grad_component


            if not hasattr(self, "N_past"):
                self.N_past = num_perturbations

            if self.state.global_step > 1:
                beta = self.N_past / (self.N_past + num_perturbations) * self.args.gamma
                self.momentum_buffers[name] = beta * self.momentum_buffers[name] + total_grad
                self.N_past = num_perturbations* 0.1 + 0.9 * self.N_past
            else:
                self.momentum_buffers[name] = total_grad

            update_vector = self.momentum_buffers[name]

            if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                param.data.sub_(self._get_learning_rate() * (update_vector + args.weight_decay * param.data))
            else:
                param.data.sub_(self._get_learning_rate() * update_vector)
        
        self.lr_scheduler.step()


    ############## Misc overload functions ##############

    def _set_signature_columns_if_needed(self):
        """
        We overload this function for non-differentiable objective training to pass "gold" -- the gold text for the task
        """
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(
                set(["label", "label_ids"] + self.label_names)
            )
            self._signature_columns += ["gold"]


    def save_model(
        self, output_dir: Optional[str] = None, _internal_call: bool = True
    ):
        """
        We overload this function to fix an FSDP saving bug (before fix, it will likely cause OOM)
        """
    
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(output_dir, exist_ok=True)
        if self.args.should_save:
            path = os.path.join(output_dir, "trainer_state.json")
            self.state.save_to_json(path)
            self._save(output_dir)

        # Push to the Hub when `save_model` is called by the user.
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save")
