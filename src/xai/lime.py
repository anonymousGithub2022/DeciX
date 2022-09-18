import torch
import torch.nn.functional as F
from captum._utils.models.linear_model import SkLearnLinearRegression, SkLearnLasso
import inspect
import math
import typing
import warnings
from typing import Any, Callable, List, Optional, Tuple, Union, cast

import torch
from captum._utils.common import (
    _expand_additional_forward_args,
    _expand_target,
    _flatten_tensor_or_tuple,
    _format_input,
    _format_output,
    _is_tuple,
    _reduce_list,
    _run_forward,
)
from captum._utils.models.linear_model import SkLearnLasso
from captum._utils.models.model import Model
from captum._utils.progress import progress
from captum._utils.typing import (
    BaselineType,
    Literal,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)
from captum.attr._utils.attribution import PerturbationAttribution
from captum.attr._utils.batching import _batch_example_iterator
from captum.attr._utils.common import (
    _construct_default_feature_mask,
    _format_input_baseline,
)
from captum.log import log_usage
from torch import Tensor
from torch.nn import CosineSimilarity
from torch.utils.data import DataLoader, TensorDataset


from captum.attr import (
    Lime,
    LimeBase
)

from .baseExp import BaseExpClass


class LimeExp(BaseExpClass, LimeBase):
    # encode text indices into latent representations & calculate cosine similarity
    def exp_embedding_cosine_distance(self, original_inp, perturbed_inp, _, **kwargs):
        original_emb = self.model.get_embedding(original_inp[0], original_inp[1])
        perturbed_emb = self.model.get_embedding(perturbed_inp[0], perturbed_inp[1])
        distance = F.cosine_similarity(original_emb, perturbed_emb, dim=-1)
        return distance.mean(1)
        # return torch.exp(-1 * ((1 - distance) ** 2) / 2)

    # binary vector where each word is selected independently and uniformly at random
    # def bernoulli_perturb(self, x, **kwargs):
    #     ori_inputs, ori_len = x
    #     batch_size, input_length = ori_inputs.shape
    #
    #     mutated_x = ori_inputs.clone()
    #     mutated_len = ori_len.clone()
    #     random_x = (torch.rand([1, input_length]) * self.vocab_size).int().to(self.device)
    #     mask = (torch.rand([1, input_length], device=self.device) < self.mutate_rate).float()
    #     mask[:, 0] = 1
    #     mask[:, ori_len + 1:] = 1
    #     mutated_x = random_x * (1 - mask) + mutated_x * mask
    #     mutated_x = mutated_x.to(torch.int64).to(self.device)
    #     return mutated_x, mutated_len

    # remove absenst token based on the intepretable representation sample
    @staticmethod
    def interp_to_input(new_x, x, **kwargs):
        ori_inputs, ori_len = x
        interp_sample, new_len = new_x
        # return ori_inputs[interp_sample.bool()].view(ori_inputs.size(0), -1)
        return new_x

    def __init__(self, model, config, device):
        BaseExpClass.__init__(self, model, config, device)
        LimeBase.__init__(
            self,
            self.model,
            interpretable_model=SkLearnLasso(alpha=0.08),
            similarity_func=self.exp_embedding_cosine_distance,
            perturb_func=self.model.bernoulli_perturb,
            perturb_interpretable_space=True,
            from_interp_rep_transform=self.interp_to_input,
            to_interp_rep_transform=None
        )

    def explain(self, x):
        # mutated_res = self.bernoulli_perturb(x)
        # (mutated_x, mutated_len, mask, ori_input_seqs, ori_input_len) = mutated_res
        x = x[0].to(self.device), x[1].to(self.device)
        src_tk, src_len = x
        out_seqs, out_len = self.predict_output_seqs(x[0], x[1])
        # index = torch.ones([1], device=self.device) * i
        weights = self.attribute([x[0], x[1]])
        # weights.append(exp.detach().cpu().numpy()[:, :ori_len])
        if self.task == 'DeepAPI':
            weights = weights[:out_len[0], :src_len + 2].detach().cpu().numpy()
        else:
            weights = weights[:out_len[0], :src_len].detach().cpu().numpy()
        weights = [d.reshape([1, -1]) for d in weights]
        return weights, out_seqs[0], out_len[0]

    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType = None,
        additional_forward_args: Any = None,
        n_samples: int = 50,
        perturbations_per_eval: int = 1,
        show_progress: bool = False,
        **kwargs,
    ) -> Tensor:
        with torch.no_grad():
            inp_tensor = (
                cast(Tensor, inputs) if isinstance(inputs, Tensor) else inputs[0]
            )
            device = self.device

            interpretable_inps = []
            similarities = []
            outputs = []

            curr_model_inputs = []
            expanded_additional_args = None
            expanded_target = None
            perturb_generator = None
            if inspect.isgeneratorfunction(self.perturb_func):
                perturb_generator = self.perturb_func(inputs, **kwargs)

            if show_progress:
                attr_progress = progress(
                    total=math.ceil(n_samples / perturbations_per_eval),
                    desc=f"{self.get_name()} attribution",
                )
                attr_progress.update(0)

            batch_count = 0
            for _ in range(n_samples):
                if perturb_generator:
                    try:
                        curr_sample = next(perturb_generator)
                    except StopIteration:
                        warnings.warn(
                            "Generator completed prior to given n_samples iterations!"
                        )
                        break
                else:
                    curr_sample = self.perturb_func(inputs, **kwargs)
                batch_count += 1
                if self.perturb_interpretable_space:
                    interpretable_inps.append(curr_sample)
                    curr_model_inputs.append(
                        self.from_interp_rep_transform(  # type: ignore
                            curr_sample, inputs, **kwargs
                        )
                    )
                else:
                    curr_model_inputs.append(curr_sample)
                    interpretable_inps.append(
                        self.to_interp_rep_transform(  # type: ignore
                            curr_sample, inputs, **kwargs
                        )
                    )
                curr_sim = self.similarity_func(
                    inputs, curr_model_inputs[-1], interpretable_inps[-1], **kwargs
                )
                similarities.append(
                    curr_sim.flatten()
                    if isinstance(curr_sim, Tensor)
                    else torch.tensor([curr_sim], device=device)
                )

                if len(curr_model_inputs) == perturbations_per_eval:
                    if expanded_additional_args is None:
                        expanded_additional_args = _expand_additional_forward_args(
                            additional_forward_args, len(curr_model_inputs)
                        )
                    if expanded_target is None:
                        expanded_target = _expand_target(target, len(curr_model_inputs))

                    model_out = self._evaluate_batch(
                        curr_model_inputs,
                        expanded_target,
                        expanded_additional_args,
                        device,
                    )

                    if show_progress:
                        attr_progress.update()

                    outputs.append(model_out)

                    curr_model_inputs = []

            if len(curr_model_inputs) > 0:
                expanded_additional_args = _expand_additional_forward_args(
                    additional_forward_args, len(curr_model_inputs)
                )
                expanded_target = _expand_target(target, len(curr_model_inputs))
                model_out = self._evaluate_batch(
                    curr_model_inputs,
                    expanded_target,
                    expanded_additional_args,
                    device,
                )
                if show_progress:
                    attr_progress.update()
                outputs.append(model_out)

            if show_progress:
                attr_progress.close()

            # combined_interp_inps = torch.cat(interpretable_inps).double()
            x = [d[0] for d in interpretable_inps]
            combined_interp_inps = torch.cat(x).double()
            # out_len = min([len(d[0]) for d in outputs])
            # outputs = [d[:, :out_len] for d in outputs]
            combined_outputs = (
                torch.cat(outputs)
                if len(outputs[0].shape) > 0
                else torch.stack(outputs)
            ).double()
            combined_sim = (
                torch.cat(similarities)
                if len(similarities[0].shape) > 0
                else torch.stack(similarities)
            ).double()
            dataset = TensorDataset(
                combined_interp_inps, combined_outputs, combined_sim
            )
            self.interpretable_model.fit(DataLoader(dataset, batch_size=batch_count))
            return self.interpretable_model.representation()

    def _evaluate_batch(
        self,
        curr_model_inputs: List[TensorOrTupleOfTensorsGeneric],
        expanded_target: TargetType,
        expanded_additional_args: Any,
        device: torch.device,
    ):
        model_out = _run_forward(
            self.forward_func,
            _reduce_list(curr_model_inputs),
            expanded_target,
            expanded_additional_args,
        )
        model_out = model_out.max(-1)[1]
        model_out = model_out[:, :self.max_length]
        return model_out
        # if isinstance(model_out, Tensor):
        #     assert model_out.numel() == len(curr_model_inputs), (
        #         "Number of outputs is not appropriate, must return "
        #         "one output per perturbed input"
        #     )
        # if isinstance(model_out, Tensor):
        #     return model_out.flatten()
        # return torch.tensor([model_out], device=device)

