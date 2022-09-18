import torch
from typing import *

from .baseExp import BaseExpClass
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
from captum.attr import (
    ShapleyValues,
    ShapleyValueSampling,
)
import itertools
import math
import warnings
import torch
from captum._utils.common import (
    _expand_additional_forward_args,
    _expand_target,
    _format_additional_forward_args,
    _format_input,
    _format_output,
    _is_tuple,
    _run_forward,
)
from captum._utils.progress import progress
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr._utils.attribution import PerturbationAttribution
from captum.attr._utils.common import (
    _construct_default_feature_mask,
    _find_output_mode_and_verify,
    _format_input_baseline,
    _tensorize_baseline,
)

class SHAPExp(BaseExpClass, ShapleyValueSampling):
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
        ShapleyValueSampling.__init__(self, self.model)

    def explain(self, x):
        # mutated_res = self.bernoulli_perturb(x)
        # (mutated_x, mutated_len, mask, ori_input_seqs, ori_input_len) = mutated_res
        x = x[0].to(self.device), x[1].to(self.device)
        src_tk, src_len = x
        out_seqs, out_len = self.predict_output_seqs(x[0], x[1])
        # index = torch.ones([1], device=self.device) * i
        weights = self.attribute(inputs=(x[0], x[1]))
        # weights.append(exp.detach().cpu().numpy()[:, :ori_len])
        weights = weights[:out_len[0], :src_len+1].detach().cpu().numpy()
        weights = [d.reshape([1, -1]) for d in weights]
        return weights, out_seqs[0], out_len[0]

    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        feature_mask: Union[None, TensorOrTupleOfTensorsGeneric] = None,
        n_samples: int = 25,
        perturbations_per_eval: int = 1,
    ) -> TensorOrTupleOfTensorsGeneric:
        r"""
        NOTE: The feature_mask argument differs from other perturbation based
        methods, since feature indices can overlap across tensors. See the
        description of the feature_mask argument below for more details.

        Args:

                inputs (tensor or tuple of tensors):  Input for which Shapley value
                            sampling attributions are computed. If forward_func takes
                            a single tensor as input, a single input tensor should
                            be provided.
                            If forward_func takes multiple tensors as input, a tuple
                            of the input tensors should be provided. It is assumed
                            that for all given input tensors, dimension 0 corresponds
                            to the number of examples (aka batch size), and if
                            multiple input tensors are provided, the examples must
                            be aligned appropriately.
                baselines (scalar, tensor, tuple of scalars or tensors, optional):
                            Baselines define reference value which replaces each
                            feature when ablated.
                            Baselines can be provided as:

                            - a single tensor, if inputs is a single tensor, with
                              exactly the same dimensions as inputs or the first
                              dimension is one and the remaining dimensions match
                              with inputs.

                            - a single scalar, if inputs is a single tensor, which will
                              be broadcasted for each input value in input tensor.

                            - a tuple of tensors or scalars, the baseline corresponding
                              to each tensor in the inputs' tuple can be:

                              - either a tensor with matching dimensions to
                                corresponding tensor in the inputs' tuple
                                or the first dimension is one and the remaining
                                dimensions match with the corresponding
                                input tensor.

                              - or a scalar, corresponding to a tensor in the
                                inputs' tuple. This scalar value is broadcasted
                                for corresponding input tensor.
                            In the cases when `baselines` is not provided, we internally
                            use zero scalar corresponding to each input tensor.
                            Default: None
                target (int, tuple, tensor or list, optional):  Output indices for
                            which difference is computed (for classification cases,
                            this is usually the target class).
                            If the network returns a scalar value per example,
                            no target index is necessary.
                            For general 2D outputs, targets can be either:

                            - a single integer or a tensor containing a single
                              integer, which is applied to all input examples

                            - a list of integers or a 1D tensor, with length matching
                              the number of examples in inputs (dim 0). Each integer
                              is applied as the target for the corresponding example.

                            For outputs with > 2 dimensions, targets can be either:

                            - A single tuple, which contains #output_dims - 1
                              elements. This target index is applied to all examples.

                            - A list of tuples with length equal to the number of
                              examples in inputs (dim 0), and each tuple containing
                              #output_dims - 1 elements. Each tuple is applied as the
                              target for the corresponding example.

                            Default: None
                additional_forward_args (any, optional): If the forward function
                            requires additional arguments other than the inputs for
                            which attributions should not be computed, this argument
                            can be provided. It must be either a single additional
                            argument of a Tensor or arbitrary (non-tuple) type or a
                            tuple containing multiple additional arguments including
                            tensors or any arbitrary python types. These arguments
                            are provided to forward_func in order following the
                            arguments in inputs.
                            For a tensor, the first dimension of the tensor must
                            correspond to the number of examples. For all other types,
                            the given argument is used for all forward evaluations.
                            Note that attributions are not computed with respect
                            to these arguments.
                            Default: None
                feature_mask (tensor or tuple of tensors, optional):
                            feature_mask defines a mask for the input, grouping
                            features which should be added together. feature_mask
                            should contain the same number of tensors as inputs.
                            Each tensor should
                            be the same size as the corresponding input or
                            broadcastable to match the input tensor. Values across
                            all tensors should be integers in the range 0 to
                            num_features - 1, and indices corresponding to the same
                            feature should have the same value.
                            Note that features are grouped across tensors
                            (unlike feature ablation and occlusion), so
                            if the same index is used in different tensors, those
                            features are still grouped and added simultaneously.
                            If the forward function returns a single scalar per batch,
                            we enforce that the first dimension of each mask must be 1,
                            since attributions are returned batch-wise rather than per
                            example, so the attributions must correspond to the
                            same features (indices) in each input example.
                            If None, then a feature mask is constructed which assigns
                            each scalar within a tensor as a separate feature
                            Default: None
                n_samples (int, optional):  The number of feature permutations
                            tested.
                            Default: `25` if `n_samples` is not provided.
                perturbations_per_eval (int, optional): Allows multiple ablations
                            to be processed simultaneously in one call to forward_fn.
                            Each forward pass will contain a maximum of
                            perturbations_per_eval * #examples samples.
                            For DataParallel models, each batch is split among the
                            available devices, so evaluations on each available
                            device contain at most
                            (perturbations_per_eval * #examples) / num_devices
                            samples.
                            If the forward function returns a single scalar per batch,
                            perturbations_per_eval must be set to 1.
                            Default: 1
                show_progress (bool, optional): Displays the progress of computation.
                            It will try to use tqdm if available for advanced features
                            (e.g. time estimation). Otherwise, it will fallback to
                            a simple output of progress.
                            Default: False

        Returns:
                *tensor* or tuple of *tensors* of **attributions**:
                - **attributions** (*tensor* or tuple of *tensors*):
                            The attributions with respect to each input feature.
                            If the forward function returns
                            a scalar value per example, attributions will be
                            the same size as the provided inputs, with each value
                            providing the attribution of the corresponding input index.
                            If the forward function returns a scalar per batch, then
                            attribution tensor(s) will have first dimension 1 and
                            the remaining dimensions will match the input.
                            If a single tensor is provided as inputs, a single tensor is
                            returned. If a tuple is provided for inputs, a tuple of
                            corresponding sized tensors is returned.


        Examples::

            >>> # SimpleClassifier takes a single input tensor of size Nx4x4,
            >>> # and returns an Nx3 tensor of class probabilities.
            >>> net = SimpleClassifier()
            >>> # Generating random input with size 2 x 4 x 4
            >>> input = torch.randn(2, 4, 4)
            >>> # Defining ShapleyValueSampling interpreter
            >>> svs = ShapleyValueSampling(net)
            >>> # Computes attribution, taking random orderings
            >>> # of the 16 features and computing the output change when adding
            >>> # each feature. We average over 200 trials (random permutations).
            >>> attr = svs.attribute(input, target=1, n_samples=200)

            >>> # Alternatively, we may want to add features in groups, e.g.
            >>> # grouping each 2x2 square of the inputs and adding them together.
            >>> # This can be done by creating a feature mask as follows, which
            >>> # defines the feature groups, e.g.:
            >>> # +---+---+---+---+
            >>> # | 0 | 0 | 1 | 1 |
            >>> # +---+---+---+---+
            >>> # | 0 | 0 | 1 | 1 |
            >>> # +---+---+---+---+
            >>> # | 2 | 2 | 3 | 3 |
            >>> # +---+---+---+---+
            >>> # | 2 | 2 | 3 | 3 |
            >>> # +---+---+---+---+
            >>> # With this mask, all inputs with the same value are added
            >>> # together, and the attribution for each input in the same
            >>> # group (0, 1, 2, and 3) per example are the same.
            >>> # The attributions can be calculated as follows:
            >>> # feature mask has dimensions 1 x 4 x 4
            >>> feature_mask = torch.tensor([[[0,0,1,1],[0,0,1,1],
            >>>                             [2,2,3,3],[2,2,3,3]]])
            >>> attr = svs.attribute(input, target=1, feature_mask=feature_mask)
        """
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = _is_tuple(inputs)
        baselines = self.model.format_input_baseline(inputs, self.config['UNK_ID'])
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        feature_mask = _format_input(feature_mask) if feature_mask is not None else None
        assert (
            isinstance(perturbations_per_eval, int) and perturbations_per_eval >= 1
        ), "Ablations per evaluation must be at least 1."

        with torch.no_grad():
            baselines = _tensorize_baseline(inputs, baselines)
            num_examples = inputs[0].shape[0]

            if feature_mask is None:
                feature_mask, total_features = _construct_default_feature_mask(inputs)
            else:
                total_features = int(
                    max(torch.max(single_mask).item() for single_mask in feature_mask)
                    + 1
                )

            initial_eval = _run_forward(
                self.forward_func, baselines, target, additional_forward_args
            )

            agg_output_mode = _find_output_mode_and_verify(
                initial_eval, num_examples, perturbations_per_eval, feature_mask
            )

            # Initialize attribution totals and counts
            total_attrib = [
                torch.zeros_like(
                    input[0:1] if agg_output_mode else input, dtype=torch.float
                )
                for input in inputs
            ]

            iter_count = 0
            # Iterate for number of samples, generate a permutation of the features
            # and evalute the incremental increase for each feature.
            for feature_permutation in self.permutation_generator(
                total_features, n_samples
            ):
                iter_count += 1
                prev_results = initial_eval
                for (
                    current_inputs,
                    current_add_args,
                    current_target,
                    current_masks,
                ) in self._perturbation_generator(
                    inputs,
                    additional_forward_args,
                    target,
                    baselines,
                    feature_mask,
                    feature_permutation,
                    perturbations_per_eval,
                ):
                    if sum(torch.sum(mask).item() for mask in current_masks) == 0:
                        warnings.warn(
                            "Feature mask is missing some integers between 0 and "
                            "num_features, for optimal performance, make sure each"
                            " consecutive integer corresponds to a feature."
                        )
                    # modified_eval dimensions: 1D tensor with length
                    # equal to #num_examples * #features in batch
                    modified_eval = _run_forward(
                        self.forward_func,
                        current_inputs,
                        current_target,
                        current_add_args,
                    )

                    if agg_output_mode:
                        eval_diff = modified_eval - prev_results
                        prev_results = modified_eval
                    else:
                        all_eval = torch.cat((prev_results, modified_eval), dim=0)
                        eval_diff = all_eval[num_examples:] - all_eval[:-num_examples]
                        prev_results = all_eval[-num_examples:]
                    for j in range(len(total_attrib)):
                        current_eval_diff = eval_diff
                        if not agg_output_mode:
                            # current_eval_diff dimensions:
                            # (#features in batch, #num_examples, 1,.. 1)
                            # (contains 1 more dimension than inputs). This adds extra
                            # dimensions of 1 to make the tensor broadcastable with the
                            # inputs tensor.
                            current_eval_diff = current_eval_diff.reshape(
                                (-1, num_examples) + (len(inputs[j].shape) - 1) * (1,)
                            )
                        total_attrib[j] += (
                            current_eval_diff * current_masks[j].float()
                        ).sum(dim=0)


            # Divide total attributions by number of random permutations and return
            # formatted attributions.
            attrib = tuple(
                tensor_attrib_total / iter_count for tensor_attrib_total in total_attrib
            )
            formatted_attr = _format_output(is_inputs_tuple, attrib)
        return formatted_attr


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
