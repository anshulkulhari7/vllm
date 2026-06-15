# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.model_executor.models.qwen2_audio import (
    Qwen2AudioForConditionalGeneration,
    _get_feat_extract_output_lengths,
)


def _make_qwen2_audio_model() -> Qwen2AudioForConditionalGeneration:
    # The token-count helpers read no instance state, so __new__ is enough to
    # exercise them without GPU / model weights.
    return Qwen2AudioForConditionalGeneration.__new__(
        Qwen2AudioForConditionalGeneration
    )


@pytest.mark.parametrize("num_audio_tokens", [1, 50, 375, 1500])
def test_qwen2_audio_tower_connector_token_counts(num_audio_tokens: int) -> None:
    # Qwen2-Audio is a 1:1 model: the Whisper-style audio tower's transformer
    # layers run on the conv-downsampled sequence (length == number of audio
    # placeholder tokens) and the projector is a single Linear, so both the
    # tower and connector LoRA mappings cover exactly `num_audio_tokens` tokens
    # (see gpu_model_runner build of the tower/connector LoRA mappings).
    model = _make_qwen2_audio_model()

    assert model.get_num_mm_encoder_tokens(num_audio_tokens) == num_audio_tokens
    assert model.get_num_mm_connector_tokens(num_audio_tokens) == num_audio_tokens

    # Round-trip: connector(tower(n)) == n.
    num_encoder_tokens = model.get_num_mm_encoder_tokens(num_audio_tokens)
    assert model.get_num_mm_connector_tokens(num_encoder_tokens) == num_audio_tokens


@pytest.mark.parametrize("mel_frames", [160, 1500, 3000])
def test_qwen2_audio_encoder_tokens_match_feat_extract_output(
    mel_frames: int,
) -> None:
    # The number of audio placeholder tokens the language model sees equals the
    # audio encoder output length. The tower helper must report exactly that
    # many encoder tokens (1:1) so the LoRA tower mapping covers every token.
    input_lengths = torch.tensor([mel_frames], dtype=torch.long)
    _, output_lengths = _get_feat_extract_output_lengths(input_lengths)
    num_audio_tokens = int(output_lengths.item())

    model = _make_qwen2_audio_model()

    assert model.get_num_mm_encoder_tokens(num_audio_tokens) == num_audio_tokens
    assert model.get_num_mm_connector_tokens(num_audio_tokens) == num_audio_tokens
