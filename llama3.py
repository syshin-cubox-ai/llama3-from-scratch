import json
import pathlib

import tiktoken.load
import torch


class Llama3:
    def __init__(self, model_path: str, device: str):
        self.model_path = pathlib.Path(model_path).resolve(strict=True)
        self.weights_path = self.model_path.joinpath("consolidated.00.pth")
        self.parameters_path = self.model_path.joinpath("params.json")
        self.tokenizer_path = self.model_path.joinpath("tokenizer.model")
        self.device = torch.device(device)

        self.model = torch.load(self.weights_path, map_location=device)
        self.params = json.loads(self.parameters_path.read_text())
        self.tokenizer = self._init_tokenizer()

    def _init_tokenizer(self) -> tiktoken.Encoding:
        special_tokens = [
                             "<|begin_of_text|>",
                             "<|end_of_text|>",
                             "<|reserved_special_token_0|>",
                             "<|reserved_special_token_1|>",
                             "<|reserved_special_token_2|>",
                             "<|reserved_special_token_3|>",
                             "<|start_header_id|>",
                             "<|end_header_id|>",
                             "<|reserved_special_token_4|>",
                             "<|eot_id|>",  # end of turn
                         ] + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)]
        mergeable_ranks = tiktoken.load.load_tiktoken_bpe(str(self.tokenizer_path))
        tokenizer = tiktoken.Encoding(
            name=self.tokenizer_path.name,
            pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            mergeable_ranks=mergeable_ranks,
            special_tokens={token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)},
        )
        return tokenizer

    def _rope_freqs_cis(self, len_tokens: int) -> torch.Tensor:
        zero_to_one_split_into_64_parts = torch.linspace(0, 1, 65, device=self.device)[:-1]
        freqs = 1 / (self.params["rope_theta"] ** zero_to_one_split_into_64_parts)
        freqs_for_each_token = torch.outer(torch.arange(len_tokens, device=self.device), freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs_for_each_token), freqs_for_each_token)
        return freqs_cis

    def _rms_norm(self, tensor: torch.Tensor, norm_weights: torch.Tensor) -> torch.Tensor:
        return (tensor * torch.rsqrt(tensor.pow(2).mean(-1, keepdim=True) + self.params["norm_eps"])) * norm_weights

    def predict(self, prompt: str) -> str:
        # Encode tokens
        tokens = torch.tensor([128000] + self.tokenizer.encode(prompt), device=self.device)
        freqs_cis = self._rope_freqs_cis(len(tokens))

        # Convert tokens to embedding
        embedding_layer = torch.nn.Embedding(self.params["vocab_size"], self.params["dim"], device=self.device)
        embedding_layer.weight.data.copy_(self.model["tok_embeddings.weight"])
        token_embeddings_unnormalized = embedding_layer(tokens).to(torch.bfloat16)

        # Process
        final_embedding = token_embeddings_unnormalized
        for layer in range(self.params["n_layers"]):
            qkv_attention_store = []
            layer_embedding_norm = self._rms_norm(final_embedding, self.model[f"layers.{layer}.attention_norm.weight"])
            q_layer = self.model[f"layers.{layer}.attention.wq.weight"]
            q_layer = q_layer.view(self.params["n_heads"], q_layer.shape[0] // self.params["n_heads"],
                                   self.params["dim"])
            k_layer = self.model[f"layers.{layer}.attention.wk.weight"]
            k_layer = k_layer.view(self.params["n_kv_heads"], k_layer.shape[0] // self.params["n_kv_heads"],
                                   self.params["dim"])
            v_layer = self.model[f"layers.{layer}.attention.wv.weight"]
            v_layer = v_layer.view(self.params["n_kv_heads"], v_layer.shape[0] // self.params["n_kv_heads"],
                                   self.params["dim"])
            for head in range(self.params["n_heads"]):
                q_layer_head = q_layer[head]
                k_layer_head = k_layer[head // 4]
                v_layer_head = v_layer[head // 4]
                q_per_token = torch.matmul(layer_embedding_norm, q_layer_head.T)
                k_per_token = torch.matmul(layer_embedding_norm, k_layer_head.T)
                v_per_token = torch.matmul(layer_embedding_norm, v_layer_head.T)
                q_per_token_split_into_pairs = q_per_token.float().view(q_per_token.shape[0], -1, 2)
                q_per_token_as_complex_numbers = torch.view_as_complex(q_per_token_split_into_pairs)
                q_per_token_split_into_pairs_rotated = torch.view_as_real(q_per_token_as_complex_numbers * freqs_cis)
                q_per_token_rotated = q_per_token_split_into_pairs_rotated.view(q_per_token.shape)
                k_per_token_split_into_pairs = k_per_token.float().view(k_per_token.shape[0], -1, 2)
                k_per_token_as_complex_numbers = torch.view_as_complex(k_per_token_split_into_pairs)
                k_per_token_split_into_pairs_rotated = torch.view_as_real(k_per_token_as_complex_numbers * freqs_cis)
                k_per_token_rotated = k_per_token_split_into_pairs_rotated.view(k_per_token.shape)
                qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T) / 128 ** 0.5
                mask = torch.full((len(token_embeddings_unnormalized), len(token_embeddings_unnormalized)),
                                  float("-inf"),
                                  device=self.device)
                mask = torch.triu(mask, diagonal=1)
                qk_per_token_after_masking = qk_per_token + mask
                qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking,
                                                                                       dim=1).to(torch.bfloat16)
                qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)
                qkv_attention_store.append(qkv_attention)

            stacked_qkv_attention = torch.cat(qkv_attention_store, dim=-1)
            w_layer = self.model[f"layers.{layer}.attention.wo.weight"]
            embedding_delta = torch.matmul(stacked_qkv_attention, w_layer.T)
            embedding_after_edit = final_embedding + embedding_delta
            embedding_after_edit_normalized = self._rms_norm(embedding_after_edit,
                                                             self.model[f"layers.{layer}.ffn_norm.weight"])
            w1 = self.model[f"layers.{layer}.feed_forward.w1.weight"]
            w2 = self.model[f"layers.{layer}.feed_forward.w2.weight"]
            w3 = self.model[f"layers.{layer}.feed_forward.w3.weight"]
            output_after_feedforward = torch.matmul(
                torch.functional.F.silu(torch.matmul(embedding_after_edit_normalized, w1.T)) * torch.matmul(
                    embedding_after_edit_normalized, w3.T), w2.T)
            final_embedding = embedding_after_edit + output_after_feedforward

        # Final linear layer
        final_embedding = self._rms_norm(final_embedding, self.model["norm.weight"])
        logits = torch.matmul(final_embedding[-1], self.model["output.weight"].T)
        next_token = torch.argmax(logits)

        # Decode embedding into token
        output = self.tokenizer.decode([next_token.item()])
        return output


def main():
    prompt = "the answer to the ultimate question of life, the universe, and everything is "

    llama3 = Llama3("../../data/Meta-Llama-3-8B", "cuda")
    output = llama3.predict(prompt)
    print(f"{output=}")

    while prompt := input("Enter prompt >>> "):
        output = llama3.predict(prompt)
        print(f"{output=}")


if __name__ == "__main__":
    main()
