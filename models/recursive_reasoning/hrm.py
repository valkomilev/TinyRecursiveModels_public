from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import wandb
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
from torch.distributions import Gamma
from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding

@dataclass
class HierarchicalReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class HierarchicalReasoningModel_ACTV1Carry:
    inner_carry: HierarchicalReasoningModel_ACTV1InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]

IGNORE_LABEL_ID = -100
BATCH_SIZE = 768
HIDDEN_DIM = 4

class IMLESubsetKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, k, tau, lambda_val, use_gumbel_k=True):
        device = logits.device
        shape = logits.shape
        
        # 1. Sample Noise (Forward Sample)
        if use_gumbel_k:
            # Replicate the Sum-of-Gamma approximation for k-subsets
            noise = torch.zeros_like(logits)
            concentration = torch.tensor(1.0 / k, device=device)
            
            # Loop matches tf.map_fn over tf.constant([1.0, ..., 10.0])
            for t in range(1, 11):
                rate = torch.tensor(k / float(t), device=device)
                # Sample and add to accumulated noise
                noise += Gamma(concentration, rate).sample(shape)
                
            # the log(m) term offset
            noise = noise - torch.log(torch.tensor(10.0, device=device))
            # divide by k
            noise = tau * (noise / k)
        else:
            # Standard Gumbel(0, 1) fallback via inverse transform sampling
            noise = -torch.empty_like(logits).exponential_().log()
            
        # 2. Add noise and compute top-k MAP threshold
        noisy_logits = logits + noise
        
        # Get the k-th largest value along the last dimension to act as the threshold
        # dim=-1 ensures it works cleanly on your (B, N, D) carry tensors
        threshold = torch.topk(noisy_logits, k, dim=-1).values[..., -1:]
        
        # 3. Create hard mask (1.0 if in top-k, 0.0 otherwise)
        y_map = (noisy_logits >= threshold).float()
        
        # Save tensors for the backward pass
        ctx.save_for_backward(logits, noise, y_map)
        # Non-tensor variables must be attached directly to ctx
        ctx.lambda_val = lambda_val
        ctx.k = k
        
        return y_map

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors and variables
        logits, noise, y_map = ctx.saved_tensors
        lambda_val = ctx.lambda_val
        k = ctx.k
        
        # 1. Perturb the logits using the incoming gradient (dy)
        target_logits = logits - (lambda_val * grad_output) + noise
        
        # 2. MAP on the target distribution (find new threshold)
        threshold = torch.topk(target_logits, k, dim=-1).values[..., -1:]
        map_y = (target_logits >= threshold).float()
        
        # 3. Compute the gradient of the conditional log-likelihood (IMLE formula)
        grad_logits = (1.0 / lambda_val) * (y_map - map_y)
        
        # Must return exactly one gradient for every input to forward()
        # Since k, tau, lambda_val, and use_gumbel_k are scalars/booleans, they get None
        return grad_logits, None, None, None, None


class IMLESubsetKLayer(nn.Module):
    def __init__(self, k=8, tau=10.0, lambda_val=10.0, use_gumbel_k=True):
        super(IMLESubsetKLayer, self).__init__()
        self.k = k
        self.tau = tau
        self.lambda_val = lambda_val
        self.use_gumbel_k = use_gumbel_k
        
    def forward(self, logits):
        """
        Expects `logits` tensor of shape (B, N, D) or similar.
        The top-k operation is automatically applied across the last dimension (dim=-1).
        """
        # Apply the custom autograd function
        return IMLESubsetKFunction.apply(
            logits, 
            self.k, 
            self.tau, 
            self.lambda_val, 
            self.use_gumbel_k
        )
class HierarchicalReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    # Alexia: added
    mlp_t: bool=False # use mlp on L instead of transformer
# def my_symbolic_rule(pixel_grid_tensor):
#     # This receives raw integer tensors of shape (B, N)
#     grid = pixel_grid_tensor.clone()
                       
#     # Determine the number of rows
#     num_rows = grid.shape[0]
#     for i in range(num_rows):
#         # Extract the row as a 1D tensor without converting to a Python list
#         row = grid[i] 
                        
#         # Use PyTorch native operations to find what numbers are present
#         # We convert it to a standard python set for lightning-fast 'in' checks
#         # without triggering PyTorch computation graphs
#         numbers_present = set(row.tolist()) 
                        
#         counter = 0
#         for j in range(10):
#             counter += 1
#             if j in numbers_present:
#                 continue
#             if counter >= 10:
#                 break
#              # In-place modification of the tensor
#             grid[i][counter] = j
                           
#     return grid

def my_symbolic_rule_1(pixel_grid_tensor):
    local_grid = pixel_grid_tensor.clone()
    local_grid = (local_grid**2)
    local_grid = torch.clamp(local_grid, min=0.0, max=9.0)
    return local_grid.round().long()

def my_symbolic_rule_2(pixel_grid_tensor):
    local_grid = pixel_grid_tensor.clone()
    local_grid = (local_grid)/5
    local_grid = torch.clamp(local_grid, min=0.0, max=9.0)
    return local_grid.round().long()
class SymbolicReadWriteBridge(nn.Module):

    def __init__(self, hidden_dim: int, vocab_size: int = 10,symbolic_func=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.imle = IMLESubsetKLayer(tau=1.0)
        self.batched_func = torch.vmap(symbolic_func)

        # 1. READ HEAD: Projects D-dimensional vector to discrete pixel logits
        self.read_head = nn.Linear(hidden_dim, vocab_size)

        # 3. WRITE HEAD: Maps discrete symbolic results back to D-dimensional vectors
        self.write_head = nn.Embedding(vocab_size, hidden_dim)

        # Optional: A learned gate to let the network decide how strongly
        # to apply the symbolic result (0.0 = ignore symbol, 1.0 = full inject)
        self.gate = nn.Parameter(torch.tensor([0.1]))
       

    def forward(self, carry: torch.Tensor,symbolic_func):
        """carry shape: (B, N, D)

        symbolic_func: A function taking a discrete tensor of shape (B, N)
        and returning a modified discrete tensor of shape (B, N).
        """
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            B, N, D = carry.shape

            # --- STEP 1: READ ---
            # Get logits over the pixel vocabulary (e.g., 10 ARC colors)
            logits = self.read_head(carry)  # Shape: (B, N, vocab_size)
            y_max = self.imle(logits)
            # Soft probabilities for the backward pass
            #soft_probs = F.softmax(logits, dim=-1)

            # Hard discrete tokens for the symbolic engine (forward pass)
            hard_tokens = torch.argmax(y_max, dim=-1)  # Shape: (B, N)

            # --- STEP 2: SYMBOLIC EXECUTION ---
            # Execute your non-differentiable Python/C++ rule engine on discrete integers
            with torch.no_grad():
                modified_tokens = self.batched_func(hard_tokens)  # Shape: (B, N)
                # Ensure output is clamped to valid vocab indices
                modified_tokens = torch.clamp(
                    modified_tokens, 0, self.vocab_size - 1
                )

            # --- STEP 3: WRITE & STRAIGHT-THROUGH ESTIMATION (STE) ---
            # Look up the D-dimensional embeddings for the new symbolic tokens
            target_embeddings = self.write_head(
                modified_tokens
            )  # Shape: (B, N, D)

            # STE Trick: We want the forward pass to use the exact `target_embeddings`,
            # but we want gradients to flow back into `read_head` via `soft_probs`.
            # We project soft_probs through the write_head's weight matrix differentiable-ly:
            soft_embeddings = torch.matmul(y_max, self.write_head.weight)

            # # Combine hard forward + soft backward
            symbolic_vector = (
                 target_embeddings - soft_embeddings.detach()
            ) + soft_embeddings
            #IMLE estimation
            #my experimental location 

            # --- STEP 4: RESIDUAL INJECTION ---
            # Add the symbolic vector back into the carry tensor without overwriting relational memory
            #updated_carry = carry + (self.gate * symbolic_vector)
            updated_carry = symbolic_vector
            return updated_carry
        
class Circuit(nn.Module): 
        def __init__(self):
            super().__init__()
            # self.input_weight_1 = torch.nn.Linear(64*BATCH_SIZE,64*BATCH_SIZE)
            # self.input_weight_2 = torch.nn.Linear(64*BATCH_SIZE,64*BATCH_SIZE)
            # self.input_weight_3 = torch.nn.Linear(64*BATCH_SIZE,64*BATCH_SIZE)
            self.softmax_1 = torch.nn.Softmax()
            self.softmax_2 = torch.nn.Softmax()
            self.softmax_3 = torch.nn.Softmax()
            self.gate = nn.Linear(65*HIDDEN_DIM, 1)

            self.bridge_1 = SymbolicReadWriteBridge(hidden_dim=HIDDEN_DIM, vocab_size=10,symbolic_func=my_symbolic_rule_1)
            self.bridge_2 = SymbolicReadWriteBridge(hidden_dim=HIDDEN_DIM, vocab_size=10,symbolic_func=my_symbolic_rule_2)
            
            #self.noise_scheduler = ScheduledLatentNoise(warmup_steps=15000)
            self.mode = 1
            self.enable_noise = False
            if self.mode == 0:
                self.multi_func = torch.nn.MultiheadAttention(embed_dim=5120,num_heads=8,batch_first=True)
            elif self.mode == 1:
                self.input_weight_1 = torch.nn.Linear(65*HIDDEN_DIM,65*HIDDEN_DIM)
                
                self.input_weight_2 = torch.nn.Linear(65*HIDDEN_DIM,65*HIDDEN_DIM)
                self.input_weight_3 = torch.nn.Linear(65*HIDDEN_DIM,65*HIDDEN_DIM)
                self.input_weight_null = torch.nn.Linear(65*HIDDEN_DIM,65*HIDDEN_DIM)
            elif self.mode == 2:
                self.cat_all = torch.nn.Linear(64*BATCH_SIZE*2,64*BATCH_SIZE)

        def soduko_solver_v1(self, memory_vb_symbolic):
                    
                        
                    # Determine the number of rows
                    num_rows = memory_vb_symbolic.shape[0]
                    for i in range(num_rows):
                        # Extract the row as a 1D tensor without converting to a Python list
                        row = memory_vb_symbolic[i] 
                        
                        # Use PyTorch native operations to find what numbers are present
                        # We convert it to a standard python set for lightning-fast 'in' checks
                        # without triggering PyTorch computation graphs
                        numbers_present = set(row.tolist()) 
                        
                        counter = 0
                        for j in range(10):
                            counter += 1
                            if j in numbers_present:
                                continue
                            if counter >= 10:
                                break
                            
                            # In-place modification of the tensor
                            memory_vb_symbolic[i][counter] = j
                            
                    return memory_vb_symbolic
        def soduko_solver_v1_no_loop(self, memory_vb_symbolic):
            
        
                # Extract the row as a 1D tensor without converting to a Python list
                row = memory_vb_symbolic[1] 
                
                # Use PyTorch native operations to find what numbers are present
                # We convert it to a standard python set for lightning-fast 'in' checks
                # without triggering PyTorch computation graphs
                numbers_present = row.tolist() 
                
                counter = 0
                for j in range(10):
                    counter += 1
                    if j in numbers_present:
                        continue
                    if counter >= 10:
                        break
                    
                    # In-place modification of the tensor
                    memory_vb_symbolic[1][counter] = j
                    
                return memory_vb_symbolic
        def аddd(self, memory_vb_symbolic):
                # Extract the row as a 1D tensor without converting to a Python list
                memory_vb_symbolic[2] +=1
                return memory_vb_symbolic
        def sub(self, memory_vb_symbolic):
                # Extract the row as a 1D tensor without converting to a Python list
                memory_vb_symbolic[3] -=1
                return memory_vb_symbolic
        def forward(self, input_vb,tau=1.0):
         with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            shape_1 = input_vb.shape[0]
            shape_2 = input_vb.shape[1]
            puzzle_embs = input_vb
       
            if self.mode == 0:
                input_vb_reshaped = torch.reshape(torch.clone(puzzle_embs), ( -1,BATCH_SIZE))

                solver_result_1 = self.soduko_solver_v1(input_vb_reshaped)
                #solver_result_2 = self.аddd(input_vb_reshaped)
                #solver_result_3 = self.sub(input_vb_reshaped)
                #if self.enable_noise:
                    #solver_result_1 = self.noise_scheduler(solver_result_1)
                    #solver_result_2 = self.noise_scheduler(solver_result_2)
                    #solver_result_3 = self.noise_scheduler(solver_result_3)
                M_tensor = torch.stack([solver_result_1], dim=1)#solver_result_2,solver_result_3
                query = input_vb_reshaped.unsqueeze(1)
                attn_output, attn_output_weights = self.multi_func(query, M_tensor,M_tensor)
                attn_output = attn_output.squeeze(1)
                attn_output = attn_output.reshape(shape_1,shape_2, -1)
                return attn_output
            elif self.mode == 1:

                #solver_result_2 = self.аddd(input_vb_reshaped)
                #solver_result_3 = self.sub(input_vb_reshaped)
                # x = torch.stack([self.input_weight_1(torch.reshape(input_vb, ( BATCH_SIZE,260))),
                #                        self.input_weight_2(torch.reshape(input_vb, ( BATCH_SIZE,260))),
                #                        self.input_weight_3(torch.reshape(input_vb, ( BATCH_SIZE,260)))], dim=0)  # the tensors you're choosing between

                solver_result_1 = self.bridge_1(input_vb, my_symbolic_rule_1)
                solver_result_2 = self.bridge_2(input_vb, my_symbolic_rule_2)
                solver_result_null = torch.zeros_like(input_vb)
                x = torch.stack([solver_result_1.view(768, 65*HIDDEN_DIM) ,
                                       solver_result_2.view(768, 65*HIDDEN_DIM) ,
                                       solver_result_null.view(768, 65*HIDDEN_DIM) ], dim=0)

                scores = self.gate(x).mean(dim=(1, 2)) 
                if self.training:
                    onehot = F.gumbel_softmax(scores, tau=tau, hard=True, dim=0)   # (3,) one-hot
                    chosen = (onehot[:, None, None] * x).sum(dim=0)
                    idx = onehot.argmax()
                else:
                    idx = scores.argmax()
                    chosen = x[idx]
                combined_results =  input_vb +  chosen.view(768, 65, 4) 
                return combined_results
                values = torch.stack([solver_result_1, solver_result_2, torch.zeros_like(solver_result_2)], dim=-1)  # the tensors you're choosing between
                chosen = values.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
                combined_results =  input_vb + chosen
                return combined_results
            elif self.mode == 2:
                input_vb_reshaped = torch.reshape(torch.clone(puzzle_embs), ( -1,BATCH_SIZE))
                solver_result_1 = self.soduko_solver_v1(input_vb_reshaped)
                #solver_result_2 = self.аddd(input_vb_reshaped)
                #solver_result_3 = self.sub(input_vb_reshaped)
                combined_results =  self.cat_all(torch.cat([input_vb_reshaped,solver_result_1,], dim=-1))#,solver_result_2,solver_result_3
                attn_output = combined_results.reshape(shape_1,shape_2, -1)
                return attn_output
class HierarchicalReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()

        self.config = config
        if self.config.mlp_t:
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len, # L
                expansion=config.expansion,
            )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False
            )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # B, L, D = hidden_states.shape
        # Post Norm
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1,2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1,2)
        else:
            # Self Attention
            hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        # Fully Connected
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states

class HierarchicalReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[HierarchicalReasoningModel_ACTV1Block]):
        super().__init__()

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        # Input injection (add)
        hidden_states = hidden_states + input_injection
        # Layers
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)

        return hidden_states


class HierarchicalReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)
        self.circuit = Circuit()
        wandb.watch(self.circuit, log="all", log_freq=100)
        # I/O
        self.embed_scale  = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale
        self.step = 0
        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            pass

        # Reasoning Layers
        self.H_level = HierarchicalReasoningModel_ACTV1ReasoningModule(layers=[HierarchicalReasoningModel_ACTV1Block(self.config) for _i in range(self.config.H_layers)])
        self.L_level = HierarchicalReasoningModel_ACTV1ReasoningModule(layers=[HierarchicalReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)])

        # Initial states
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]

            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: HierarchicalReasoningModel_ACTV1InnerCarry):
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(self, carry: HierarchicalReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor],tau=1.0) -> Tuple[HierarchicalReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        #print('shape1',batch["inputs"].shape)
        #print('shape2',batch["labels"].shape)

        input_embeddings = self._input_embeddings(batch["inputs"].clone(), batch["puzzle_identifiers"].clone())
        safe_labels = batch["labels"].clone()
        safe_labels = torch.where(safe_labels < 0, torch.zeros_like(safe_labels), safe_labels)
        safe_labels = torch.where(safe_labels >= 12, torch.zeros_like(safe_labels), safe_labels)
        labels_embeddings = self._input_embeddings(safe_labels, batch["puzzle_identifiers"].clone())
        #print('shape3',input_embeddings.shape)
        #print('shape4',labels_embeddings.shape)
        # Forward iterations
        #if self.step % 10 == 0:
        #    wandb.log({"bridge/gate_value": self.bridge.gate.item()}, step=self.step)
        #self.step +=1
        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L
            for _H_step in range(self.config.H_cycles):
                for _L_step in range(self.config.L_cycles):
                    if not ((_H_step == self.config.H_cycles - 1) and (_L_step == self.config.L_cycles - 1)):
                        z_L = self.L_level(z_L, z_H, **seq_info)
                
                if not (_H_step == self.config.H_cycles - 1):
                    #print("z_H",z_H.shape)
                    #print("labels_embeddings",labels_embeddings.shape)
                    #z_H = torch.add(z_H,torch.mul(labels_embeddings,0.1))
                    z_H = self.circuit(z_H,tau=tau)

                    z_H = self.H_level(z_H, z_L, **seq_info)
        assert not z_H.requires_grad and not z_L.requires_grad
        # 1-step grad
        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        #print("z_H",z_H.shape)
        #print("labels_embeddings",labels_embeddings.shape)
        #z_H = torch.add(z_H,torch.mul(labels_embeddings,0.1))
        z_H = self.circuit(z_H,tau=tau)
        z_H = self.H_level(z_H, z_L, **seq_info)

        # LM Outputs
        new_carry = HierarchicalReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())  # New carry no grad
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]

        # Q head
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class HierarchicalReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HierarchicalReasoningModel_ACTV1Config(**config_dict)
        self.inner = HierarchicalReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return HierarchicalReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),  # Default to halted
            
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(self, carry: HierarchicalReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor],tau=1.0) -> Tuple[HierarchicalReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data,tau=tau)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }
        
        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):
                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
                halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)

                halted = halted & (new_steps >= min_halt_steps)

                # Compute target Q
                # NOTE: No replay buffer and target networks for computing target Q-value.
                # As batch_size is large, there're many parallel envs.
                # Similar concept as PQN https://arxiv.org/abs/2407.04811
                next_q_halt_logits, next_q_continue_logits = self.inner(new_inner_carry, new_current_data)[-1]
                
                outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return HierarchicalReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
