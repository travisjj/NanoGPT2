using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using F = TorchSharp.torch.nn.functional;
using Module = TorchSharp.torch.nn.Module;

namespace NanoGPT2
{
    /// <summary>
    /// GPT2 implementation using HuggingFace naming conventions, inspired by Karpathy's NanoGPT.
    /// This class implements a GPT-2 transformer model, including token embeddings, 
    /// positional embeddings, multi-layer transformer blocks, and a final projection layer.
    /// </summary>
    public class GPT2 : Module<Tensor, Tensor>
    {
        /// <summary>
        /// Gets the hyperparameters for the GPT-2 model via shared state.
        /// </summary>
        public HyperParams hp { get { return this.gps.hp; } }

        /// <summary>
        /// Encapsulates state shared across the model, such as hyperparameters and configuration.
        /// </summary>
        public GPTState gps { get; set; }

        /// <summary>
        /// Stores the model's submodules (embeddings, layers, etc.) using HuggingFace-style naming conventions.
        /// </summary>
        public ModuleDict<Module> transformer;

        /// <summary>
        /// Final linear layer that projects the transformer outputs to logits for the vocabulary.
        /// </summary>
        public Linear ln_f;

        /// <summary>
        /// Initializes a new instance of the GPT2 class with the specified hyperparameters.
        /// </summary>
        /// <param name="HyperParams">The hyperparameters for the model, including dimensions and layer counts.</param>
        public GPT2(HyperParams HyperParams) : base("GPT2")
        {
            // Initialize shared state
            this.gps = new GPTState(HyperParams);

            // Closure function to share state between components
            Func<GPTState> ShareState = () => { return this.gps; };

            /// <summary>
            /// Initializes transformer submodules:
            /// - Token embeddings (`wte`) map vocabulary indices to dense vectors.
            /// - Positional embeddings (`wpe`) map sequence positions to dense vectors.
            /// - Transformer blocks (`h`) process input data sequentially.
            /// - Layer normalization (`ln_f`) normalizes the final transformer outputs.
            /// </summary>
            this.transformer = nn.ModuleDict(new (string, Module)[] {
                ("wte", nn.Embedding(hp.VocabSize, hp.EmbedDimensions, device: hp.Device)),
                ("wpe", nn.Embedding(hp.BlockSize, hp.EmbedDimensions, device: hp.Device)),
                ("h", nn.ModuleList(Enumerable.Range(0, hp.BlockLayers).Select(_ => new Block(ShareState)).ToArray())),
                ("ln_f", nn.LayerNorm(hp.EmbedDimensions, device: hp.Device))
            });

            // Register the embeddings and transformer blocks as modules
            register_module("wte", (Embedding)this.transformer["wte"]);
            register_module("wpe", (Embedding)this.transformer["wpe"]);
            register_module("h", (ModuleList<Block>)this.transformer["h"]);
            register_module("ln_f", (LayerNorm)this.transformer["ln_f"]);

            /// <summary>
            /// Defines the final linear layer that projects the outputs to logits for the vocabulary.
            /// This layer shares weights with the token embedding layer for parameter efficiency.
            /// </summary>
            this.ln_f = nn.Linear(hp.EmbedDimensions, hp.VocabSize, hasBias: false, device: hp.Device);

            // Share weights between token embeddings and final projection layer
            ((Embedding)this.transformer["wte"]).weight = this.ln_f.weight;

            // Apply weight initialization to all submodules
            this.apply(InitWeights);
        }

        /// <summary>
        /// Initializes the weights of the model components (Linear and Embedding layers).
        /// </summary>
        /// <param name="module">The module whose weights will be initialized.</param>
        private void InitWeights(Module module)
        {
            if (module is Linear)
            {
                double std = 0.02;
                var isProj = this.gps.HashCodes.ContainsKey(module.GetHashCode());
                if (isProj)
                {
                    std *= Math.Sqrt(2.0 * gps.hp.BlockLayers);
                }
                torch.nn.init.normal_(((Linear)module).weight, mean: 0.0, std: std);

                if (((Linear)module).bias is not null)
                {
                    torch.nn.init.zeros_(((Linear)module).bias);
                }
            }
            if (module is Embedding)
            {
                torch.nn.init.normal_(((Embedding)module).weight, mean: 0.0, std: 0.02);
            }
        }

        /// <summary>
        /// Performs a forward pass through the GPT-2 model.
        /// </summary>
        /// <param name="input">Input tensor of token indices (batch size x sequence length).</param>
        /// <returns>Output logits tensor (batch size x sequence length x vocabulary size).</returns>
        public override Tensor forward(Tensor input)
        {
            var (B, T) = input.size();

            // Create position indices for the sequence
            var position = torch.arange(0, T, dtype: ScalarType.Int64, device: hp.Device);

            // Obtain positional embeddings
            var positionEmbedding = ((Embedding)this.transformer["wpe"]).forward(position);

            // Obtain token embeddings
            var tokenEmbedding = ((Embedding)this.transformer["wte"]).forward(input);

            // Combine token and positional embeddings
            input = tokenEmbedding + positionEmbedding;

            // Pass through each transformer block
            foreach (var block in (ModuleList<Block>)this.transformer["h"])
            {
                input = block.forward(input);
            }

            // Apply final layer normalization
            input = ((LayerNorm)this.transformer["ln_f"]).forward(input);

            // Project to vocabulary logits
            var logits = this.ln_f.forward(input);

            return logits;
        }

        /// <summary>
        /// Performs a forward pass through the GPT-2 model and computes the loss.
        /// </summary>
        /// <param name="input">Input tensor of token indices (batch size x sequence length).</param>
        /// <param name="Targets">Target tensor of token indices (batch size x sequence length).</param>
        /// <returns>A tuple containing the output logits and the loss.</returns>
        public (Tensor, Tensor) forward(Tensor input, Tensor Targets)
        {
            var logits = this.forward(input);
            var loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Targets.view(-1));
            return (logits, loss);
        }
    }

    /// <summary>
    /// Represents a single transformer block within the GPT-2 model.
    /// A block consists of a LayerNorm, a causal self-attention mechanism, another LayerNorm, and a feed-forward MLP.
    /// </summary>
    public class Block : Module<Tensor, Tensor>
    {
        /// <summary>
        /// Gets the hyperparameters for the GPT-2 model via shared state.
        /// </summary>
        public HyperParams hp { get { return this.GPTState.hp; } }

        /// <summary>
        /// Gets the shared GPT state that contains configuration and reusable objects.
        /// </summary>
        public GPTState GPTState { get { return this._shareState(); } }

        /// <summary>
        /// Delegate function to access shared GPT state.
        /// </summary>
        private Func<GPTState> _shareState { get; set; }

        /// <summary>
        /// First layer normalization applied before the causal self-attention.
        /// </summary>
        public LayerNorm ln_1;

        /// <summary>
        /// Causal self-attention mechanism for processing sequential data.
        /// </summary>
        public CausalSelfAttention attn;

        /// <summary>
        /// Second layer normalization applied before the feed-forward MLP.
        /// </summary>
        public LayerNorm ln_2;

        /// <summary>
        /// Feed-forward multi-layer perceptron (MLP) for additional processing of the attention output.
        /// </summary>
        public MLP mlp;

        /// <summary>
        /// Initializes a new instance of the Block class with the specified shared GPT state.
        /// </summary>
        /// <param name="State">A function that provides access to the shared GPT state.</param>
        public Block(Func<GPTState> State) : base("Block")
        {
            // Store the delegate function for shared state access
            this._shareState = State;

            // Initialize the first LayerNorm
            this.ln_1 = nn.LayerNorm(hp.EmbedDimensions, device: hp.Device);
            register_module("ln_1", this.ln_1);

            // Initialize the causal self-attention mechanism
            this.attn = new CausalSelfAttention(State);

            // Initialize the second LayerNorm
            this.ln_2 = nn.LayerNorm(hp.EmbedDimensions, device: hp.Device);
            register_module("ln_2", this.ln_2);

            // Initialize the feed-forward MLP
            this.mlp = new MLP(State);
            register_module("mlp", this.mlp);
        }

        /// <summary>
        /// Performs a forward pass through the block.
        /// </summary>
        /// <param name="input">Input tensor of shape (batch size x sequence length x embedding dimensions).</param>
        /// <returns>Output tensor of the same shape as the input, after processing by the block.</returns>
        public override Tensor forward(Tensor input)
        {
            // Apply first LayerNorm and add the result of causal self-attention to the input (residual connection)
            input = input + this.attn.forward(this.ln_1.forward(input));

            // Apply second LayerNorm and add the result of the feed-forward MLP to the input (residual connection)
            input = input + this.mlp.forward(this.ln_2.forward(input));

            return input;
        }
    }

    /// <summary>
    /// Represents the feed-forward multi-layer perceptron (MLP) module used within each transformer block of GPT-2.
    /// The MLP consists of a linear transformation, activation function (GELU), and a second linear transformation.
    /// </summary>
    public class MLP : Module<Tensor, Tensor>
    {
        /// <summary>
        /// Gets the hyperparameters for the GPT-2 model via shared state.
        /// </summary>
        public HyperParams hp { get { return this.GPTState.hp; } }

        /// <summary>
        /// Gets the shared GPT state that contains configuration and reusable objects.
        /// </summary>
        public GPTState GPTState { get { return this._shareState(); } }

        /// <summary>
        /// Delegate function to access shared GPT state.
        /// </summary>
        private Func<GPTState> _shareState { get; set; }

        /// <summary>
        /// First linear layer that expands the input dimensionality by a factor of 4.
        /// </summary>
        public Linear c_fc;

        /// <summary>
        /// Second linear layer that projects back to the original input dimensionality.
        /// </summary>
        public Linear c_proj;

        /// <summary>
        /// GELU activation function applied between the two linear layers.
        /// </summary>
        public GELU gelu;

        /// <summary>
        /// Dropout layer for regularization, applied after the GELU activation.
        /// </summary>
        public Dropout dropout;

        /// <summary>
        /// Initializes a new instance of the MLP class with the specified shared GPT state.
        /// </summary>
        /// <param name="State">A function that provides access to the shared GPT state.</param>
        public MLP(Func<GPTState> State) : base("MLP")
        {
            // Store the delegate function for shared state access
            this._shareState = State;

            // Initialize the first linear layer to expand dimensionality
            this.c_fc = nn.Linear(hp.EmbedDimensions, 4 * hp.EmbedDimensions, device: hp.Device);

            // Initialize the GELU activation function
            this.gelu = nn.GELU();

            // Initialize the second linear layer to reduce dimensionality back to the original size
            this.c_proj = nn.Linear(4 * hp.EmbedDimensions, hp.EmbedDimensions, device: hp.Device);

            // Track the hash code for weight-sharing purposes in the GPT state
            GPTState.HashCodes[this.c_proj.GetHashCode()] = true;

            // Initialize the dropout layer
            this.dropout = nn.Dropout(hp.Dropout);

            // Register all components as submodules
            RegisterComponents();
        }

        /// <summary>
        /// Performs a forward pass through the MLP.
        /// </summary>
        /// <param name="input">Input tensor of shape (batch size x sequence length x embedding dimensions).</param>
        /// <returns>Output tensor of the same shape as the input, after processing by the MLP.</returns>
        public override Tensor forward(Tensor input)
        {
            // Apply first linear layer
            input = this.c_fc.forward(input);

            // Apply GELU activation
            input = this.gelu.forward(input);

            // Apply second linear layer
            input = this.c_proj.forward(input);

            return input;
        }
    }

    /// <summary>
    /// Represents the causal self-attention mechanism used within each transformer block of GPT-2.
    /// This mechanism computes attention scores, applies masking for causal dependencies, and projects the results.
    /// </summary>
    public class CausalSelfAttention : Module<Tensor, Tensor>
    {
        /// <summary>
        /// Gets the hyperparameters for the GPT-2 model via shared state.
        /// </summary>
        public HyperParams hp { get { return this.GPTState.hp; } }

        /// <summary>
        /// Gets the shared GPT state that contains configuration and reusable objects.
        /// </summary>
        public GPTState GPTState { get { return this._shareState(); } }

        /// <summary>
        /// Delegate function to access shared GPT state.
        /// </summary>
        private Func<GPTState> _shareState { get; set; }

        /// <summary>
        /// Linear layer for computing queries, keys, and values (combined into a single tensor).
        /// </summary>
        public Linear c_attn;

        /// <summary>
        /// Linear layer for projecting the output of the attention mechanism.
        /// </summary>
        public Linear c_proj;

        /// <summary>
        /// Initializes a new instance of the CausalSelfAttention class with the specified shared GPT state.
        /// </summary>
        /// <param name="State">A function that provides access to the shared GPT state.</param>
        public CausalSelfAttention(Func<GPTState> State) : base("CausalSelfAttention_hf")
        {
            // Store the delegate function for shared state access
            this._shareState = State;

            // Initialize the linear layer for query, key, and value computation
            this.c_attn = nn.Linear(hp.EmbedDimensions, 3 * hp.EmbedDimensions, device: hp.Device);
            register_module("c_attn", this.c_attn);

            // Initialize the linear layer for projecting the attention output
            this.c_proj = nn.Linear(hp.EmbedDimensions, hp.EmbedDimensions, device: hp.Device);
            register_module("c_proj", this.c_proj);

            // Track the hash code for weight-sharing purposes in the GPT state
            GPTState.HashCodes[this.c_proj.GetHashCode()] = true;

            // Register the causal mask buffer to enforce attention masking
            this.register_buffer("bias",
                torch.tril(torch.ones(hp.BlockSize, hp.BlockSize, device: hp.Device))
                    .view(1, 1, hp.BlockSize, hp.BlockSize), false
            );
        }

        /// <summary>
        /// Performs a forward pass through the causal self-attention mechanism.
        /// </summary>
        /// <param name="input">Input tensor of shape (batch size x sequence length x embedding dimensions).</param>
        /// <returns>Output tensor of the same shape as the input, after processing by the attention mechanism.</returns>
        public override Tensor forward(Tensor input)
        {
            // Extract input dimensions
            var (B, T, C) = input.size();

            // Compute queries, keys, and values from the input
            var qkv = this.c_attn.forward(input);
            var split = qkv.split(hp.EmbedDimensions, dim: 2);
            var (q, k, v) = (split[0], split[1], split[2]);

            // Reshape and transpose for multi-head attention
            k = k.view(B, T, hp.NumberOfHeads, (int)(C / hp.NumberOfHeads)).transpose(1, 2);
            q = q.view(B, T, hp.NumberOfHeads, (int)(C / hp.NumberOfHeads)).transpose(1, 2);
            v = v.view(B, T, hp.NumberOfHeads, (int)(C / hp.NumberOfHeads)).transpose(1, 2);

            // "Flash" attention (default implementation)
            var y = F.scaled_dot_product_attention(q, k, v, is_casual: true);

            // Reshape and project the output back to the original dimensionality
            y = y.transpose(1, 2).contiguous().view(B, T, C);
            y = this.c_proj.forward(y);

            return y;
        }
    }

    /// <summary>
    /// Allows for a unified HyperParams object to be used through the GPT-2 LLM, in case there needs to be state retention. 
    /// Provides for a HashCodes collection for later lookup used to assist in pre-activiation.
    /// </summary>
    public class GPTState
    {
        /// <summary>
        /// The shared HyperParams instance.
        /// </summary>
        public HyperParams hp { get; set; }

        /// <summary>
        /// A collection of HashCode references for composition during pre-activation.
        /// </summary>
        public Dictionary<int, bool> HashCodes = new();

        /// <summary>
        /// Initializes the State with a configured HyperParameter set.
        /// </summary>
        public GPTState(HyperParams hp)
        {
            this.hp = hp;
        }
    }

    /// <summary>
    /// Represents the hyperparameters used for configuring a GPT-2 style LLM environment.
    /// This class encapsulates all relevant settings, ensuring consistent parameter usage 
    /// throughout the model's lifecycle.
    /// </summary>
    public class HyperParams
    {
        /// <summary>
        /// The size of the vocabulary, representing the total number of unique tokens 
        /// that the model can process.
        /// </summary>
        public int VocabSize { get; set; }

        /// <summary>
        /// The size of the embedding vectors, determining the dimensionality of token embeddings.
        /// </summary>
        public int EmbedDimensions { get; set; }

        /// <summary>
        /// The maximum sequence length the model can process in one forward pass.
        /// </summary>
        public int BlockSize { get; set; }

        /// <summary>
        /// The number of transformer layers (blocks) in the model.
        /// </summary>
        public int BlockLayers { get; set; }

        /// <summary>
        /// The probability of dropping units in dropout layers, used for regularization.
        /// </summary>
        public float Dropout { get; set; }

        /// <summary>
        /// The number of attention heads in the multi-head attention mechanism.
        /// </summary>
        public int NumberOfHeads { get; set; }

        /// <summary>
        /// The learning rate for the optimizer, controlling the step size in gradient updates.
        /// Default value is 0.001.
        /// </summary>
        public float LearningRate = 0.001f;

        /// <summary>
        /// The device (CPU or CUDA) on which the model will run. Default is CPU.
        /// </summary>
        public Device Device { get; set; }

        /// <summary>
        /// Indicates whether the CUDA device is available and being used.
        /// </summary>
        public bool IsCuda { get; set; }

        /// <summary>
        /// The size of each attention head, calculated as EmbedDimensions / NumberOfHeads.
        /// </summary>
        public int HeadSize
        {
            get
            {
                return (int)(this.EmbedDimensions / this.NumberOfHeads);
            }
        }

        /// <summary>
        /// Initializes the hyperparameters with default settings. 
        /// Detects if CUDA is available and sets the device accordingly.
        /// </summary>
        public HyperParams()
        {
            this.Device = torch.CPU;
            if (torch.cuda.is_available())
            {
                this.Device = torch.CUDA;
                this.IsCuda = true;
            }
        }
    }

    /// <summary>
    /// Provides extension methods to mimic Python's tuple unpacking behavior in C#.
    /// These methods enable "deconstruction" of arrays into individual variables,
    /// making array handling more convenient and intuitive.
    /// </summary>
    public static class TupleExtend
    {
        /// <summary>
        /// Deconstructs a single-element long array into one variable.
        /// </summary>
        /// <param name="Array">The input array, expected to have at least one element.</param>
        /// <param name="Val1">The first value extracted from the array.</param>
        public static void Deconstruct(this long[] Array, out long Val1)
        {
            Val1 = Array[0];
        }

        /// <summary>
        /// Deconstructs a two-element long array into two variables.
        /// </summary>
        /// <param name="Array">The input array, expected to have at least two elements.</param>
        /// <param name="Val1">The first value extracted from the array.</param>
        /// <param name="Val2">The second value extracted from the array.</param>
        public static void Deconstruct(this long[] Array, out long Val1, out long Val2)
        {
            Val1 = Array[0];
            Val2 = Array[1];
        }

        /// <summary>
        /// Deconstructs a three-element long array into three variables.
        /// </summary>
        /// <param name="Array">The input array, expected to have at least three elements.</param>
        /// <param name="Val1">The first value extracted from the array.</param>
        /// <param name="Val2">The second value extracted from the array.</param>
        /// <param name="Val3">The third value extracted from the array.</param>
        public static void Deconstruct(this long[] Array, out long Val1, out long Val2, out long Val3)
        {
            Val1 = Array[0];
            // Skip the first element and take the next two elements, assigning them to Val2 and Val3
            (Val2, Val3) = Array.Skip(1).ToArray();
        }
    }
}
