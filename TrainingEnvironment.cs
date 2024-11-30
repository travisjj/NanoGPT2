using Microsoft.ML.Tokenizers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch;
using TorchSharp;

namespace NanoGPT2
{
    public enum GPT2Size
    {
        Small,
        Medium,
        Large,
        XL
    }

    /// <summary>
    /// Represents the environment configuration for training a GPT model using CUDA.
    /// Includes file paths, hyperparameter settings, and runtime configurations.
    /// </summary>
    public class TrainingEnvironment
    {
        /// <summary>
        /// Holds the raw training text loaded from the specified text file path.
        /// </summary>
        private string _RawTrainingText = "";

        /// <summary>
        /// Gets the raw training text. If not already loaded, it reads the content from the specified file path.
        /// </summary>
        public string RawTrainingText
        {
            get
            {
                if (String.IsNullOrEmpty(this._RawTrainingText))
                {
                    string rootPath = Path.GetFullPath(CorpusFilePath);
                    this._RawTrainingText = File.ReadAllText(rootPath);
                }
                return this._RawTrainingText;
            }
        }

        /// <summary>
        /// Gets or sets the file path to the training text data (the Corpus).
        /// </summary>
        public string CorpusFilePath = "";

        /// <summary>
        /// Gets or sets the file path to save or load the trained model.
        /// </summary>
        public string ModelFilePath = "";

        /// <summary>
        /// Token Batch Size
        /// </summary>
        public int B { get; set; }

        /// <summary>
        /// Token "Time" (context) Size, also typically used the same as the BlockSize
        /// </summary>
        public int T { get; set; }

        /// <summary>
        /// Gets or sets the batch size for training.
        /// </summary>
        public int BatchSize;

        /// <summary>
        /// Gets or sets the maximum runtime for training in minutes.
        /// </summary>
        public int RuntimeMinutes;

        /// <summary>
        /// Gets or sets a value indicating whether to use gradient accumulation during training.
        /// </summary>
        public bool GradientAccumulation = false;

        /// <summary>
        /// Gets or sets the text prompt used during training or inference.
        /// </summary>
        public string Prompt = "";

        /// <summary>
        /// Gets or sets the name of the state file for saving or resuming training progress.
        /// </summary>
        public string RunningStateFileName = "";

        /// <summary>
        /// Gets or sets the hyperparameters used for configuring the GPT model.
        /// </summary>
        public HyperParams hp = new();

        /// <summary>
        /// Flag for whether to save the model to file after training completion or not.
        /// </summary>
        public bool SaveModelOnComplete = true;

        /// <summary>
        /// Flag for whether to generate from the model after training completion or not.
        /// </summary>
        public bool GenerateOnComplete = true;

        /// <summary>
        /// Initializes a new instance of the TrainingEnvironment class with the specified GPT model size.
        /// </summary>
        /// <param name="GPT2_Size">The size of the GPT model (e.g., "small", "medium", "large", "xl").</param>
        public TrainingEnvironment(GPT2Size GPT2_Size)
        {
            switch (GPT2_Size)
            {
                case GPT2Size.Small:
                    this.hp = GPT2Small();
                    break;
                case GPT2Size.Medium:
                    this.hp = GPT2Medium();
                    break;
                case GPT2Size.Large:
                    this.hp = GPT2Large();
                    break;
                case GPT2Size.XL:
                    this.hp = GPT2XL();
                    break;
                default:
                    this.hp = GPT2Small();
                    break;
            }
        }
        /// <summary>
        /// Initializes a new instance of the TrainingEnvironment class with the specified GPT model size.
        /// </summary>
        /// <param name="GPT2_Size">The size of the GPT model (e.g., "small", "medium", "large", "xl").</param>
        public TrainingEnvironment(string GPT2_Size)
        {
            switch (GPT2_Size.ToLower())
            {
                case "small":
                    this.hp = GPT2Small();
                    break;
                case "medium":
                    this.hp = GPT2Medium();
                    break;
                case "large":
                    this.hp = GPT2Large();
                    break;
                case "xl":
                    this.hp = GPT2XL();
                    break;
                default:
                    this.hp = GPT2Small();
                    break;
            }
        }

        /// <summary>
        /// Initializes a new instance of the TrainingEnvironment class with default values.
        /// </summary>
        public TrainingEnvironment(){}

        /// <summary>
        /// Gets the tokenized representation of the training text.
        /// </summary>
        public Tensor Tokens
        {
            get
            {
                Tokenizer tokenizer = TiktokenTokenizer.CreateForModel("gpt2");
                var tokenize = tokenizer.EncodeToIds(this.RawTrainingText);
                var tokens = torch.tensor(tokenize.Select(i => (long)i).ToList());
                return tokens;
            }
        }

        /// <summary>
        /// Configures the hyperparameters for the GPT-2 Small model.
        /// </summary>
        /// <returns>The hyperparameter configuration for GPT-2 Small.</returns>
        public HyperParams GPT2Small()
        {
            return new HyperParams()
            {
                BlockSize = 1024,
                VocabSize = 50257,
                BlockLayers = 12,
                NumberOfHeads = 12,
                EmbedDimensions = 768
            };
        }

        /// <summary>
        /// Configures the hyperparameters for the GPT-2 Medium model.
        /// </summary>
        /// <returns>The hyperparameter configuration for GPT-2 Medium.</returns>
        public HyperParams GPT2Medium()
        {
            return new HyperParams()
            {
                BlockSize = 1024,
                VocabSize = 50257,
                BlockLayers = 24,
                NumberOfHeads = 16,
                EmbedDimensions = 1024
            };
        }

        /// <summary>
        /// Configures the hyperparameters for the GPT-2 Large model.
        /// </summary>
        /// <returns>The hyperparameter configuration for GPT-2 Large.</returns>
        public HyperParams GPT2Large()
        {
            return new HyperParams()
            {
                BlockSize = 1024,
                VocabSize = 50257,
                BlockLayers = 36,
                NumberOfHeads = 20,
                EmbedDimensions = 1280
            };
        }

        /// <summary>
        /// Configures the hyperparameters for the GPT-2 XL model.
        /// </summary>
        /// <returns>The hyperparameter configuration for GPT-2 XL.</returns>
        public HyperParams GPT2XL()
        {
            return new HyperParams()
            {
                BlockSize = 1024,
                VocabSize = 50257,
                BlockLayers = 48,
                NumberOfHeads = 25,
                EmbedDimensions = 1600
            };
        }
    }
}
