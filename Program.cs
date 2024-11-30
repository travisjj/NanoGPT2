
/*************************************************
 *  Package Manager Console                      *
 *  -----------------------                      *
 *  Use these commands to install dependancies   *
 *  if they are not included or if replicating   *
 * ***********************************************/

// For Tiktoken (required)
// Install-Package Microsoft.ML.Tokenizers -Version 0.22.0-preview.24378.1

// For TorchSharp (required)
// Install-Package TorchSharp -Version 0.103.0

// libtorch configuration:
// 1. Go to Packages in the Solution Explorer, and remove the libtorch reference.
// 2. Open the Package Manager Console and Install the proper configuration from below
// 3. Publish/Run

// For libtorch (required)
// Depending on where you train / generate.... you will need to modify the libtorch reference
// Windows CPU (default)
// Install-Package libtorch-cpu-win-x64 -Version 2.4.0

// Linux CPU
// Install-Package libtorch-cpu-linux-x64 -Version 2.4.0

// Linux GPU (Note: this requires a good amount of space)
// Install-Package libtorch-cuda-12.1-linux-x64 -Version 2.4.0

using NanoGPT2;

public static class Program
{

    // This is used for the loading and saving of files and data. If this directory is incorrect then the project will probably
    // throw and exception on load, alternatively, it could fail gracefully with the message that nothing happened.
    public static string Directory
    {
        get
        {
            string dir = Environment.CurrentDirectory;
            string proj = "NanoGPT2//";

            int idx = dir.IndexOf(proj);

            // Attempts to fall back to the current directory
            if (idx < 0) return dir + "/";

            string path = dir.Substring(0, idx + proj.Length);
            return path;
        }
    }

    public static String VisualStudio_PathOffset = "../../../";

    public static void Main()
    {
        // Running State File Name Suffix
        // Used to ensure uniqueness in running state
        // Otherwise, we may re-use a different model's running state loop counter
        string TrainingDescription = "SH-Trainer";


        // LLM Runtime Environment for Training GPT2 small
        var smallGPT2 = new TrainingEnvironment("small")
        {
            // Token Batch Size
            B = 64,
            // Token "Time" (context) Size
            T = 512,

            // Weights and Bias (model) file
            ModelFilePath = Directory + "GPTNano-SH.dat", // Sample model name

            // Auto-Save on training completion to the ModelFilePath (overwrites)
            // If manually saving, make sure to save AFTER generating or some thread based processors
            // may have asyn problems with file writing
            SaveModelOnComplete = true,

            // Main text file for training
            CorpusFilePath = Directory + "shakespeare.txt",

            // Location where the training state is held between code runs
            RunningStateFileName = $"NanoGPT2-{TrainingDescription}.txt",

            // Used for gradient accumulation
            BatchSize = (int)Math.Pow(2, 19), // ~0.5M per GPT2
            GradientAccumulation = false, // A huge batch size was used for training GPT2

            // How many minutes the main loop runs until breaking out
            RuntimeMinutes = 2,

            // Auto-produce 5 sequences on complete
            // Use trainer.Generate(n) where n is sequences for manual generation
            // If manually generating, make sure to save AFTER generating or some thread based processors
            // may have asyn problems with file writing
            GenerateOnComplete = true,

            // Generate prompt at the end of training
            Prompt = "ARTHUR: NO SLEEP TIL"
        };

        using (var trainer = new Trainer(smallGPT2))
        {
            trainer.Train();
        }
    }
}