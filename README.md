# NanoGPT2

NanoGPT2 is an implementation of GPT-2 which follows the naming conventions and model architetcture associated with OpenAI's original release per what is publicly available on HuggingFace.

This library was inspired by Andrej Karpathy's video series.

This is not a production ready library, although it does train fairly well. It is mostly meant for demonstration purposes and comes with no support. That said, if you find something wrong or think something needs to be fixed, please let me know and I will look into it.

This will run without any modification so long as the dependencies as installed. It defaults to writing a few files to disk, the .dat file it saves for training which can be up to a gig or more depending on the size chosen (I believe small is ~600mb), a .txt file with the information from the last run, and a .txt file that it saves its training step number into so trianing can be executed in time sequences.

# Package Manager Console  

Use these commands to install dependancies if they are not included or if replicating.

 - For Tiktoken (required)  
   Install-Package Microsoft.ML.Tokenizers -Version 0.22.0-preview.24378.1

- For TorchSharp (required)  
   Install-Package TorchSharp -Version 0.103.0

*libtorch configuration:*  
1. Go to Packages in the Solution Explorer, and remove the libtorch reference.
2. Open the Package Manager Console and Install the proper configuration from below
3. Publish/Run


*For libtorch (required)*    
Depending on where you train / generate.... you will need to modify the libtorch reference  

- Windows CPU (default)  
Install-Package libtorch-cpu-win-x64 -Version 2.4.0

- Windows GPU (Note: this requires a good amount of space)  
Install-Package libtorch-cuda-12.1-win-x64 -Version 2.4.0

- Linux CPU
Install-Package libtorch-cpu-linux-x64 -Version 2.4.0

- Linux GPU (Note: this requires a good amount of space)
Install-Package libtorch-cuda-12.1-linux-x64 -Version 2.4.0
