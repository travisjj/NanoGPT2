using Microsoft.ML.Tokenizers;
using TorchSharp;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace NanoGPT2
{
    public class Trainer : Running
    {
        public TrainingEnvironment Environment;
        public GPT2 Model;
        public bool RanToCompletion = false;

        public Trainer(TrainingEnvironment Environment)
        {
            this.Environment = Environment;
            this.Model = new GPT2(Environment.hp);
        }

        public void Train()
        {
            var Env = this.Environment;
            var hp = Env.hp;
            var model = this.Model;
            var (B, T) = (Env.B, Env.T);

            p($"Model Created, using {hp.Device}");
            var tokens = Env.Tokens;

            p($"Model Path: {Env.ModelFilePath}");
            if (File.Exists(Env.ModelFilePath))
            {
                Model.load(Env.ModelFilePath);
            }

            this.RunningStateFileName = Env.RunningStateFileName;
            int startFrom = 0;
            if (File.Exists(this.Directory + this.RunningStateFileName))
            {
                string stateInfo = File.ReadAllText(this.Directory + this.RunningStateFileName);
                startFrom = int.Parse(stateInfo);
            }

            var opto = torch.optim.AdamW(model.parameters(), lr: hp.LearningRate * 3, beta2: 0.95);

            if (hp.IsCuda)
                opto.to(hp.Device);

            int CurrentPosition = (B * T) * startFrom;
            int maxOptoRuns = 768; // Cap for safety
            if (startFrom > maxOptoRuns) startFrom = 0;

            int epoch = (int)(tokens.NumberOfElements / (B * T));

            long tokenCount = tokens.NumberOfElements;
            p($"Loaded {tokenCount} tokens.");
            p($"1 epoch = {epoch}");

            int batchSize = Env.BatchSize;
            int batchIterations = batchSize / (B * T);

            int maxMinutes = Env.RuntimeMinutes;

            // Console Output Table Column Widths
            const int c1 = 9; // step #
            const int c2 = 15; // loss
            const int c3 = 14; //  elapsed ms
            const int c4 = 8; // elapsed seconds
            const int c5 = 12; // tokens per second
            string header = $"| {"Step ",-c1}| {"Loss ",-c2}| {"Millis ",-c3}| {"Sec ",-c4}| {"Tokens/s ",-c5}|";
            /*
             * ---------------------------------------------------------------------
             * | Step     | Loss           | Millis        | Sec     | Tokens/s    |
             * ---------------------------------------------------------------------
            */

            int step = startFrom;
            for (; step < maxOptoRuns && RunningMinutes < maxMinutes && step / epoch < 10; step++)
            {
                float lossAccum = 0.0f;

                p(String.Join("", Enumerable.Range(0, header.Length).Select(z => "-").ToArray()));
                p($"| {"Step ",-c1}| {"Loss ",-c2}| {"Millis ",-c3}| {"Sec ",-c4}| {"Tokens/s ",-c5}|");
                p(String.Join("", Enumerable.Range(0, header.Length).Select(z => "-").ToArray()));

                for (int miniBatch = 0; miniBatch < batchIterations; miniBatch++)
                {
                    // Track this iteration's timing
                    var benchStart = this.Timer.ElapsedMilliseconds;
                    string pstep = "", ploss = "";
                    // Prevent excessive tensors from building up in memory
                    using (torch.NewDisposeScope())
                    {
                        // Back sliding window up for current batch if it is out of bounds
                        if (tokens.NumberOfElements - CurrentPosition < (B * T))
                        {
                            CurrentPosition = (int)(tokens.NumberOfElements - (long)(B * T)) - 1;
                        }

                        // Get the current batch for use
                        var (x, y) = NextBatch(B, T, CurrentPosition, tokens);

                        // Adjust the position by the batch size
                        CurrentPosition += (B * T); //(B * T + 1);
                        // Reset to 0 if the window goes past the end of the boundary
                        if (CurrentPosition >= tokens.NumberOfElements - 1)
                        {
                            CurrentPosition = 0;
                        }

                        if (hp.IsCuda)
                            (x, y) = (x.to(hp.Device), y.to(hp.Device)); // move to cuda

                        opto.zero_grad();

                        // torch.autocast would be used here
                        // not implemented atm in TorchSharp

                        var (logits, loss) = model.forward(x, y);

                        pstep = $"{step}.{miniBatch}";
                        ploss = $"{loss.item<float>()}";

                        if (Env.GradientAccumulation)
                        {
                            loss = loss / batchIterations; // scale down accumulated loss ("normalizer") ts 2:45:00
                            lossAccum += loss.detach().item<float>();
                        }

                        loss.backward();

                        if (!Env.GradientAccumulation)
                            opto.step();

                    }
                    var benchEnd = this.Timer.ElapsedMilliseconds;
                    var dt = benchEnd - benchStart;
                    p($"| {pstep,-c1}| {ploss,-c2}| {dt + " ms",-c3}| {(int)(dt / 1000) + " s",-c4}| {(int)(tokenCount * 1.0 / (dt / 1000.0)),-c5}|");
                }

                if (Env.GradientAccumulation)
                    p($"step: {step}    accumloss: {lossAccum}");

                // Clip gradients
                var norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0);

                // todo: optionally manual update lr 
                /*foreach( var pg in opto.ParamGroups )
                {
                    pg.LearningRate = updated value
                }*/

                if (Env.GradientAccumulation)
                    opto.step();

                if (hp.IsCuda)
                    torch.cuda.synchronize();
            }

            this.RunningStateFileData = step.ToString();

            RanToCompletion = true;
            this.Model = model;
            if(Env.GenerateOnComplete)
            {
                Generate();
            }
            if(Env.SaveModelOnComplete)
            {
                SaveModel();
            }
        }

        private (Tensor, Tensor) NextBatch(int B, int T, int CurrentPosition, Tensor Tokens)
        {
            var buf = Tokens[CurrentPosition..(CurrentPosition + 1 + (B * T))];
            var x = buf[..^1].view(B, T);
            var y = buf[1..].view(B, T);
            return (x, y);
        }

        public void Generate(int num_return_sequences = 5)
        {
            var Prompt = Environment.Prompt;
            var model = this.Model;
            p(Prompt);

            Tokenizer tokenizer = TiktokenTokenizer.CreateForModel("gpt2");

            var longTokens = tokenizer.EncodeToIds(Prompt).Select(i => (long)i).ToArray();
            var tokens = torch.tensor(longTokens, dtype: ScalarType.Int64, device: model.hp.Device);
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1);
            var x = tokens;

            model.eval();

            int max_length = 128;
            while (x.size(1) < max_length)
            {
                using (torch.no_grad())
                {
                    var logits = model.forward(x);
                    logits = logits[.., -1, ..];
                    var probs = F.softmax(logits, dim: -1);
                    var (topk_probs, topk_indices) = torch.topk(probs, 50, dim: -1);
                    var ix = torch.multinomial(topk_probs, 1);
                    var xcol = torch.gather(topk_indices, -1, ix);
                    x = torch.cat([x, xcol], dim: 1);
                }
            }

            for (int i = 0; i < num_return_sequences; i++)
            {
                var theseTokens = x[i, ..max_length].data<long>().Select(l => (int)l).ToList();
                var decoded = tokenizer.Decode(theseTokens);
                p($">{decoded}");
            }
        }

        public void SaveModel()
        {
            if (this.RanToCompletion) { 
                p($"Writing File... {this.Environment.ModelFilePath}");
                this.Model.save(this.Environment.ModelFilePath);
            }
        }
    }
}
