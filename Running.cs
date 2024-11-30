using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NanoGPT2
{
    // Wrapper for the GPT environment in order to help free resources in the event of a memory leak
    // Also uses the Dispose pattern in order to track the training lifetime for a tracking system
    public abstract class Running : System.IDisposable
    {
        // false will disable saving to drive
        public bool SaveToFile = true;

        // State data, eventually intended to be a json structure but for now, it is a simple string 
        public string RunningStateFileData = "";

        // File name for the ending state of the program, for now it just holds where it left off in the training steps
        public string RunningStateFileName = "RunningState.txt";

        // This is used for the loading and saving of files and data. If this directory is incorrect then the project will probably
        // throw and exception on load, alternatively, it could fail gracefully with the message that nothing happened.
        public string Directory
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

        // Used for tracking purposes after the run
        public List<string> ConsoleList = new List<string>();

        // The name of the Calling class, just for header purposes
        public string Runner { get; set; }

        // Global LLM timer for tracking runtime
        public Stopwatch Timer { get; set; }

        int minutes = 0;
        int milliInMinute = 1000 * 60; // 1000ms = 1s, 60s = 1m

        public int RunningMinutes
        {
            get
            {
                return (int)(Timer.ElapsedMilliseconds / milliInMinute);
            }
        }

        // This class is meant to be inherited, so this starts silently at each instantiation of the parent (which should be only once)
        public Running()
        {
            this.Runner = (new System.Diagnostics.StackTrace()).GetFrame(1).GetMethod().DeclaringType.Name;

            p(DateTime.Now + "\n");
            p($",.-~*´¨¯¨`*·~-.¸-(       {this.Runner}()       )-,.-~*´¨¯¨`*·~-.¸\n\n");

            this.Timer = new Stopwatch();
            Timer.Start();
        }

        // Indicates training run complete, optionally record to filesystem the results if SaveToFile = true
        void IDisposable.Dispose()
        {
            this.Timer.Stop();
            p($"\n\n,.-~*´¨¯¨`*·~-.¸-(      /{Runner}()       )-,.-~*´¨¯¨`*·~-.¸");
            p($"\nTraining time to completion: {((int)(Timer.ElapsedMilliseconds / 100)) * 0.1} seconds");

            if (SaveToFile)
            {
                // Create unique-ish suffix in order to review in the future
                var dt = DateTime.Now;
                string suffix = String.Join("-", new int[] { dt.Year, dt.DayOfYear, dt.Hour, DateTime.Now.Minute });
                string FileName = $"LastRunInfo{suffix}.txt";

                // Write Console output to file
                p($"Writing File... {this.Directory + FileName}");
                File.WriteAllLines(this.Directory + FileName, ConsoleList);

                // No suffix here as it is to be re-used (this ensures that we don't train only starting at position 0 all the time)
                // FileData could be expanded to incorporate other facets of training as well. For example, the training environment
                // could be serialized and set here then read at class instantiation in order to facilitate a more modular runtime
                // if using containers or ssh
                p($"Writing File... {this.Directory + RunningStateFileName}");
                File.WriteAllLines(this.Directory + RunningStateFileName, [RunningStateFileData]);
            }

            Console.WriteLine("•"); // old school Ding sound for a little fun (always 5pm somewhere!) "Never underestimate the value of joy"
        }

        // Very thin wrapper for Console.WriteLine which also appends into a list for saving to disk upon runtime completion
        public void p<T>(T arg)
        {
            Console.WriteLine(arg);
            ConsoleList.Add(arg.ToString()); // Later written to LastRunInfo{}.txt
        }
    }
}
