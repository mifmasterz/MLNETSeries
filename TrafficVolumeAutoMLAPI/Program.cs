using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;

namespace TrafficVolumeAutoMLAPI
{
    class Program
    {
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();
            IDataView trainDataView = mlContext.Data.LoadFromTextFile<TrafficData>(GetAbsolutePath("../../../Data/Metro_Interstate_Traffic_Volume.csv"), hasHeader: true,separatorChar:',');
            //configure experiment settings
            var experimentSettings = new RegressionExperimentSettings();
            experimentSettings.MaxExperimentTimeInSeconds = 10;
            var cts = new CancellationTokenSource();
            experimentSettings.CancellationToken = cts.Token;
            experimentSettings.OptimizingMetric = RegressionMetric.MeanSquaredError;
            experimentSettings.CacheDirectory = null;
            
            // Cancel experiment after the user presses any key
            CancelExperimentAfterAnyKeyPress(cts);
            //create experiment
            RegressionExperiment experiment = mlContext.Auto().CreateRegressionExperiment(experimentSettings);
            var handler = new RegressionExperimentProgressHandler();
            //execute experiment
            ExperimentResult<RegressionMetrics> experimentResult = experiment.Execute(trainDataView,   labelColumnName: "Label", progressHandler:handler);
            //Evaluate
            RegressionMetrics metrics = experimentResult.BestRun.ValidationMetrics;
            Console.WriteLine($"Best Algorthm: {experimentResult.BestRun.TrainerName}");
            Console.WriteLine($"R-Squared: {metrics.RSquared:0.##}");
            Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError:0.##}");
            Console.ReadKey();
        }

        static async void CancelExperimentAfterAnyKeyPress(CancellationTokenSource cts)
        {
            Task.Run(() => {
                while (true)
                {
                    Console.ReadKey();
                    cts.Cancel();
                }
            });
        }
        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;
            string fullPath = Path.Combine(assemblyFolderPath, relativePath);
            return fullPath;
        }
    }
}
