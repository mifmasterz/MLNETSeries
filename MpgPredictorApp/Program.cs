using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;
using System.IO;
using System.Collections.Generic;
using System.Linq;

namespace MpgPredictorApp
{
    class Program
    {
        static string TRAIN_DATA_FILEPATH = @"auto-mpg.data.csv";
        static string modelRelativePath = "MLModel.zip";
        static void Main(string[] args)
        {
            var DataDir = new DirectoryInfo(GetAbsolutePath(@"..\..\..\Data"));

            MLContext mlContext = new MLContext(seed: 1);
            // Load Data
            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<ModelInput>(
                                            path: Path.Combine(DataDir.FullName, TRAIN_DATA_FILEPATH),
                                            hasHeader: true,
                                            separatorChar: ',',
                                            allowQuoting: true,
                                            allowSparse: false);
            //For debugging only
            //var viewData = trainingDataView.Preview();

            // Data process configuration with pipeline data transformations 
            var dataProcessPipeline = mlContext.Transforms.Concatenate("Features", new[] { "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin" });

            // Set the training algorithm 
            var trainer = mlContext.Regression.Trainers.FastTree(new FastTreeRegressionTrainer.Options() { NumberOfLeaves = 6, MinimumExampleCountPerLeaf = 1, NumberOfTrees = 500, LearningRate = 0.08338325f, Shrinkage = 0.08298761f, LabelColumnName = "mpg", FeatureColumnName = "Features" });
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            ITransformer model = trainingPipeline.Fit(trainingDataView);
            mlContext.Model.Save(model, trainingDataView.Schema, GetAbsolutePath(modelRelativePath));
            var crossValidationResults = mlContext.Regression.CrossValidate(trainingDataView, trainingPipeline, numberOfFolds: 5, labelColumnName: "mpg");
            PrintRegressionFoldsAverageMetrics(crossValidationResults);
            ITransformer mlModel = mlContext.Model.Load(GetAbsolutePath(modelRelativePath), out DataViewSchema inputSchema);
            var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);

            ModelInput sampleData = mlContext.Data.CreateEnumerable<ModelInput>(trainingDataView, false).First();

            // Try a single prediction
            ModelOutput predictionResult = predEngine.Predict(sampleData);

            Console.WriteLine($"Single Prediction --> Actual value: {sampleData.Mpg} | Predicted value: {predictionResult.Score}");

            //var hasil = mlModel.Transform(trainingDataView);
            //var metric = mlContext.Regression.Evaluate(hasil, labelColumnName: "mpg");
            //Console.WriteLine($"r-sq {metric.RSquared}");

            Console.ReadLine();
        }
        public static void PrintRegressionFoldsAverageMetrics(IEnumerable<TrainCatalogBase.CrossValidationResult<RegressionMetrics>> crossValidationResults)
        {
            var L1 = crossValidationResults.Select(r => r.Metrics.MeanAbsoluteError);
            var L2 = crossValidationResults.Select(r => r.Metrics.MeanSquaredError);
            var RMS = crossValidationResults.Select(r => r.Metrics.RootMeanSquaredError);
            var lossFunction = crossValidationResults.Select(r => r.Metrics.LossFunction);
            var R2 = crossValidationResults.Select(r => r.Metrics.RSquared);

            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Regression model      ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       Average L1 Loss:       {L1.Average():0.###} ");
            Console.WriteLine($"*       Average L2 Loss:       {L2.Average():0.###}  ");
            Console.WriteLine($"*       Average RMS:           {RMS.Average():0.###}  ");
            Console.WriteLine($"*       Average Loss Function: {lossFunction.Average():0.###}  ");
            Console.WriteLine($"*       Average R-squared:     {R2.Average():0.###}  ");
            Console.WriteLine($"*************************************************************************************************************");
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
