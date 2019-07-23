using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace SeedsApp
{
    class Program
    {
        static void Main(string[] args)
        {
            //Create MLContext
            MLContext mlContext = new MLContext();

            //Load Data File
            IDataView trainData = mlContext.Data.LoadFromTextFile<SeedData>(GetAbsolutePath("../../../Data/seeds_dataset.txt"), separatorChar: '\t', hasHeader: false);
            
            //Data process configuration with pipeline data transformations 
            var dataPrepTransform = mlContext.Transforms.Conversion.MapValueToKey("Label", "Category")
                                      .Append(mlContext.Transforms.Concatenate("Features", new[] { "Area", "Perimeter", "Compactness", "Length", "Width", "AsymmetryCoefficient", "LengthOfKernel"}))
                                      .Append(mlContext.Transforms.NormalizeMinMax("Features", "Features"))
                                      .AppendCacheCheckpoint(mlContext);

            // Create data prep transformer
            ITransformer dataPrepTransformer = dataPrepTransform.Fit(trainData);
            IDataView transformedTrainingData = dataPrepTransformer.Transform(trainData);
            // Choose learner
            var SdcaEstimator = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features");

            // Build machine learning model
            var trainedModel = dataPrepTransformer.Append(SdcaEstimator.Fit(transformedTrainingData));

            // Measure trained model performance
            var testData = trainedModel.Transform(transformedTrainingData);
            var testMetrics = mlContext.MulticlassClassification.Evaluate(testData);
            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
            Console.WriteLine($"*       MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");
            Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
            Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
            Console.WriteLine($"*************************************************************************************************************");

            var modelRelativePath = GetAbsolutePath("MLModel.zip");
            mlContext.Model.Save(trainedModel, trainData.Schema, GetAbsolutePath(modelRelativePath));
            Console.WriteLine("The model is saved to {0}", GetAbsolutePath(modelRelativePath));

            ITransformer mlModel = mlContext.Model.Load(GetAbsolutePath(modelRelativePath), out DataViewSchema inputSchema);
            var predEngine = mlContext.Model.CreatePredictionEngine<SeedData, SeedPrediction>(mlModel);
            VBuffer<int> keys = default;

            predEngine.OutputSchema["PredictedLabel"].GetKeyValues(ref keys);

            var labelsArray = keys.DenseValues().ToArray();

            Dictionary<int, string> CancerTypes = new Dictionary<int, string>();

            CancerTypes.Add(1, "Kama");

            CancerTypes.Add(2, "Rosa");

            CancerTypes.Add(3, "Canadian");

            // Create sample data to do a single prediction with it 
            var sampleData = mlContext.Data.CreateEnumerable<SeedData>(trainData, false).First();

            // Try a single prediction
            SeedPrediction predictionResult = predEngine.Predict(sampleData);
            var Maks = Array.IndexOf(predictionResult.Score, predictionResult.Score.Max());
            Console.WriteLine($"Single Prediction --> Predicted label and score:  {CancerTypes[labelsArray[Maks]]}: {predictionResult.Score[Maks]:0.####}");
         
            Console.ReadKey();
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
