using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace RestauranRecommenderApp
{
    class Program
    {
        static void Main(string[] args)
        {

            //Create MLContext
            MLContext mlContext = new MLContext();

            var filePath = GetAbsolutePath("../../../Data/IENS_USER_ITEM.csv");
            //Load Data File
            IDataView trainData = mlContext.Data.LoadFromTextFile<RestaurantData>(filePath, separatorChar: ',', hasHeader: true);
            //var xx = trainData.Preview();
            //Data process configuration with pipeline data transformations 
            var dataPrepTransform = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "RestaurantNameEncoded", inputColumnName: nameof(RestaurantData.RestaurantName))
                .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "ReviewerEncoded", inputColumnName: nameof(RestaurantData.Reviewer)))
                .Append(mlContext.Transforms.CopyColumns("Label", nameof(RestaurantData.Score)))
                .AppendCacheCheckpoint(mlContext);

            // Create data transformer
            ITransformer dataPrepTransformer = dataPrepTransform.Fit(trainData);

            IDataView transformedTrainingData = dataPrepTransformer.Transform(trainData);
            // Choose learner
            var Estimator = mlContext.Recommendation().Trainers.MatrixFactorization(
                                    labelColumnName: "Label",
                                    matrixColumnIndexColumnName: "RestaurantNameEncoded",
                                    matrixRowIndexColumnName: "ReviewerEncoded");
           
            // Build machine learning model
            var trainedModel = dataPrepTransformer.Append(Estimator.Fit(transformedTrainingData));

            // Measure trained model performance
            var testData = trainedModel.Transform(transformedTrainingData);
            var metrics = mlContext.Regression.Evaluate(testData, labelColumnName: "Label", scoreColumnName: "Score");

            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("Root Mean Squared Error : " + metrics.RootMeanSquaredError.ToString());
            Console.WriteLine("RSquared: " + metrics.RSquared.ToString());
            Console.WriteLine("=============== End of model evaluation ===============");

            var modelRelativePath = GetAbsolutePath("MLModel.zip");
            mlContext.Model.Save(trainedModel, trainData.Schema, GetAbsolutePath(modelRelativePath));
            Console.WriteLine("The model is saved to {0}", GetAbsolutePath(modelRelativePath));

            ITransformer mlModel = mlContext.Model.Load(GetAbsolutePath(modelRelativePath), out DataViewSchema inputSchema);
            var predEngine = mlContext.Model.CreatePredictionEngine<RestaurantData, RestaurantPrediction>(mlModel);

            // Create sample data to do a single prediction with it 
            var sampleDatas = mlContext.Data.CreateEnumerable<RestaurantData>(trainData, false).Take(10);

            foreach (var sampleData in sampleDatas)
            {
                // Try a single prediction
                RestaurantPrediction predictionResult = predEngine.Predict(sampleData);
                if (Math.Round(predictionResult.Score, 1) > 7.5)
                {
                    Console.WriteLine("Restaurant " + sampleData.RestaurantName + " is recommended for reviewer " + sampleData.Reviewer);
                }
                else
                {
                    Console.WriteLine("Restaurant " + sampleData.RestaurantName + " is not recommended for reviewer " + sampleData.Reviewer);
                }
            }
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
