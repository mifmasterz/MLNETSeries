using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace IncomeClusteringApp
{
    class Program
    {
        static void Main(string[] args)
        {
            //Create MLContext
            MLContext mlContext = new MLContext();

            //Load Data File
            IDataView trainData = mlContext.Data.LoadFromTextFile<IncomeData>(GetAbsolutePath("../../../Data/AdultIncome.csv"), separatorChar: ',', hasHeader: true);

            //Data process configuration with pipeline data transformations 
            var dataPrepTransform = mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "workclass_encoded", inputColumnName: "workclass")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "education_encoded", inputColumnName: "education"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "marital_status_encoded", inputColumnName: "marital_status"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "occupation_encoded", inputColumnName: "occupation"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "relationship_encoded", inputColumnName: "relationship"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "race_encoded", inputColumnName: "race"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "sex_encoded", inputColumnName: "sex"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "native_country_encoded", inputColumnName: "native_country"))
                .Append(mlContext.Transforms.Concatenate("Features", new[] { "age", "fnlwgt", "education_num", "marital_status_encoded", "relationship_encoded", "race_encoded", "sex_encoded", "hours_per_week" }))
                .Append(mlContext.Transforms.NormalizeMinMax("Features", "Features"))
                .AppendCacheCheckpoint(mlContext);
            
            // Create data transformer
            ITransformer dataPrepTransformer = dataPrepTransform.Fit(trainData);

            IDataView transformedTrainingData = dataPrepTransformer.Transform(trainData);
            // Choose learner
            var CluteringEstimator = mlContext.Clustering.Trainers.KMeans(featureColumnName: "Features", numberOfClusters: 2);

            // Build machine learning model
            var trainedModel = dataPrepTransformer.Append(CluteringEstimator.Fit(transformedTrainingData));

            // Measure trained model performance
            var testData = trainedModel.Transform(transformedTrainingData);
            var testMetrics = mlContext.Clustering.Evaluate(testData);
            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Clustering model - Test Data     ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       AverageDistance:            {testMetrics.AverageDistance:0.###}");
            Console.WriteLine($"*       DaviesBouldinIndex:         {testMetrics.DaviesBouldinIndex:0.###}");
            Console.WriteLine($"*       NormalizedMutualInformation:{testMetrics.NormalizedMutualInformation:#.###}");
            Console.WriteLine($"*************************************************************************************************************");

            var modelRelativePath = GetAbsolutePath("MLModel.zip");
            mlContext.Model.Save(trainedModel, trainData.Schema, GetAbsolutePath(modelRelativePath));
            Console.WriteLine("The model is saved to {0}", GetAbsolutePath(modelRelativePath));

            ITransformer mlModel = mlContext.Model.Load(GetAbsolutePath(modelRelativePath), out DataViewSchema inputSchema);
            var predEngine = mlContext.Model.CreatePredictionEngine<IncomeData, ClusterPrediction>(mlModel);

            // Create sample data to do a single prediction with it 
            var sampleData1 = mlContext.Data.CreateEnumerable<IncomeData>(trainData, false).First();
            var sampleData2 = mlContext.Data.CreateEnumerable<IncomeData>(trainData, false).Skip(1).First();
            // Try a single prediction
            ClusterPrediction predictionResult = predEngine.Predict(sampleData1);
            Console.WriteLine($"Sample 1");
            Console.WriteLine($"Cluster: {predictionResult.PredictedClusterId}");
            Console.WriteLine($"Distances: {string.Join(" ", predictionResult.Distances)}");
            Console.WriteLine($"Sample 2");
            ClusterPrediction predictionResult2 = predEngine.Predict(sampleData2);
            Console.WriteLine($"Cluster: {predictionResult2.PredictedClusterId}");
            Console.WriteLine($"Distances: {string.Join(" ", predictionResult2.Distances)}");
            Console.WriteLine("both sample must be on the same cluster..");
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
