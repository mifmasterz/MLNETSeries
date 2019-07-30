using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Immutable;
using System.IO;
using System.Linq;

namespace FeatureSelectionWithPFI
{
    class Program
    {
        static void Main(string[] args)
        { 
            //Create MLContext
            MLContext mlContext = new MLContext(seed:0);
            var filePath = GetAbsolutePath("../../../Data/machine.data");
            //Load Data File
            IDataView data = mlContext.Data.LoadFromTextFile<CpuData>(filePath,',',false);
            //var xx = data.Preview();
            // 1. Get the column name of input features.
            var featureColumnNames =
                data.Schema
                    .Select(column => column.Name)
                    .Where(columnName => columnName != "Label" && columnName != "Vendor" && columnName != "Model").ToList();
            featureColumnNames.AddRange(new string []{"ModelEncoded","VendorEncoded" });

            // 2. Define estimator with data pre-processing steps
            IEstimator<ITransformer> dataPrepEstimator = mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorEncoded",inputColumnName:"Vendor")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "ModelEncoded", inputColumnName: "Model"))
                .Append(mlContext.Transforms.Concatenate("Features",  featureColumnNames.ToArray()))
                .Append(mlContext.Transforms.NormalizeMinMax("Features"));

            // 3. Create transformer using the data pre-processing estimator
            ITransformer dataPrepTransformer = dataPrepEstimator.Fit(data);

            // 4. Pre-process the training data
            IDataView preprocessedTrainData = dataPrepTransformer.Transform(data);

            // 5. Define Stochastic Dual Coordinate Ascent machine learning estimator
            var sdcaEstimator = mlContext.Regression.Trainers.Sdca();

            // 6. Train machine learning model
            var sdcaModel = sdcaEstimator.Fit(preprocessedTrainData);
            ImmutableArray<RegressionMetricsStatistics> permutationFeatureImportance = mlContext.Regression
            .PermutationFeatureImportance(sdcaModel, preprocessedTrainData, permutationCount: 3);
            // Order features by importance
            var featureImportanceMetrics =
                permutationFeatureImportance
                    .Select((metric, index) => new { index, metric.RSquared })
                    .OrderByDescending(myFeatures => Math.Abs(myFeatures.RSquared.Mean));

            Console.WriteLine("Feature\tPFI");

            foreach (var feature in featureImportanceMetrics)
            {
                if (feature.index < 7)
                    Console.WriteLine($"{featureColumnNames[feature.index],-20}|\t{feature.RSquared.Mean:F6}");
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
