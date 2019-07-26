using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace AirBnBApp
{
    class Program
    {
        static void Main(string[] args)
        {
            var filePath = GetAbsolutePath("../../../Data/listings_tsv.txt");
            List<ListingData> DataFromCSV = new List<ListingData>();
            int RowCount = 0;
            foreach (var line in File.ReadAllLines(filePath))
            {
                RowCount++;
                //skip header
                if (RowCount > 1)
                {
                    var cols = line.Split('\t');
                    DataFromCSV.Add(new ListingData() {
                        name = cols[1],
                        neighbourhood = cols[5],
                        room_type = cols[8],
                        price = float.Parse(cols[9]),
                        minimum_nights = int.Parse(cols[10]),
                        availability_365 = int.Parse(cols[15]),
                        //calculate recommendation (min night <5, num review > 1, rev per month > 0.1, host list count >= 1, availability per year > 10
                        Label = (int.Parse(cols[10]) < 5 && int.Parse(cols[11]) > 1 && float.Parse(cols[13]) > 0.1 && int.Parse(cols[14]) >= 1 && int.Parse(cols[15]) > 10)
                    });
                }
            }
            
            //Create MLContext
            MLContext mlContext = new MLContext();

            //Load Data File
            IDataView trainData = mlContext.Data.LoadFromEnumerable<ListingData>(DataFromCSV);
            var xx = trainData.Preview();
            //Data process configuration with pipeline data transformations 
            var dataPrepTransform = mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "neighbourhood_encoded", inputColumnName: "neighbourhood")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "room_type_encoded", inputColumnName: "room_type"))
                .Append(mlContext.Transforms.Concatenate("Features", new[] { "neighbourhood_encoded", "room_type_encoded", "price", "minimum_nights", "availability_365" }))
                .AppendCacheCheckpoint(mlContext);

            // Create data transformer
            ITransformer dataPrepTransformer = dataPrepTransform.Fit(trainData);

            IDataView transformedTrainingData = dataPrepTransformer.Transform(trainData);
            // Choose learner
            var CluteringEstimator = mlContext.BinaryClassification.Trainers.FieldAwareFactorizationMachine(new string[] { "Features" });

            // Build machine learning model
            var trainedModel = dataPrepTransformer.Append(CluteringEstimator.Fit(transformedTrainingData));

            // Measure trained model performance
            var testData = trainedModel.Transform(transformedTrainingData);
            var metrics = mlContext.BinaryClassification.Evaluate(testData);
            Console.WriteLine();
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine("--------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
            Console.WriteLine("=============== End of model evaluation ===============");

            var modelRelativePath = GetAbsolutePath("MLModel.zip");
            mlContext.Model.Save(trainedModel, trainData.Schema, GetAbsolutePath(modelRelativePath));
            Console.WriteLine("The model is saved to {0}", GetAbsolutePath(modelRelativePath));

            ITransformer mlModel = mlContext.Model.Load(GetAbsolutePath(modelRelativePath), out DataViewSchema inputSchema);
            var predEngine = mlContext.Model.CreatePredictionEngine<ListingData, ListingPrediction>(mlModel);

            // Create sample data to do a single prediction with it 
            var sampleDatas = mlContext.Data.CreateEnumerable<ListingData>(trainData, false).Take(10);
            foreach (var sampleData in sampleDatas)
            {
                // Try a single prediction
                ListingPrediction predictionResult = predEngine.Predict(sampleData);
                Console.WriteLine($"Single Prediction {sampleData.name} --> Predicted:  { (predictionResult.PredictedLabel ? "Recommended" : "Not Recommended") }");
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
