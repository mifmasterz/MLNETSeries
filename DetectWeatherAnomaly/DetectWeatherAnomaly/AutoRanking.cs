﻿
using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML;
using Microsoft.ML.AutoML;

namespace DetectWeatherAnomaly
{
    public class AutoRanking
    {
        private static string ModelPath = "Model.zip";
        private static string LabelColumnName = "Label";
        private static string GroupColumnName = "GroupId";
        private static uint ExperimentTime = 60;

        public static void Run()
        {
            MLContext mlContext = new MLContext();

            // STEP 1: Load data
            IDataView trainDataView = mlContext.Data.LoadFromEnumerable<SearchData>(GenerateData());
            IDataView testDataView = mlContext.Data.LoadFromEnumerable <SearchData>(GenerateData(10));

            // STEP 2: Run AutoML experiment
            Console.WriteLine($"Running AutoML recommendation experiment for {ExperimentTime} seconds...");
            ExperimentResult<RankingMetrics> experimentResult = mlContext.Auto()
                .CreateRankingExperiment(new RankingExperimentSettings() { MaxExperimentTimeInSeconds = ExperimentTime })
                .Execute(trainDataView, testDataView,
                    new ColumnInformation()
                    {
                        LabelColumnName = LabelColumnName,
                        GroupIdColumnName = GroupColumnName
                    });

            // STEP 3: Print metric from best model
            RunDetail<RankingMetrics> bestRun = experimentResult.BestRun;
            Console.WriteLine($"Total models produced: {experimentResult.RunDetails.Count()}");
            Console.WriteLine($"Best model's trainer: {bestRun.TrainerName}");
            Console.WriteLine($"Metrics of best model from validation data --");
            PrintMetrics(bestRun.ValidationMetrics);

            // STEP 5: Evaluate test data
            IDataView testDataViewWithBestScore = bestRun.Model.Transform(testDataView);
            RankingMetrics testMetrics = mlContext.Ranking.Evaluate(testDataViewWithBestScore, labelColumnName: LabelColumnName);
            Console.WriteLine($"Metrics of best model on test data --");
            PrintMetrics(testMetrics);

            // STEP 6: Save the best model for later deployment and inferencing
            mlContext.Model.Save(bestRun.Model, trainDataView.Schema, ModelPath);

            // STEP 7: Create prediction engine from the best trained model
            var predictionEngine = mlContext.Model.CreatePredictionEngine<SearchData, SearchDataPrediction>(bestRun.Model);

            // STEP 8: Initialize a new test, and get the prediction
            var testPage = new SearchData
            {
                GroupId = "1",
                Features = 9,
                Label = 1
            };
            var prediction = predictionEngine.Predict(testPage);
            Console.WriteLine($"Predicted rating for: {prediction.Prediction}");

            // New Page
            testPage = new SearchData
            {
                GroupId = "2",
                Features = 2,
                Label = 9
            };
            prediction = predictionEngine.Predict(testPage);
            Console.WriteLine($"Predicted: {prediction.Prediction}");

            Console.WriteLine("Press any key to continue...");
            Console.ReadKey();
        }

        private static void PrintMetrics(RankingMetrics metrics)
        {
            Console.WriteLine($"NormalizedDiscountedCumulativeGains: {metrics.NormalizedDiscountedCumulativeGains}");
            Console.WriteLine($"DiscountedCumulativeGains: {metrics.DiscountedCumulativeGains}");

        }

        public static List<SearchData> GenerateData(int count=100)
        {
            var data = new List<SearchData>();
            var rnd = new Random();
            for(int i = 0; i < count; i++)
            {
                var newNode = new SearchData()
                {
                    GroupId = rnd.Next(1, 5).ToString(), //nama yang di rating
                    Label = rnd.Next(0,4), //rating 0-4
                    Features = rnd.Next(1,100) //user atau identitas

                };
                data.Add(newNode);
            }
            return data;
        }
    }
    public class SearchDataPrediction
    {
        [ColumnName("PredictedLabel")]
        public float Prediction;

        public float Score { get; set; }
    }
    public class SearchData
    {
        [LoadColumn(0)]
        public string GroupId;

        [LoadColumn(1)]
        public float Features;

        [LoadColumn(2)]
        public float Label;
    }

    
}
