﻿using System;
using System.Collections.Generic;
using Microsoft.Data.Analysis;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.TimeSeries;
namespace DetectWeatherAnomaly
{
    class Program
    {
        static void Main(string[] args)
        {
            AutoRanking.Run();
            Console.ReadLine();
            return;

            DetectSeasonality.Example();
            Console.ReadLine();
            return;
            RootCauseDetect.Example();
            Console.ReadLine();
            return;
            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, 
            // as well as the source of randomness.
            var ml = new MLContext();

            var ds = DataFrame.LoadCsv("../../../Dataset/weather.csv", ',', true);
            // Generate sample series data with an anomaly
            var data = new List<TimeSeriesData>();
            foreach (var dr in ds.Rows)
            {
                data.Add(new TimeSeriesData { Value = Convert.ToDouble(dr[10]) });

            }

            //for (int index = 0; index < 20; index++)
            //{
            //    data.Add(new TimeSeriesData { Value = 5 });
            //}
            //data.Add(new TimeSeriesData { Value = 10 });
            //for (int index = 0; index < 5; index++)
            //{
            //    data.Add(new TimeSeriesData { Value = 5 });
            //}

            // Convert data to IDataView.
            var dataView = ml.Data.LoadFromEnumerable(data);

            // Setup the detection arguments
            string outputColumnName = nameof(SrCnnAnomalyDetection.Prediction);
            string inputColumnName = nameof(TimeSeriesData.Value);

            // Do batch anomaly detection
            var outputDataView = ml.AnomalyDetection.DetectEntireAnomalyBySrCnn(dataView, outputColumnName, inputColumnName,
                threshold: 0.35, batchSize: 512, sensitivity: 90.0, detectMode: SrCnnDetectMode.AnomalyAndMargin);

            // Getting the data of the newly created column as an IEnumerable of
            // SrCnnAnomalyDetection.
            var predictionColumn = ml.Data.CreateEnumerable<SrCnnAnomalyDetection>(
                outputDataView, reuseRowObject: false);

            Console.WriteLine("Index\tData\tAnomaly\tAnomalyScore\tMag\tExpectedValue\tBoundaryUnit\tUpperBoundary\tLowerBoundary");

            int k = 0;
            foreach (var prediction in predictionColumn)
            {
                PrintPrediction(k, data[k].Value, prediction);
                k++;
            }
            //Index Data    Anomaly AnomalyScore    Mag ExpectedValue   BoundaryUnit UpperBoundary   LowerBoundary
            //0       5.00    0               0.00    0.21            5.00            5.00            5.01            4.99
            //25      5.00    0               0.00    0.19            5.00            5.00            5.01            4.99
        }

        private static void PrintPrediction(int idx, double value, SrCnnAnomalyDetection prediction) 
            {
            if (prediction.Prediction[0] == 0)
                Console.ForegroundColor = ConsoleColor.White;
            else
                Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("{0}\t{1:0.00}\t{2}\t\t{3:0.00}\t{4:0.00}\t\t{5:0.00}\t\t{6:0.00}\t\t{7:0.00}\t\t{8:0.00}",
                idx, value, prediction.Prediction[0], prediction.Prediction[1], prediction.Prediction[2],
                prediction.Prediction[3], prediction.Prediction[4], prediction.Prediction[5], prediction.Prediction[6]);
            }
        private class TimeSeriesData
        {
            public double Value { get; set; }
        }

        private class SrCnnAnomalyDetection
        {
            [VectorType]
            public double[] Prediction { get; set; }
        }
    
    }
}
