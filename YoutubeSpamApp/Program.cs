using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML.Trainers;
using System.IO;
namespace YoutubeSpamApp
{
    class Program
    {
        static string TRAIN_DATA_FILEPATH = @"C:\Users\gravi\Documents\Spam\Youtube01-Psy.csv";
        static void Main(string[] args)
        {
            var modelRelativePath = "MLModel.zip";
            MLContext mlContext = new MLContext(seed: 1);
            // Load Data
            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<ModelInput>(
                                            path: TRAIN_DATA_FILEPATH,
                                            hasHeader: true,
                                            separatorChar: ',',
                                            allowQuoting: true,
                                            allowSparse: false);

            // Data process configuration with pipeline data transformations 
            var dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText("COMMENT_ID_tf", "COMMENT_ID")
                                      .Append(mlContext.Transforms.Text.FeaturizeText("CONTENT_tf", "CONTENT"))
                                      .Append(mlContext.Transforms.Concatenate("Features", new[] { "COMMENT_ID_tf", "CONTENT_tf" }))
                                      .Append(mlContext.Transforms.NormalizeMinMax("Features", "Features"))
                                      .AppendCacheCheckpoint(mlContext);

            // Set the training algorithm 
            var trainer = mlContext.BinaryClassification.Trainers.SymbolicSgdLogisticRegression(new SymbolicSgdLogisticRegressionBinaryTrainer.Options() { NumberOfIterations = 20, LearningRate = 0.01f, L2Regularization = 0f, LabelColumnName = "CLASS", FeatureColumnName = "Features" });
            var trainingPipeline = dataProcessPipeline.Append(trainer);
            ITransformer model = trainingPipeline.Fit(trainingDataView);
            var crossValidationResults = mlContext.BinaryClassification.CrossValidateNonCalibrated(trainingDataView, trainingPipeline, numberOfFolds: 5, labelColumnName: "CLASS");
            PrintBinaryClassificationFoldsAverageMetrics(crossValidationResults);
            mlContext.Model.Save(model, trainingDataView.Schema, GetAbsolutePath(modelRelativePath));

            ITransformer mlModel = mlContext.Model.Load(GetAbsolutePath(modelRelativePath), out DataViewSchema inputSchema);
            var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);

            // Create sample data to do a single prediction with it 
           
            ModelInput sampleData = mlContext.Data.CreateEnumerable<ModelInput>(trainingDataView, false)
                                                                        .First();
            // Try a single prediction
            ModelOutput predictionResult = predEngine.Predict(sampleData);

            Console.WriteLine($"Single Prediction --> Actual value: {sampleData.CLASS} | Predicted value: {predictionResult.Prediction}");
            //Console.WriteLine("Hello World!");
            Console.ReadLine();
        }
         public static double CalculateStandardDeviation(IEnumerable<double> values)
        {
            double average = values.Average();
            double sumOfSquaresOfDifferences = values.Select(val => (val - average) * (val - average)).Sum();
            double standardDeviation = Math.Sqrt(sumOfSquaresOfDifferences / (values.Count() - 1));
            return standardDeviation;
        }

        public static double CalculateConfidenceInterval95(IEnumerable<double> values)
        {
            double confidenceInterval95 = 1.96 * CalculateStandardDeviation(values) / Math.Sqrt((values.Count() - 1));
            return confidenceInterval95;
        }
        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }
        public static void PrintBinaryClassificationFoldsAverageMetrics(IEnumerable<TrainCatalogBase.CrossValidationResult<BinaryClassificationMetrics>> crossValResults)
        {
            var metricsInMultipleFolds = crossValResults.Select(r => r.Metrics);

            var AccuracyValues = metricsInMultipleFolds.Select(m => m.Accuracy);
            var AccuracyAverage = AccuracyValues.Average();
            var AccuraciesStdDeviation = CalculateStandardDeviation(AccuracyValues);
            var AccuraciesConfidenceInterval95 = CalculateConfidenceInterval95(AccuracyValues);


            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Binary Classification model      ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       Average Accuracy:    {AccuracyAverage:0.###}  - Standard deviation: ({AccuraciesStdDeviation:#.###})  - Confidence Interval 95%: ({AccuraciesConfidenceInterval95:#.###})");
            Console.WriteLine($"*************************************************************************************************************");
        }

    }

    public class ModelOutput
    {
        // ColumnName attribute is used to change the column name from
        // its default value, which is the name of the field.
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        public float Score { get; set; }
    }
    public class ModelInput
    {
        [ColumnName("COMMENT_ID"), LoadColumn(0)]
        public string COMMENT_ID { get; set; }


        [ColumnName("AUTHOR"), LoadColumn(1)]
        public string AUTHOR { get; set; }


        [ColumnName("DATE"), LoadColumn(2)]
        public string DATE { get; set; }


        [ColumnName("CONTENT"), LoadColumn(3)]
        public string CONTENT { get; set; }


        [ColumnName("CLASS"), LoadColumn(4)]
        public bool CLASS { get; set; }


    }
}
