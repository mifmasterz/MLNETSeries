using System;
using Microsoft.ML;
using System.IO;
using Microsoft.ML.Trainers;
using Microsoft.ML.Data;

using System.Linq;
using System.Collections.Generic;
namespace BreastCancerApp
{
    class Program
    {
        static void Main(string[] args)
        {
            //Create MLContext
            MLContext mlContext = new MLContext();
           
            //Load Data File
            //
            IDataView allData = mlContext.Data.LoadFromTextFile<BreastCancerData>(GetAbsolutePath("../../../breast-cancer-wisconsin.data"), separatorChar: ',', hasHeader: true);
            
            DataOperationsCatalog.TrainTestData dataSplit = mlContext.Data.TrainTestSplit(allData, testFraction: 0.2);
            IDataView trainData = dataSplit.TrainSet;
            IDataView testData = dataSplit.TestSet;

            /*
                        var dataPrepTransform = mlContext.Transforms.Concatenate("Features", nameof(BreastCancerData.ClumpThickness), nameof(BreastCancerData.UniformityOfCellSize), nameof(BreastCancerData.UniformityOfCellShape), nameof(BreastCancerData.MarginalAdhesion), nameof(BreastCancerData.SingleEpithelialCellSize)
                        , nameof(BreastCancerData.BareNuclei), nameof(BreastCancerData.BlandChromatin), nameof(BreastCancerData.NormalNucleoli), nameof(BreastCancerData.Mitoses)).Append(
                            mlContext.Transforms.Conversion.MapValueToKey("Label", "ClassCategory")
                        );
            */
            // Data process configuration with pipeline data transformations 
            var dataPrepTransform = mlContext.Transforms.Conversion.MapValueToKey("Label", "ClassCategory")
                                      .Append(mlContext.Transforms.IndicateMissingValues(new[] { new InputOutputColumnPair("BareNuclei_MissingIndicator", "BareNuclei") }))
                                      .Append(mlContext.Transforms.Conversion.ConvertType(new[] { new InputOutputColumnPair("BareNuclei_MissingIndicator", "BareNuclei_MissingIndicator") }))
                                      .Append(mlContext.Transforms.ReplaceMissingValues(new[] { new InputOutputColumnPair("BareNuclei", "BareNuclei") }))
                                      .Append(mlContext.Transforms.Concatenate("Features", new[] { "BareNuclei_MissingIndicator", "BareNuclei", "ClumpThickness", "UniformityOfCellSize", "UniformityOfCellShape", "MarginalAdhesion", "SingleEpithelialCellSize", "BlandChromatin", "NormalNucleoli", "Mitoses" }))
                                      .Append(mlContext.Transforms.NormalizeMinMax("Features", "Features"))
                                      .AppendCacheCheckpoint(mlContext);



            // Create data prep transformer
            ITransformer dataPrepTransformer = dataPrepTransform.Fit(trainData);
            IDataView transformedTrainingData = dataPrepTransformer.Transform(trainData);

            /* 
            // Set the training algorithm 
            var trainer = mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.SgdCalibrated(new SgdCalibratedTrainer.Options() { L2Regularization = 1E-05f, ConvergenceTolerance = 0.01f, NumberOfIterations = 5, Shuffle = false, LabelColumnName = "Label", FeatureColumnName = "Features" }), labelColumnName: "Label")
                                       .Append(mlContext.Transforms.Conversion.MapKeyToValue("Label", "Label"));
            var trainedModel = trainer.Fit(transformedTrainingData);
            */
            //var trainingPipeline = dataPrepTransform.Append(trainer);
            var SdcaEstimator = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features").Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: "Label", inputColumnName: "Label"));

            // Build machine learning model
            var trainedModel = dataPrepTransformer.Append(SdcaEstimator.Fit(transformedTrainingData));

            // Measure trained model performance
            // Apply data prep transformer to test data and train
            IDataView testDataPredictions = trainedModel.Transform(testData);

            // Use trained model to make inferences on test data
            //IDataView testDataPredictions = trainedModel.Transform(transformedTestData);

            // Extract model metrics and get eval params
            var trainedModelMetrics = mlContext.MulticlassClassification.Evaluate(testDataPredictions);
            Console.WriteLine(trainedModelMetrics.ConfusionMatrix.ToString());
            System.Console.WriteLine($"LogLoss : {trainedModelMetrics.LogLoss}, Macro Accuracy : {trainedModelMetrics.MacroAccuracy}, Micro Accuracy : {trainedModelMetrics.MicroAccuracy} ");

            var modelRelativePath = GetAbsolutePath("MLModel.zip");

            mlContext.Model.Save(trainedModel, trainData.Schema, GetAbsolutePath(modelRelativePath));
            Console.WriteLine("The model is saved to {0}", GetAbsolutePath(modelRelativePath));

            ITransformer mlModel = mlContext.Model.Load(GetAbsolutePath(modelRelativePath), out DataViewSchema inputSchema);
            var predEngine = mlContext.Model.CreatePredictionEngine<BreastCancerData, PredictionBreastData>(mlModel);
            VBuffer<float> keys = default;

            predEngine.OutputSchema["PredictedLabel"].GetKeyValues(ref keys);

            var labelsArray = keys.DenseValues().ToArray();

            Dictionary<float, string> CancerTypes = new Dictionary<float, string>();

            CancerTypes.Add(2, "Jinak");

            CancerTypes.Add(4, "Ganas");

            // Create sample data to do a single prediction with it 
            BreastCancerData sampleData = new BreastCancerData()
            {
                SampleNo = 0,
                ClumpThickness = 5,
                UniformityOfCellSize = 1,
                UniformityOfCellShape = 1,
                MarginalAdhesion = 1,
                SingleEpithelialCellSize = 2,
                BareNuclei = 1,
                BlandChromatin = 3,
                NormalNucleoli = 1,
                Mitoses = 1
                
            };
            //8,10,10,8,7,10,9,7,1 = 4
            //5,1,1,1,2,1,3,1,1
            // Try a single prediction
            PredictionBreastData predictionResult = predEngine.Predict(sampleData);
            var Maks = Array.IndexOf(predictionResult.Score, predictionResult.Score.Max()); 
            Console.WriteLine($"Single Prediction --> Predicted label and score:  {CancerTypes[labelsArray[Maks]]}: {predictionResult.Score[Maks]:0.####}");
            //Console.WriteLine($"Single Prediction --> Actual value: {sampleData.ClassCategory} | Predicted value: {predictionResult.Prediction} | Predicted scores: [{String.Join(",", predictionResult.Score)}]");

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
