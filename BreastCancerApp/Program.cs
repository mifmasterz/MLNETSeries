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
        static string TrainingFile = "breast-cancer-wisconsin.data";
        static void Main(string[] args)
        {
            var DataDir = new DirectoryInfo(GetAbsolutePath(@"..\..\..\Data"));
            //Create MLContext
            MLContext mlContext = new MLContext();

            //Load Data File
            var Lines = File.ReadAllLines(Path.Combine(DataDir.FullName, TrainingFile));
            var ListData = new List<BreastCancerData>();
            int counter = 0;
            Lines.ToList().ForEach(x => {
                counter++;
                //skip header
                if (counter > 1)
                {
                    var Cols = x.Split(',');
                    ListData.Add(new BreastCancerData() { SampleNo = float.Parse(Cols[0]), ClumpThickness = float.Parse(Cols[1]), UniformityOfCellSize = float.Parse(Cols[2]), UniformityOfCellShape = float.Parse(Cols[3]), MarginalAdhesion = float.Parse(Cols[4]), SingleEpithelialCellSize = float.Parse(Cols[5]), BareNuclei = float.Parse(Cols[6] == "?" ? "0" : Cols[6]), BlandChromatin = float.Parse(Cols[7]), NormalNucleoli = float.Parse(Cols[8]), Mitoses = float.Parse(Cols[9]), ClassCategory = int.Parse(Cols[10]), IsBenign = Cols[10] == "4" ? false:true });
                }
            });

            IDataView allData = mlContext.Data.LoadFromEnumerable<BreastCancerData>(ListData);
            
            DataOperationsCatalog.TrainTestData dataSplit = mlContext.Data.TrainTestSplit(allData, testFraction: 0.2);
            IDataView trainData = dataSplit.TrainSet;
            IDataView testData = dataSplit.TestSet;

            // Data process configuration with pipeline data transformations 
            var dataPrepTransform = mlContext.Transforms.CopyColumns("Label", "IsBenign")
                                      .Append(mlContext.Transforms.IndicateMissingValues(new[] { new InputOutputColumnPair("BareNuclei_MissingIndicator", "BareNuclei") }))
                                      .Append(mlContext.Transforms.Conversion.ConvertType(new[] { new InputOutputColumnPair("BareNuclei_MissingIndicator", "BareNuclei_MissingIndicator") }))
                                      .Append(mlContext.Transforms.ReplaceMissingValues(new[] { new InputOutputColumnPair("BareNuclei", "BareNuclei") }))
                                      .Append(mlContext.Transforms.Concatenate("Features", new[] { "BareNuclei_MissingIndicator", "BareNuclei", "ClumpThickness", "UniformityOfCellSize", "UniformityOfCellShape", "MarginalAdhesion", "SingleEpithelialCellSize", "BlandChromatin", "NormalNucleoli", "Mitoses" }))
                                      .Append(mlContext.Transforms.NormalizeMinMax("Features", "Features"))
                                      .AppendCacheCheckpoint(mlContext);

            // Create data prep transformer
            ITransformer dataPrepTransformer = dataPrepTransform.Fit(trainData);
            IDataView transformedTrainingData = dataPrepTransformer.Transform(trainData);

            var SdcaEstimator = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features");

            // Build machine learning model
            var trainedModel = dataPrepTransformer.Append(SdcaEstimator.Fit(transformedTrainingData));

            // Apply data prep transformer to test data 
            IDataView testDataPredictions = trainedModel.Transform(testData);

            // Measure trained model performance
            // Extract model metrics and get eval params
            var metrics = mlContext.BinaryClassification.Evaluate(testDataPredictions);
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
            var predEngine = mlContext.Model.CreatePredictionEngine<BreastCancerData, PredictionBreastCancerData>(mlModel);

            // Create sample data to do a single prediction with it 
            /*
            //jinak
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
                
            };*/
            //ganas
            BreastCancerData sampleData = new BreastCancerData()
            {
                SampleNo = 0,
                ClumpThickness = 8,
                UniformityOfCellSize = 10,
                UniformityOfCellShape = 10,
                MarginalAdhesion = 8,
                SingleEpithelialCellSize = 7,
                BareNuclei = 10,
                BlandChromatin = 9,
                NormalNucleoli = 7,
                Mitoses = 1

            };
            // Try a single prediction
            PredictionBreastCancerData predictionResult = predEngine.Predict(sampleData);
            Console.WriteLine($"Single Prediction --> Predicted:  { (predictionResult.Prediction ? "Jinak":"Ganas") }");
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
