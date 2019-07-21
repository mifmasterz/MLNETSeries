using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System.Collections.Generic;
using System.Linq;
using System.IO;

namespace SeedsClusteringApp
{
    class Program
    {
        static void Main(string[] args)
        {
            var DataPath = @"C:\Users\gravi\Downloads\seeds_dataset.csv";
            var modelRelativePath = "MLModel.zip";
            //Create the MLContext to share across components for deterministic results
            MLContext mlContext = new MLContext(seed: 1);  //Seed set to any number so you have a deterministic environment

            // STEP 1: Common data loading configuration
            IDataView fullData = mlContext.Data.LoadFromTextFile<SeedsData>(path: DataPath,
                                                            hasHeader: true, allowQuoting: true,
                                                            separatorChar: ',');

            //Split dataset in two parts: TrainingDataset (80%) and TestDataset (20%)
            DataOperationsCatalog.TrainTestData trainTestData = mlContext.Data.TrainTestSplit(fullData, testFraction: 0.2);
            var trainingDataView = trainTestData.TrainSet;
            var testingDataView = trainTestData.TestSet;

            //STEP 2: Process data transformations in pipeline
            var dataProcessPipeline = mlContext.Transforms.Concatenate("Features", nameof(SeedsData.area), nameof(SeedsData.perimeter), nameof(SeedsData.compactness), nameof(SeedsData.length_kernel)
            , nameof(SeedsData.width_kernel), nameof(SeedsData.asymmetry_coef), nameof(SeedsData.length_kernel_grove));

            // STEP 3: Create and train the model     
            var trainer = mlContext.Clustering.Trainers.KMeans(featureColumnName: "Features", numberOfClusters: 3);
            var trainingPipeline = dataProcessPipeline.Append(trainer);
            var trainedModel = trainingPipeline.Fit(trainingDataView);
            mlContext.Model.Save(trainedModel, trainingDataView.Schema, GetAbsolutePath(modelRelativePath));
            ITransformer mlModel = mlContext.Model.Load(GetAbsolutePath(modelRelativePath), out DataViewSchema inputSchema);
            // Test with one sample text 
            var sampleData = mlContext.Data.CreateEnumerable<SeedsData>(trainingDataView, false)
                                                              .First();
            var sampleData2 = mlContext.Data.CreateEnumerable<SeedsData>(trainingDataView, false)
                                                              .Skip(1).First();

            // Create prediction engine related to the loaded trained model
            var predEngine = mlContext.Model.CreatePredictionEngine<SeedsData, SeedsDataPrediction>(mlModel);

            //Score
            var resultprediction = predEngine.Predict(sampleData);
            var resultprediction2 = predEngine.Predict(sampleData2);
            Console.WriteLine($"Cluster assigned for type 1 :" + resultprediction.SelectedClusterId);
            Console.WriteLine($"Cluster assigned for type 1 :" + resultprediction2.SelectedClusterId);
            Console.ReadLine();
        }
        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }
    }
    public class SeedsDataPrediction

    {

        [ColumnName("PredictedLabel")]

        public uint SelectedClusterId;



        [ColumnName("Score")]

        public float[] Distance;

    }
    public class SeedsData
    {
        //area	perimeter	compactness	length_kernel	width_kernel	asymmetry_coef	length_kernel_grove	type
        [LoadColumn(0)]
        public float area { get; set; }
        [LoadColumn(1)]
        public float perimeter { get; set; }
        [LoadColumn(2)]
        public float compactness { get; set; }
        [LoadColumn(3)]
        public float length_kernel { get; set; }
        [LoadColumn(4)]
        public float width_kernel { get; set; }
        [LoadColumn(5)]
        public float asymmetry_coef { get; set; }
        [LoadColumn(6)]
        public float length_kernel_grove { get; set; }
        [LoadColumn(7)]
        public int type { get; set; }

    }
}
