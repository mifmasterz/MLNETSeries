using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Image;
using static Microsoft.ML.Transforms.Image.ImagePixelExtractingEstimator;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Onnx;

namespace TransferLearningImageClassification
{
    public class ImagePrediction : ImageData
    {
        public float[] Score;

        public string PredictedLabelValue;
    }
    public class ImageData
    {
        [LoadColumn(0)]
        public string ImagePath;

        [LoadColumn(1)]
        public string Label;
    }
    class Program
    {
        static readonly string _assetsPath = GetAbsolutePath("../../../assets/");
        static readonly string _trainTagsTsv = Path.Combine(_assetsPath, "inputs-train", "data", "tags.tsv");
        static readonly string _predictImageListTsv = Path.Combine(_assetsPath, "inputs-predict", "data", "image_list.tsv");
        static readonly string _trainImagesFolder = Path.Combine(_assetsPath, "inputs-train", "data");
        static readonly string _predictImagesFolder = Path.Combine(_assetsPath, "inputs-predict", "data");
        static readonly string _predictSingleImage = Path.Combine(_assetsPath, "inputs-predict-single", "data", "toaster3.jpg");
        static readonly string _inceptionPb = Path.Combine(_assetsPath, "inputs-train", "inception", "mobilenetv2-1.0.onnx");//Path.Combine(_assetsPath, "inputs-train", "inception", "tensorflow_inception_graph.pb");
        
        static readonly string _inputImageClassifierZip = Path.Combine(_assetsPath, "inputs-predict", "imageClassifier.zip");
        static readonly string _outputImageClassifierZip = Path.Combine(_assetsPath, "outputs", "imageClassifier.zip");
        private static string LabelTokey = nameof(LabelTokey);
        private static string PredictedLabelValue = nameof(PredictedLabelValue);
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext(seed: 1);
            var model = ReuseAndTuneInceptionModel(mlContext,_trainTagsTsv,_trainImagesFolder,_inceptionPb,_outputImageClassifierZip);
            ClassifyImages(mlContext, _predictImageListTsv, _predictImagesFolder, _outputImageClassifierZip, model);
            ClassifySingleImage(mlContext, _predictSingleImage, _outputImageClassifierZip, model);
            Console.WriteLine("Hello World!");
        }
        public static IEnumerable<ImageData> ReadFromTsv(string file, string folder)
        {
            return File.ReadAllLines(file)
             .Select(line => line.Split('\t'))
             .Select(line => new ImageData()
             {
                 ImagePath = Path.Combine(folder, line[0])
             });

        }
        public static void ClassifySingleImage(MLContext mlContext, string imagePath, string outputModelLocation, ITransformer model)
        {
            var imageData = new ImageData()
            {
                ImagePath = imagePath
            };
            // Make prediction function (input = ImageData, output = ImagePrediction)
            var predictor = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
            var prediction = predictor.Predict(imageData);
            Console.WriteLine($"Image: {Path.GetFileName(imageData.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score.Max()} ");
        }
        public static void ClassifyImages(MLContext mlContext, string dataLocation, string imagesFolder, string outputModelLocation, ITransformer model)
        {
            var imageData = ReadFromTsv(dataLocation, imagesFolder);
            var imageDataView = mlContext.Data.LoadFromEnumerable<ImageData>(imageData);
            var predictions = model.Transform(imageDataView);
            var imagePredictionData = mlContext.Data.CreateEnumerable<ImagePrediction>(predictions, false, true);
            DisplayResults(imagePredictionData);
        }
        public static ITransformer ReuseAndTuneInceptionModel(MLContext mlContext, string dataLocation, string imagesFolder, string inputModelLocation, string outputModelLocation)
        {
            /*
             //--mobilenet--
             var estimator = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: LabelTokey, inputColumnName: "Label").Append(mlContext.Transforms.LoadImages(outputColumnName: "input", imageFolder: _trainImagesFolder, inputColumnName: nameof(ImageData.ImagePath)))
            .Append(mlContext.Transforms.ResizeImages(outputColumnName: "input", imageWidth: InceptionSettings.ImageWidth, imageHeight: InceptionSettings.ImageHeight, inputColumnName: "input"))
            .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input", colorsToExtract: ColorBits.Rgb, interleavePixelColors: true, outputAsFloatArray: true, offsetImage: 128f, scaleImage: 1 / 255f))
            //.Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input", interleavePixelColors: InceptionSettings.ChannelsLast, offsetImage: InceptionSettings.Mean))
            .Append(mlContext.Model.LoadTensorFlowModel(inputModelLocation)
            .ScoreTensorFlowModel(outputColumnNames: new[] { "MobilenetV2/Predictions/Reshape_1" }, inputColumnNames: new[] { "input" }, addBatchDimensionInput: true))
            //.ScoreTensorFlowModel(outputColumnNames: new[] { "softmax2_pre_activation" }, inputColumnNames: new[] { "input" }, addBatchDimensionInput: true))
            .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: LabelTokey, featureColumnName: "MobilenetV2/Predictions/Reshape_1"))//softmax2_pre_activation
            .Append(mlContext.Transforms.Conversion.MapKeyToValue(PredictedLabelValue, "PredictedLabel"))
            .AppendCacheCheckpoint(mlContext);
             */
            /*
             //inception
             var estimator = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: LabelTokey, inputColumnName: "Label").Append(mlContext.Transforms.LoadImages(outputColumnName: "input", imageFolder: _trainImagesFolder, inputColumnName: nameof(ImageData.ImagePath)))
           .Append(mlContext.Transforms.ResizeImages(outputColumnName: "input", imageWidth: InceptionSettings.ImageWidth, imageHeight: InceptionSettings.ImageHeight, inputColumnName: "input"))
           .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input", interleavePixelColors: InceptionSettings.ChannelsLast, offsetImage: InceptionSettings.Mean))
           .Append(mlContext.Model.LoadTensorFlowModel(inputModelLocation)
           .ScoreTensorFlowModel(outputColumnNames: new[] { "softmax2_pre_activation" }, inputColumnNames: new[] { "input" }, addBatchDimensionInput: true))
           .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: LabelTokey, featureColumnName: "MobilenetV2/Predictions/Reshape_1"))//softmax2_pre_activation
           .Append(mlContext.Transforms.Conversion.MapKeyToValue(PredictedLabelValue, "PredictedLabel"))
           .AppendCacheCheckpoint(mlContext);
            */
            /*
             //onnx
             var data = mlContext.Data.LoadFromTextFile<ImageData>(path: dataLocation, hasHeader: false);
            var estimator = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: LabelTokey, inputColumnName: "Label")
            .Append(mlContext.Transforms.LoadImages(outputColumnName: "input", imageFolder: _trainImagesFolder, inputColumnName: nameof(ImageData.ImagePath)))
            .Append(mlContext.Transforms.ResizeImages(outputColumnName: "ImageResized", imageWidth: InceptionSettings.ImageWidth, imageHeight: InceptionSettings.ImageHeight, inputColumnName: "input"))
            .Append(mlContext.Transforms.ExtractPixels(outputColumnName:"Red", inputColumnName: "ImageResized",
                            colorsToExtract: ColorBits.Red, offsetImage: 0.485f * 255, scaleImage: 1 / (0.229f * 255)))                    
            .Append(mlContext.Transforms.ExtractPixels(outputColumnName:"Green", inputColumnName:"ImageResized",
                            colorsToExtract: ColorBits.Green, offsetImage: 0.456f * 255, scaleImage: 1 / (0.224f * 255)))                    
            .Append(mlContext.Transforms.ExtractPixels(outputColumnName:"Blue",inputColumnName: "ImageResized",
                            colorsToExtract: ColorBits.Blue, offsetImage: 0.406f * 255, scaleImage: 1 / (0.225f * 255)))                    
            .Append(mlContext.Transforms.Concatenate("data", "Red", "Green", "Blue"))                    
            .Append(mlContext.Transforms.ApplyOnnxModel(modelFile: inputModelLocation,inputColumnName:"data",outputColumnName: "mobilenetv20_output_flatten0_reshape0"))
            .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: LabelTokey, featureColumnName: "mobilenetv20_output_flatten0_reshape0"))
            .Append(mlContext.Transforms.Conversion.MapKeyToValue(PredictedLabelValue, "PredictedLabel"))
            .AppendCacheCheckpoint(mlContext);
             */
            var data = mlContext.Data.LoadFromTextFile<ImageData>(path: dataLocation, hasHeader: false);
            var estimator = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: LabelTokey, inputColumnName: "Label")
            .Append(mlContext.Transforms.LoadImages(outputColumnName: "input", imageFolder: _trainImagesFolder, inputColumnName: nameof(ImageData.ImagePath)))
            .Append(mlContext.Transforms.ResizeImages(outputColumnName: "ImageResized", imageWidth: InceptionSettings.ImageWidth, imageHeight: InceptionSettings.ImageHeight, inputColumnName: "input"))
            .Append(mlContext.Transforms.ExtractPixels(outputColumnName:"Red", inputColumnName: "ImageResized",
                            colorsToExtract: ColorBits.Red, offsetImage: 0.485f * 255, scaleImage: 1 / (0.229f * 255)))                    
            .Append(mlContext.Transforms.ExtractPixels(outputColumnName:"Green", inputColumnName:"ImageResized",
                            colorsToExtract: ColorBits.Green, offsetImage: 0.456f * 255, scaleImage: 1 / (0.224f * 255)))                    
            .Append(mlContext.Transforms.ExtractPixels(outputColumnName:"Blue",inputColumnName: "ImageResized",
                            colorsToExtract: ColorBits.Blue, offsetImage: 0.406f * 255, scaleImage: 1 / (0.225f * 255)))                    
            .Append(mlContext.Transforms.Concatenate("data", "Red", "Green", "Blue"))                    
            .Append(mlContext.Transforms.ApplyOnnxModel(modelFile: inputModelLocation,inputColumnName:"data",outputColumnName: "mobilenetv20_output_flatten0_reshape0"))
            .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: LabelTokey, featureColumnName: "mobilenetv20_output_flatten0_reshape0"))
            .Append(mlContext.Transforms.Conversion.MapKeyToValue(PredictedLabelValue, "PredictedLabel"))
            .AppendCacheCheckpoint(mlContext);
            ITransformer model = estimator.Fit(data);
            var predictions = model.Transform(data);
            var imageData = mlContext.Data.CreateEnumerable<ImageData>(data, false, true);
            var imagePredictionData = mlContext.Data.CreateEnumerable<ImagePrediction>(predictions, false, true);
            DisplayResults(imagePredictionData);
            var multiclassContext = mlContext.MulticlassClassification;
            var metrics = multiclassContext.Evaluate(predictions, labelColumnName: LabelTokey, predictedLabelColumnName: "PredictedLabel");
            Console.WriteLine($"LogLoss is: {metrics.LogLoss}");
            Console.WriteLine($"PerClassLogLoss is: {String.Join(" , ", metrics.PerClassLogLoss.Select(c => c.ToString()))}");
            mlContext.Model.Save(model,data.Schema,outputModelLocation);
            return model;

        }
        private static void DisplayResults(IEnumerable<ImagePrediction> imagePredictionData)
        {
            foreach (ImagePrediction prediction in imagePredictionData)
            {
                Console.WriteLine($"Image: {Path.GetFileName(prediction.ImagePath)} predicted as: {prediction.PredictedLabelValue} with score: {prediction.Score.Max()} ");
            }
        }
        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;
            string fullPath = Path.Combine(assemblyFolderPath, relativePath);
            return fullPath;
        }
        private struct InceptionSettings
        {
            public const int ImageHeight = 224;
            public const int ImageWidth = 224;
            public const float Mean = 117;
            public const float Scale = 1;
            public const bool ChannelsLast = true;
        }
    }
}
