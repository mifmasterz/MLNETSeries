using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Globalization;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
namespace DiabetesAnalysisApp
{
    public class DiabetesRecord
    {
        public DateTime TimeStamp { set; get; }
        public string Code { get; set; }
        public float Data { get; set; }
    }
    public class DiabetesRecordPrediction
    {
        //vector to hold alert,score,p-value values
        [VectorType(3)]
        public double[] Prediction { get; set; }
    }
    class Program
    { 
        static MLContext mlContext; 
        static void Main(string[] args)
        {
            var TrainData = new List<DiabetesRecord>();
            var DataFolder = @"C:\Users\gravi\Downloads\diabetes-data\Diabetes-Data";
            var Files = Directory.GetFiles(DataFolder, "*");
            foreach (var filePath in Files)
            {
                foreach (var line in File.ReadAllLines(filePath))
                {
                    var cols = line.Split('\t');
                    CultureInfo ci = new CultureInfo("id-ID");
                    var DateStr = $"{cols[0]} {cols[1]}";//Convert.ToDateTime($"{cols[0]} {cols[1]}", ci );
                    var len = DateStr.Length;
                    float dataValue = 0;
                    float.TryParse(cols[3], out dataValue);
                    if (len >= 15)
                    {
                        DateTime.TryParse(DateStr, out DateTime dt);
                        if (dt.Year > DateTime.MinValue.Year)
                            TrainData.Add(new DiabetesRecord() { TimeStamp = dt, Code = cols[2], Data = dataValue });
                    }
                }
            }
            HashSet<string> CodeIn = new HashSet<string> ();
            CodeIn.Add("48");
            CodeIn.Add("57");CodeIn.Add("58");CodeIn.Add("59");CodeIn.Add("60");CodeIn.Add("61");CodeIn.Add("62");CodeIn.Add("63");CodeIn.Add("64");
            TrainData = TrainData.Where(x=> CodeIn.Contains(x.Code)).OrderBy(a => a.TimeStamp).ToList();
            Console.WriteLine($"Total data : {TrainData.Count}");
            // Create MLContext
            mlContext = new MLContext();

            //Load Data
            IDataView data = mlContext.Data.LoadFromEnumerable<DiabetesRecord>(TrainData);
            //assign the Number of records in dataset file to cosntant variable
            const int size = 36;
            //STEP 1: Create Esimtator   
            DetectSpike(size, data);
            //To detect persistent change in the pattern
            DetectChangepoint(size, data);

            Console.WriteLine("=============== End of process, hit any key to finish ===============");

            Console.ReadLine();

        }

        static void DetectSpike(int size,IDataView dataView)
        {
           Console.WriteLine("===============Detect temporary changes in pattern===============");

            //STEP 1: Create Esimtator   
            var estimator = mlContext.Transforms.DetectIidSpike(outputColumnName: nameof(DiabetesRecordPrediction.Prediction), inputColumnName: nameof(DiabetesRecord.Data),confidence: 95, pvalueHistoryLength: size / 4);

            //STEP 2:The Transformed Model.
            //In IID Spike detection, we don't need to do training, we just need to do transformation. 
            //As you are not training the model, there is no need to load IDataView with real data, you just need schema of data.
            //So create empty data view and pass to Fit() method. 
            ITransformer tansformedModel = estimator.Fit(CreateEmptyDataView());

            //STEP 3: Use/test model
            //Apply data transformation to create predictions.
            IDataView transformedData = tansformedModel.Transform(dataView);
            var predictions = mlContext.Data.CreateEnumerable<DiabetesRecordPrediction>(transformedData, reuseRowObject: false);
                      
            Console.WriteLine("Alert\tScore\tP-Value");
            foreach (var p in predictions)
            {
                if (p.Prediction[0] == 1)
                {
                    Console.BackgroundColor = ConsoleColor.DarkYellow;
                    Console.ForegroundColor = ConsoleColor.Black;
                }
                Console.WriteLine("{0}\t{1:0.00}\t{2:0.00}", p.Prediction[0], p.Prediction[1], p.Prediction[2]);
                Console.ResetColor();
            }
            
        }
        static void DetectChangepoint(int size, IDataView dataView)
        {
          Console.WriteLine("===============Detect Persistent changes in pattern===============");

          //STEP 1: Setup transformations using DetectIidChangePoint
          var estimator = mlContext.Transforms.DetectIidChangePoint(outputColumnName: nameof(DiabetesRecordPrediction.Prediction), inputColumnName: nameof(DiabetesRecord.Data), confidence: 95, changeHistoryLength: size / 4);

          //STEP 2:The Transformed Model.
          //In IID Change point detection, we don't need need to do training, we just need to do transformation. 
          //As you are not training the model, there is no need to load IDataView with real data, you just need schema of data.
          //So create empty data view and pass to Fit() method. 
          ITransformer tansformedModel = estimator.Fit(CreateEmptyDataView());

          //STEP 3: Use/test model
          //Apply data transformation to create predictions.
          IDataView transformedData = tansformedModel.Transform(dataView);
          var predictions = mlContext.Data.CreateEnumerable<DiabetesRecordPrediction>(transformedData, reuseRowObject: false);
                       
          Console.WriteLine($"{nameof(DiabetesRecordPrediction.Prediction)} column obtained post-transformation.");
          Console.WriteLine("Alert\tScore\tP-Value\tMartingale value");
            
          foreach(var p in predictions)
          {
             if (p.Prediction[0] == 1)
             {
                 Console.WriteLine("{0}\t{1:0.00}\t{2:0.00}\t{3:0.00}  <-- alert is on, predicted changepoint", p.Prediction[0], p.Prediction[1], p.Prediction[2], p.Prediction[3]);
             }
             else
             { 
                 Console.WriteLine("{0}\t{1:0.00}\t{2:0.00}\t{3:0.00}",  p.Prediction[0], p.Prediction[1], p.Prediction[2], p.Prediction[3]);                  
             }            
          }
          Console.WriteLine("");
        }
        private static IDataView CreateEmptyDataView()
        {
            //Create empty DataView. We just need the schema to call fit()
            IEnumerable<DiabetesRecord> enumerableData = new List<DiabetesRecord>();
            var dv = mlContext.Data.LoadFromEnumerable(enumerableData);
            return dv;
        }
    }
}
