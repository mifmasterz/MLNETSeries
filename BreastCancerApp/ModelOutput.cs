using System;
using Microsoft.ML.Data;

namespace BreastCancerApp
{   
    public class PredictionBreastData : BreastCancerData
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }
    }
}
