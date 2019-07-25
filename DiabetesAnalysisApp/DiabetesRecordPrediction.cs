using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace DiabetesAnalysisApp
{
    public class DiabetesRecordPrediction
    {
        //vector to hold alert,score,p-value values
        [VectorType(3)]
        public double[] Prediction { get; set; }
    }
}
