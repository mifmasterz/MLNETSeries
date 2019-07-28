using System;
using System.Collections.Generic;
using System.Text;

namespace TransferLearningImageClassification
{
    public class ImagePrediction : ImageData
    {
        public float[] Score;

        public string PredictedLabelValue;
    }
}
