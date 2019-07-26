using System;
using System.Collections.Generic;
using System.Text;

namespace AirBnBApp
{
    public class ListingPrediction
    {
        public bool PredictedLabel;

        public float Probability;

        public string neighbourhood { get; set; }
       
        public string room_type { get; set; }
     
        public float price { get; set; }
     
        public float minimum_nights { get; set; }
      
        public float availability_365 { get; set; }
    }
}
