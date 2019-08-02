using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace TrafficVolumeAutoMLAPI
{
    class TrafficData
    {
        [LoadColumn(0), ColumnName("holiday")]
        public string holiday { get; set; }

        [LoadColumn(1), ColumnName("temp")]

        public float temp { get; set; }

        [LoadColumn(2), ColumnName("rain")]

        public float rain { get; set; }

        [LoadColumn(3), ColumnName("snow")]

        public float snow { get; set; }

        [LoadColumn(4), ColumnName("cloud")]

        public float cloud { get; set; }

        [LoadColumn(5), ColumnName("weather")]

        public string weather { get; set; }

        [LoadColumn(6), ColumnName("weather_desc")]

        public string weather_desc { get; set; }

       

        [LoadColumn(7), ColumnName("Label")]

        public float traffic_volume { get; set; }

    }
}
