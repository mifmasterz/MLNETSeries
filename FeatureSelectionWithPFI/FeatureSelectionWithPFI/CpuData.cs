using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace FeatureSelectionWithPFI
{
    class CpuData
    {
        [LoadColumn(0)]
        public string Vendor { get; set; }

        [LoadColumn(1)]
        public string Model { get; set; }

        [LoadColumn(2)]
        public float MYCT { get; set; }

        [LoadColumn(3)]
        public float MMIN { get; set; }

        [LoadColumn(4)]
        public float MMAX { get; set; }

        [LoadColumn(5)]
        public float CACH { get; set; }

        [LoadColumn(6)]
        public float CHMIN { get; set; }

        [LoadColumn(7)]
        public float CHMAX { get; set; }

        [LoadColumn(8)]
        public float PRP { get; set; }

        [LoadColumn(9)]
        [ColumnName("Label")]
        public float ERP { get; set; }

      
    }
}
