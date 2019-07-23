using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace SeedsApp
{
    public class SeedData
    {
        [ColumnName("Area"), LoadColumn(0)]
        public float Area { get; set; }

        [ColumnName("Perimeter"), LoadColumn(1)]
        public float Perimeter { get; set; }

        [ColumnName("Compactness"), LoadColumn(2)]
        public float Compactness { get; set; }

        [ColumnName("Length"), LoadColumn(3)]
        public float Length { get; set; }

        [ColumnName("Width"), LoadColumn(4)]
        public float Width { get; set; }

        [ColumnName("AsymmetryCoefficient"), LoadColumn(5)]
        public float AsymmetryCoefficient  { get; set; }

        [ColumnName("LengthOfKernel"), LoadColumn(6)]
        public float LengthOfKernel { get; set; }

        [ColumnName("Category"), LoadColumn(7)]
        public int Category { get; set; }
    }
    
}
