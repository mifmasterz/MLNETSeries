using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace IncomeClusteringApp
{
    public class IncomeData
    {
        [ColumnName("age"), LoadColumn(0)]
        public float age { get; set; }
        [ColumnName("workclass"), LoadColumn(1)]
        public string workclass { get; set; }
        [ColumnName("fnlwgt"), LoadColumn(2)]
        public float fnlwgt { get; set; }
        [ColumnName("education"), LoadColumn(3)]
        public string education { get; set; }
        [ColumnName("education_num"), LoadColumn(4)]
        public float education_num { get; set; }
        [ColumnName("marital_status"), LoadColumn(5)]
        public string marital_status { get; set; }
        [ColumnName("occupation"), LoadColumn(6)]
        public string occupation { get; set; }
        [ColumnName("relationship"), LoadColumn(7)]
        public string relationship { get; set; }
        [ColumnName("race"), LoadColumn(8)]
        public string race { get; set; }
        [ColumnName("sex"), LoadColumn(9)]
        public string sex { get; set; }
        [ColumnName("capital_gain"), LoadColumn(10)]
        public float capital_gain { get; set; }
        [ColumnName("capital_loss"), LoadColumn(11)]
        public float capital_loss { get; set; }
        [ColumnName("hours_per_week"), LoadColumn(12)]
        public float hours_per_week { get; set; }
        [ColumnName("native_country"), LoadColumn(13)]
        public string native_country { get; set; }
        [ColumnName("income"), LoadColumn(14)]
        public string income { get; set; }
    }
}
