using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace RestauranRecommenderApp
{
    class RestaurantData
    {
        [ColumnName("Reviewer"), LoadColumn(0)]
        public string Reviewer { get; set; }
        [ColumnName("RestaurantName"), LoadColumn(1)]
        public string RestaurantName { get; set; }
        [ColumnName("Score"), LoadColumn(2)]
        public float Score { get; set; }
    }
}
