using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace AirBnBApp
{
    public class ListingData
    {
        [ColumnName("id"), LoadColumn(0)]
        public int id { get; set; }
        [ColumnName("name"), LoadColumn(1)]
        public string name { get; set; }
        [ColumnName("host_id"), LoadColumn(2)]
        public string host_id { get; set; }
        [ColumnName("host_name"), LoadColumn(3)]
        public string host_name { get; set; }
        [ColumnName("neighbourhood_group"), LoadColumn(4)]
        public string neighbourhood_group { get; set; }
        [ColumnName("neighbourhood"), LoadColumn(5)]
        public string neighbourhood { get; set; }
        [ColumnName("latitude"), LoadColumn(6)]
        public float latitude { get; set; }
        [ColumnName("longitude"), LoadColumn(7)]
        public float longitude { get; set; }
        [ColumnName("room_type"), LoadColumn(8)]
        public string room_type { get; set; }
        [ColumnName("price"), LoadColumn(9)]
        public float price { get; set; }
        [ColumnName("minimum_nights"), LoadColumn(10)]
        public float minimum_nights { get; set; }
        [ColumnName("number_of_reviews"), LoadColumn(11)]
        public int number_of_reviews { get; set; }
        [ColumnName("last_review"), LoadColumn(12)]
        public string last_review { get; set; }
        [ColumnName("reviews_per_month"), LoadColumn(13)]
        public float reviews_per_month { get; set; }
        [ColumnName("calculated_host_listings_count"), LoadColumn(14)]
        public int calculated_host_listings_count { get; set; }
        [ColumnName("availability_365"), LoadColumn(15)]
        public float availability_365 { get; set; }

        public bool Label;
    }
}
