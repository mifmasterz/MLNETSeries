using Microsoft.ML.Data;
namespace BreastCancerApp
{
    public class BreastCancerData
    {
        [ColumnName("SampleNo"), LoadColumn(0)]
        public float SampleNo { get; set; }


        [ColumnName("ClumpThickness"), LoadColumn(1)]
        public float ClumpThickness { get; set; }


        [ColumnName("UniformityOfCellSize"), LoadColumn(2)]
        public float UniformityOfCellSize { get; set; }


        [ColumnName("UniformityOfCellShape"), LoadColumn(3)]
        public float UniformityOfCellShape { get; set; }


        [ColumnName("MarginalAdhesion"), LoadColumn(4)]
        public float MarginalAdhesion { get; set; }


        [ColumnName("SingleEpithelialCellSize"), LoadColumn(5)]
        public float SingleEpithelialCellSize { get; set; }


        [ColumnName("BareNuclei"), LoadColumn(6)]
        public float BareNuclei { get; set; }


        [ColumnName("BlandChromatin"), LoadColumn(7)]
        public float BlandChromatin { get; set; }


        [ColumnName("NormalNucleoli"), LoadColumn(8)]
        public float NormalNucleoli { get; set; }


        [ColumnName("Mitoses"), LoadColumn(9)]
        public float Mitoses { get; set; }


        [ColumnName("ClassCategory"), LoadColumn(10)]
        public int ClassCategory { get; set; }

        public bool IsBenign { get; set; }
    }
}