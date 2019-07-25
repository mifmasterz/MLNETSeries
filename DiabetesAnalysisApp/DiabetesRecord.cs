using System;
using System.Collections.Generic;
using System.Text;

namespace DiabetesAnalysisApp
{
    public class DiabetesRecord
    {
        public DateTime TimeStamp { set; get; }
        public string Code { get; set; }
        public float Data { get; set; }
    }
}
