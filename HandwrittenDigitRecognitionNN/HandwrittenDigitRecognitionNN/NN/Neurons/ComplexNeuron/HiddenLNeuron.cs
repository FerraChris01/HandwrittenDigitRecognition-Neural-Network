using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HandwrittenDigitRecognitionNN.NN
{
    class HiddenLNeuron : ComplexNeuron
    {
        public List<Synapsis> RightS { get; set; }
        public override float DelCost
        {
            get
            {
                float tempDCost = 0;
                foreach (Synapsis s in RightS)
                {
                    float denom = 0f;
                    foreach (Synapsis sj in s.Right.LeftS)
                        denom += sj.Weight;

                    tempDCost += s.Right.DelCost * (s.Weight / denom);
                }
                //DataStream.Instance.DebugWriteStringOnFile("Debug/HLNdelcosts.txt", tempDCost.ToString());
                return (float)Math.Round(tempDCost, 2);
            }
            set { }
        }

        public HiddenLNeuron() : base()      
        {
            RightS = new List<Synapsis>();
        }
        public void AddSynapsis(Synapsis s, bool direction)
        {
            if (!direction)
                LeftS.Add(s);
            else
                RightS.Add(s);
        }      
        
    }
}
