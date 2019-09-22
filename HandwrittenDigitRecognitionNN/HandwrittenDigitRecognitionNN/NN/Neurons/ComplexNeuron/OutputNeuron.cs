using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HandwrittenDigitRecognitionNN.NN
{
    class OutputNeuron : ComplexNeuron
    {
        public int RepresentingValue { get; set; }
        public int Y { get; set; }
        public float Cost
        {
            get { return (float)Math.Pow(Activation - Y, 2); }
            set { }
        }
        public override float DelCost
        {
            get { return -(Y - Activation); }
            set { }
        }
        public OutputNeuron() : base()
        {
            Cost = 0f;
        }        
        public void AddSynapsis(Synapsis s)
        {
            LeftS.Add(s);
        }      

    }
}
