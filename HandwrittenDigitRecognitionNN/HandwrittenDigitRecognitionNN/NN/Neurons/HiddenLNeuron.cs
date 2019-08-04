using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HandwrittenDigitRecognitionNN.NN
{
    class HiddenLNeuron : Neuron
    {
        private List<Synapse> LeftS;
        private List<Synapse> RightS;
        public float Bias { get; set; }
        public void AddSynapsis(Synapse s, bool direction)
        {
            if (!direction)
                LeftS.Add(s);
            else
                RightS.Add(s);
        }
        public void UpdateActivation()
        {
            Activation = 0f;
            foreach (Synapse s in LeftS)
                Activation += s.Left.Activation * s.Weight;

            Activation += Bias;
            Activation = MyMath.Instance.Sigmoid(Activation);
        }
        public void Init()
        {
            Random rn = new Random();
            Bias = (float)rn.NextDouble();
        }
    }
}
