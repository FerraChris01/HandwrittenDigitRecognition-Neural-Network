using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HandwrittenDigitRecognitionNN.NN
{
    class OutputNeuron : Neuron
    {
        private List<Synapse> LeftS;
        public int RepresentingValue { get; set; }
        public float Bias { get; set; }
        public void AddSynapsis(Synapse s)
        {
            LeftS.Add(s);
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
