using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HandwrittenDigitRecognitionNN.NN
{
    class OutputNeuron : Neuron
    {
        private List<Synapsis> LeftS;
        public int RepresentingValue { get; set; }
        public float Bias { get; set; }
        public int Y { get; set; }
        public float DelCost
        { 
            get { return 2 * (Activation - Y) }
        }

        public OutputNeuron()
        {
            Cost = 0;
            LeftS = new List<Synapsis>();
        }
        public void AddSynapsis(Synapsis s)
        {
            LeftS.Add(s);
        }
        public void UpdateActivation()
        {
            Activation = 0f;
            foreach (Synapsis s in LeftS)
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
