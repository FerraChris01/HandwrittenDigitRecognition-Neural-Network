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
        public List<float> LBiases { get; set; }
        public int Y { get; set; }
        public float DelCost
        {
            get { return 2 * (Activation - Y); }
            set { }
        }
        public OutputNeuron()
        {
            DelCost = 0;
            LeftS = new List<Synapsis>();
            LBiases = new List<float>();
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
        public void BackPropagation(float Cost)
        {
            foreach (Synapsis s in LeftS)
              s.LWeights.Add(s.Left.Activation* MyMath.Instance.DelSigmoid(MyMath.Instance.Logit(Activation) * Cost));

            LBiases.Add(MyMath.Instance.DelSigmoid(MyMath.Instance.Logit(Activation)) * Cost);
        }
        public void NodgeWB(float eta)
        {
            NodgeWeights(eta);
            NodgeBias(eta);
            LBiases.Clear();
        }
        private void NodgeWeights(float eta)
        {
            foreach (Synapsis s in LeftS)
                s.NodgeWeight(eta);
        }
        private void NodgeBias(float eta)
        {
            float av = LBiases.Average();
            Bias -= eta * av;
        }


    }
}
