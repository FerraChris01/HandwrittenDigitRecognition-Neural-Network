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
            get { return 2 * (Activation - Y); }
            set { }
        }

        public OutputNeuron()
        {
            DelCost = 0;
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
        public void BackPropagation(float Cost, int NeuronIndex)
        {
            for (int i = 0; i < LeftS.Count; i++)
            {
                float Wj = LeftS[i].Left.Activation * MyMath.Instance.DelSigmoid(MyMath.Instance.Logit(Activation)) * Cost;
                DataStream.Instance.Learning_WriteRecordOnFile("Learning/LWeights/OutputLayer/Neuron" + NeuronIndex + "/W" + i, Wj);
            }
            float Bj = MyMath.Instance.DelSigmoid(MyMath.Instance.Logit(Activation)) * Cost;

        }
        public void NodgeWeights()
        {

        }
    }
}
