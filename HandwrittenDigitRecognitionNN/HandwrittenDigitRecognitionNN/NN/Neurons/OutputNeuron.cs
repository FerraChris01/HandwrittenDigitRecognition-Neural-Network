using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HandwrittenDigitRecognitionNN.NN
{
    class OutputNeuron : Neuron
    {
        public float Z;
        public List<Synapsis> LeftS { get; set; }
        public int RepresentingValue { get; set; }
        public float Bias { get; set; }
        public List<float> LBiases { get; set; }
        public int Y { get; set; }
        public float DelCost
        {
            get { return 2 * (Activation - Y); }
            set { }
        }
        public OutputNeuron() : base()
        {
            Z = 0f;
            DelCost = 0f;
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
            Z = 0f;
            foreach (Synapsis s in LeftS)
                Z += s.Left.Activation * s.Weight;

            Z += Bias;
            Activation = MyMath.Instance.Sigmoid(Z);
        }
        public void Init()
        {
            Random rn = new Random();
            Bias = (float)rn.NextDouble();
        }
        public void BackPropagation(float Cost)
        {
            string str = "BP output neuron" + Environment.NewLine;
            foreach (Synapsis s in LeftS)
            {
                s.LWeights.Add((float)(s.Left.Activation * MyMath.Instance.DelSigmoid(Z) * Cost));
                //str += s.Left.Activation + " x " + MyMath.Instance.DelSigmoid(MyMath.Instance.Logit(Activation)).ToString() + " x " + Cost + " = " + 
                //    ((float)(s.Left.Activation * (float)(MyMath.Instance.DelSigmoid(MyMath.Instance.Logit(Activation)) * Cost))) + Environment.NewLine;

                str += s.Left.Activation + " x " + MyMath.Instance.DelSigmoid(Z).ToString() + " x " + Cost + " = " +
                    ((float)(s.Left.Activation * (float)(MyMath.Instance.DelSigmoid(Z) * Cost))) + Environment.NewLine;
            }
            str += Environment.NewLine + "-----------------------------" + Environment.NewLine;
            DataStream.Instance.DebugWriteStringOnFile("Debug/debugBP.txt", str);
            LBiases.Add(MyMath.Instance.DelSigmoid(Z) * Cost);
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
