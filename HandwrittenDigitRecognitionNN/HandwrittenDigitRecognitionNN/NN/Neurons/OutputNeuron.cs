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
        public float Cost
        {
            get { return (float)Math.Pow(Activation - Y, 2); }
            set { }
        }
        public OutputNeuron() : base()
        {
            LBiases = new List<float>();
            Z = 0f;
            DelCost = 0f;
            LeftS = new List<Synapsis>();
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

            //string str = "";
            //for (int i = 0; i < LeftS.Count; i++)
            //{
            //    str += "K" + i + " = " + LeftS[i].Left.Activation + " x " + LeftS[i].Weight + " = " +
            //        (LeftS[i].Left.Activation * LeftS[i].Weight).ToString() + Environment.NewLine;
            //    Z += LeftS[i].Left.Activation * LeftS[i].Weight;
            //}
            //str += " -> Z = SumAW(" + Z + ") + " + Bias + " = " + (Z + Bias);
            //Z += Bias;
            //Activation = MyMath.Instance.Sigmoid(Z);
            //str += ", Activation = " + Activation + Environment.NewLine + "------------------------" + Environment.NewLine;
            //DataStream.Instance.DebugWriteStringOnFile("Debug/OLacts.txt", str);
        }
        public void Init()
        {
            Random rn = new Random();
            Bias = ((float)rn.Next(-100, 100)) / 10.0f;
        }
        public void BackPropagation(float cost)
        {
            //string str = "BP output neuron" + Environment.NewLine;
            foreach (Synapsis s in LeftS)
            {
                //s.AddLearningWeight((float)(s.Left.Activation * MyMath.Instance.DelSigmoid(Z) * cost));
                s.AddLearningWeight((float)(s.Left.Activation * MyMath.Instance.DelSigmoid(Z) * cost));
            }
            LBiases.Add(MyMath.Instance.DelSigmoid(Z) * cost);
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
