using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HandwrittenDigitRecognitionNN.NN
{
    class HiddenLNeuron : Neuron
    {
        public List<Synapsis> LeftS { get; set; }
        public List<Synapsis> RightS { get; set; }
        public float[] LBiases { get; set; }
        public int LBiasesIndex { get; set; }
        public float Bias { get; set; }
        public float Z;

        public HiddenLNeuron() : base()
        {
            Z = 0f;
            LeftS = new List<Synapsis>();
            RightS = new List<Synapsis>();
            LBiases = new float[100];
            LBiasesIndex = 0;

        }
        public void AddSynapsis(Synapsis s, bool direction)
        {
            if (!direction)
                LeftS.Add(s);
            else
                RightS.Add(s);
        }
        public void UpdateActivation()
        {
            Activation = 0f;
            Z = 0f;
            foreach (Synapsis s in LeftS)
                Z += s.Left.Activation * s.Weight;

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
            //DataStream.Instance.DebugWriteStringOnFile("Debug/HLacts.txt", str);
        }
        public void BackPropagation(float Cost)
        {
            //string str = "BP hidden neuron" + Environment.NewLine;
            foreach (Synapsis s in LeftS)
            {
                //s.AddLearningWeight((float)(s.Left.Activation * MyMath.Instance.DelSigmoid(Z) * Cost));
                s.AddLearningWeight((float)(s.Left.Activation * MyMath.Instance.DelSigmoid(Z) * Cost));
            }
            //LBiases[LBiasesIndex++] = MyMath.Instance.DelSigmoid(Z) * Cost;
            LBiases[LBiasesIndex++] = MyMath.Instance.DelSigmoid(Z) * Cost;
        }
        public void NodgeWB(float eta)
        {
            NodgeWeights(eta);
            NodgeBias(eta);
            LBiasesIndex = 0;
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
