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
        public List<float> LBiases { get; set; }
        public float Bias { get; set; }
        public float Z;
        public float DelCost { get; set; }

        public HiddenLNeuron() : base()
        {
            DelCost = 0f;
            Z = 0f;
            LeftS = new List<Synapsis>();
            RightS = new List<Synapsis>();
            LBiases = new List<float>();
        }
        public void UpdateDelCost()
        {
            DelCost = 0f;
            foreach (Synapsis s in RightS)
            {
               // s.Right.;
            }
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

            //str += ", Activation = " + Activation + Environment.NewLine + "------------------------" + Environment.NewLine;
            //DataStream.Instance.DebugWriteStringOnFile("Debug/HLacts.txt", str);
        }
        public void BackPropagation()
        {
            //foreach (Synapsis s in LeftS)
            //{
            //    s.AddLearningWeight((float)(s.Left.Activation * MyMath.Instance.DelSigmoid(Z) * Cost));
            //}
            //LBiases[LBiasesIndex++] = MyMath.Instance.DelSigmoid(Z) * Cost;

            //DataStream.Instance.DebugWriteStringOnFile("Debug/bpHL.txt", DelCost.ToString() + Environment.NewLine);


            //string temp = "Activation(" + Activation + ") * DelSigmoidZ(" + MyMath.Instance.DelSigmoid(Z) + ") * " +
            //    "DelCost(" + DelCost + ") = " + (Activation * MyMath.Instance.DelSigmoid(Z) * DelCost).ToString();
            //DataStream.Instance.DebugWriteStringOnFile("Debug/WkHL.txt", temp);

            UpdateDelCost();
            foreach (Synapsis s in LeftS)
                s.AddLearningWeight(s.Left.Activation * MyMath.Instance.DelSigmoid(Z) * DelCost);

            LBiases.Add(MyMath.Instance.DelSigmoid(Z) * DelCost);
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
