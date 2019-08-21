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

        public HiddenLNeuron() : base()
        {
            Z = 0f;
            LeftS = new List<Synapsis>();
            RightS = new List<Synapsis>();
            LBiases = new List<float>();
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
            //foreach (Synapsis s in LeftS)
            //    Activation += s.Left.Activation * s.Weight;
            string str = "";
            for (int i = 0; i < LeftS.Count; i++)
            {
                str += "activation of " + i + " neuron of prev layer: " + LeftS[i].Left.Activation + "---"
                    + " weight of synapse: " + LeftS[i].Weight + Environment.NewLine;
                Z += LeftS[i].Left.Activation * LeftS[i].Weight;
            }
            DataStream.Instance.DebugWriteStringOnFile("debugActivations.txt", str);
            Z += Bias;
            Activation = MyMath.Instance.Sigmoid(Z);
        }
        public void BackPropagation(float Cost)
        {
            string str = "BP hidden neuron" + Environment.NewLine;
            foreach (Synapsis s in LeftS)
            { 
                s.LWeights.Add((float)(s.Left.Activation * MyMath.Instance.DelSigmoid(Z) * Cost));
                //str += MyMath.Instance.DelSigmoid(MyMath.Instance.Logit(Activation)).ToString() + " x " + Cost + " = " +
                //    ((float)(s.Left.Activation * (float)(MyMath.Instance.DelSigmoid(MyMath.Instance.Logit(Activation)) * Cost))) + Environment.NewLine;
                str += MyMath.Instance.DelSigmoid(Z).ToString() + " x " + Cost + " = " +
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
