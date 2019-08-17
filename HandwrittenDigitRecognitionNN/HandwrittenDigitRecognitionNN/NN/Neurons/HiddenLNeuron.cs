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
        public float Bias { get; set; }
        
        public HiddenLNeuron()
        {
            LeftS = new List<Synapsis>();
            RightS = new List<Synapsis>();
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
            //foreach (Synapsis s in LeftS)
            //    Activation += s.Left.Activation * s.Weight;
            string str = "";
            for (int i = 0; i < LeftS.Count; i++)
            {
                str += "activation of " + i + " neuron of prev layer: " + LeftS[i].Left.Activation + "---"
                    + " weight of synapse: " + LeftS[i].Weight + Environment.NewLine;
                Activation += LeftS[i].Left.Activation * LeftS[i].Weight;
            }
            DataStream.Instance.DebugWriteStringOnFile("debugActivations.txt", str);
            Activation += Bias;
            Activation = MyMath.Instance.Sigmoid(Activation);
        }
    }
}
