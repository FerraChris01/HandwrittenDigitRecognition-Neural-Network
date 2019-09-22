using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HandwrittenDigitRecognitionNN.NN
{
    class ComplexNeuron : Neuron
    {
        public float Z;
        public float Bias { get; set; }
        public List<Synapsis> LeftS { get; set; }        
        public List<float> LBiases { get; set; }
        public virtual float DelCost { get; set; }   
        public ComplexNeuron() : base()
        {
            LBiases = new List<float>();
            Z = 0f;
            DelCost = 0f;
            LeftS = new List<Synapsis>();
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
        public void BackPropagation()
        {
            float DCost = DelCost;
            foreach (Synapsis s in LeftS)
            {
                float temp = (float)Math.Round(s.Left.Activation * MyMath.Instance.DelSigmoid(Z) * DCost, 3);
                s.AddLearningWeight(temp);
                //string str = "ActivationL-1(" + s.Left.Activation + ") * DelSigmoid[" + Z + "](" +                    
                //    MyMath.Instance.DelSigmoid(Z) + ") * DelCost(" + DCost + ") = " +
                //    (temp).ToString();

                //DataStream.Instance.DebugWriteStringOnFile("Debug/delEdelW.txt", str);
            }
            //DataStream.Instance.DebugWriteStringOnFile("Debug/delEdelW.txt", "--------------------------");

            LBiases.Add(MyMath.Instance.DelSigmoid(Z) * DCost);
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
