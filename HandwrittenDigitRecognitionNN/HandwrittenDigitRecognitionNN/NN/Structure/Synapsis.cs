using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HandwrittenDigitRecognitionNN.NN
{
    class Synapsis
    {
        public float Weight { get; set; }
        public List<float> LWeights { get; set; }
        public Neuron Left { get; set; }
        public Neuron Right { get; set; }

        public Synapsis(float weight, Neuron left, Neuron right)
        {
            Weight = weight;
            Left = left;
            Right = right;
            LWeights = new List<float>();
        }
        public Synapsis(float weight)
        {
            Weight = weight;
        }
        public void NodgeWeight(float eta)
        {
            float av = LWeights.Average();
            Weight -= eta * av;
            LWeights.Clear();
        }
        public void AddLearningWeight(float weight)
        {
            LWeights.Add(weight);
        }

    }
}
