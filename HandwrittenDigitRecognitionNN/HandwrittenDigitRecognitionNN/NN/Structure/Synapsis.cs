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
        public Neuron Left { get; set; }
        public Neuron Right { get; set; }

        public Synapsis()
        {
        }
        public Synapsis(float weight, Neuron left, Neuron right)
        {
            Weight = weight;
            Left = left;
            Right = right;
        }
        public Synapsis(float weight)
        {
            Weight = weight;
        }
    }
}
