using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HandwrittenDigitRecognitionNN.NN
{
    class Synapse
    {
        public float Weight { get; set; }
        public Neuron Left { get; set; }
        public Neuron Right { get; set; }

        public Synapse()
        {
        }
        public Synapse(float weight, Neuron left, Neuron right)
        {
            Weight = weight;
            Left = left;
            Right = right;
        }
        public Synapse(float weight)
        {
            Weight = weight;
        }
    }
}
