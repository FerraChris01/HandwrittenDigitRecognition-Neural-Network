using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HandwrittenDigitRecognitionNN.NN
{
    class Neuron
    {
        public float Activation { get; set; }

        public Neuron()
        {
            Activation = 0f;
        }
        public Neuron(float activation)
        {
            Activation = activation;
        }
    }
}
