using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HandwrittenDigitRecognitionNN.NN
{
    class InputNeuron : Neuron
    {
        private List<Synapse> RightS;
        public void AddSynapsis(Synapse s)
        {
            RightS.Add(s);
        }
    }
}
