using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HandwrittenDigitRecognitionNN.NN
{
    class InputNeuron : Neuron
    {
        public List<Synapsis> RightS { get; set; }

        public InputNeuron() : base()
        {
            RightS = new List<Synapsis>();
        }
        public void AddSynapsis(Synapsis s)
        {
            RightS.Add(s);
        }
    }
}
