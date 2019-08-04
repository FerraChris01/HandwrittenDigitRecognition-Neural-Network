using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HandwrittenDigitRecognitionNN.NN
{
    class OutputLayer : Layer
    {
        public OutputNeuron[] Neurons { get; set; }
        private List<Synapse> SRecords;
        public OutputLayer() { }
        public OutputLayer(int size)
        {
            NeuronNumber = size;
            Neurons = new OutputNeuron[NeuronNumber];
            SRecords = new List<Synapse>();
            Init();
        }
        private void Init()
        {
            Random rn = new Random();
            for (int i = 0; i < NeuronNumber; i++)
            {
                Neurons[i] = new OutputNeuron();
                Neurons[i].Init();
            }

        }
        public void OutputsAsDigits(int from, int to)
        {
            for (int i = 0; i <= to; i++)
                Neurons[i].RepresentingValue = i;
        }
        public void Init_CreateSynapsisNetwork(HiddenLayer hl)
        {
            Random rn = new Random();
            for (int i = 0; i < Neurons.Length; i++)
            {
                for (int j = 0; j < hl.Neurons.Length; j++)
                {
                    Synapse temp = new Synapse((float)rn.NextDouble(), hl.Neurons[j], Neurons[i]);
                    hl.Neurons[j].AddSynapsis(temp, true);
                    Neurons[i].AddSynapsis(temp);
                    SRecords.Add(temp);
                }
            }
        }
        public void CreateSynapsisNetwork(HiddenLayer hl)
        {
            List<Synapse> STemp = SRecords;
            int k = 0;
            for (int i = 0; i < Neurons.Length; i++)
            {
                for (int j = 0; j < hl.Neurons.Length; j++)
                {
                    Synapse temp = STemp[k];
                    hl.Neurons[j].AddSynapsis(temp, true);
                    Neurons[i].AddSynapsis(temp);
                    k++;
                }
            }
        }
        public void ForwardPropagation()
        {
            foreach (OutputNeuron on in Neurons)
                on.UpdateActivation();
        }
        public int NetworkGuess()
        {
            float temp = -1;
            OutputNeuron tempN = new OutputNeuron();
            foreach (OutputNeuron on in Neurons)
            {
                if (on.Activation > temp)
                {
                    temp = on.Activation;
                    tempN = on;
                }
            }
            return tempN.RepresentingValue;               

        }
    }
}
