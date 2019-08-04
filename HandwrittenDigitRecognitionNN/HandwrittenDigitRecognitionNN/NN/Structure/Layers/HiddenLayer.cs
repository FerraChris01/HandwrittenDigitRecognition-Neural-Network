using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HandwrittenDigitRecognitionNN.NN
{
    class HiddenLayer : Layer
    {
        public HiddenLNeuron[] Neurons { get; set; }
        private List<Synapse> SRecords;
        public HiddenLayer() { }
        public HiddenLayer(int size)
        {
            NeuronNumber = size;
            Neurons = new HiddenLNeuron[NeuronNumber];
            SRecords = new List<Synapse>();
            Init();

        }
        private void Init()
        {
            Random rn = new Random();
            for (int i = 0; i < NeuronNumber; i++)
            {
                Neurons[i] = new HiddenLNeuron();
                Neurons[i].Init();
            }


        }
        public void Init_CreateSynapsisNetwork(InputLayer il)
        {
            Random rn = new Random();
            for (int i = 0; i < Neurons.Length; i++)
            {
                for (int j = 0; j < il.Neurons.Length; j++)
                {
                    Synapse temp = new Synapse((float)rn.NextDouble(), il.Neurons[j], Neurons[i]);
                    il.Neurons[j].AddSynapsis(temp);
                    Neurons[i].AddSynapsis(temp, false);
                    SRecords.Add(temp);
                }                    
            }
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
                    Neurons[i].AddSynapsis(temp, false);
                    SRecords.Add(temp);
                }
            }
        }
        public void CreateSynapsisNetwork(InputLayer il)
        {
            List<Synapse> STemp = SRecords;
            int k = 0;
            for (int i = 0; i < Neurons.Length; i++)
            {
                for (int j = 0; j < il.Neurons.Length; j++)
                {
                    Synapse temp = STemp[k];
                    il.Neurons[j].AddSynapsis(temp);
                    Neurons[i].AddSynapsis(temp, false);
                    k++;
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
                    Neurons[i].AddSynapsis(temp, false);
                    k++;
                }
            }
        }
        public void ForwardPropagation()
        {
            foreach (HiddenLNeuron hn in Neurons)
                hn.UpdateActivation();
        }
    }
}
