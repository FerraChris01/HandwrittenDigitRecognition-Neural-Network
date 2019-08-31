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
        public List<float> WeightRecords { get; set; }
        public List<float> BiasRecords { get; set; }
        private string SynapsesFile;
        private string BiasesFile;

        public HiddenLayer() { }
        public HiddenLayer(int size, string SynapsesFile, string BiasesFile, bool init)
        {
            NeuronNumber = size;
            this.SynapsesFile = SynapsesFile;
            this.BiasesFile = BiasesFile;
            Neurons = new HiddenLNeuron[NeuronNumber];
            WeightRecords = new List<float>();
            BiasRecords = new List<float>();

            for (int i = 0; i < NeuronNumber; i++)
                Neurons[i] = new HiddenLNeuron();


            //WeightRecords = DataStream.Instance.ReadWBFromFile(SynapsesFile);
            //BiasRecords = DataStream.Instance.ReadWBFromFile(BiasesFile);
            //SetBiases();

            //Init();
            if (init)
                Init();
            else
            {
                WeightRecords = DataStream.Instance.ReadWBFromFile(SynapsesFile);
                BiasRecords = DataStream.Instance.ReadWBFromFile(BiasesFile);
                SetBiases();
            }

        }
        private void SetBiases()
        {
            for (int i = 0; i < Neurons.Length; i++)
                Neurons[i].Bias = BiasRecords[i];
        }
        private void Init()
        {
            Random rn = new Random();
            for (int i = 0; i < NeuronNumber; i++)
            {
                float temp = ((float)rn.Next(-100, 100)) / 10.0f;
                Neurons[i].Bias = temp;
                BiasRecords.Add(temp);
            }
            DataStream.Instance.WriteWBOnFile(BiasRecords, BiasesFile);
        }
        public void Init_CreateSynapsisNetwork(InputLayer il)
        {
            Random rn = new Random();
            for (int i = 0; i < Neurons.Length; i++)
            {
                for (int j = 0; j < il.Neurons.Length; j++)
                {
                    float weightTemp = ((float)rn.Next(-10, 10)) / 10.0f;
                    Synapsis temp = new Synapsis(weightTemp, il.Neurons[j], Neurons[i]);
                    il.Neurons[j].AddSynapsis(temp);
                    Neurons[i].AddSynapsis(temp, false);
                    WeightRecords.Add(weightTemp);
                }
            }
            DataStream.Instance.WriteWBOnFile(WeightRecords, SynapsesFile);
        }
        public void Init_CreateSynapsisNetwork(HiddenLayer hl)
        {
            Random rn = new Random();
            for (int i = 0; i < Neurons.Length; i++)
            {
                for (int j = 0; j < hl.Neurons.Length; j++)
                {
                    float weightTemp = ((float)rn.Next(-10, 10)) / 10.0f;
                    Synapsis temp = new Synapsis(weightTemp, hl.Neurons[j], Neurons[i]);
                    hl.Neurons[j].AddSynapsis(temp, true);
                    Neurons[i].AddSynapsis(temp, false);
                    WeightRecords.Add(weightTemp);
                }
            }
            DataStream.Instance.WriteWBOnFile(WeightRecords, SynapsesFile);
        }
        public void CreateSynapsisNetwork(InputLayer il)
        {            
            int k = 0;
            for (int i = 0; i < Neurons.Length; i++)
            {
                for (int j = 0; j < il.Neurons.Length; j++)
                {
                    Synapsis temp = new Synapsis(WeightRecords[k], il.Neurons[j], Neurons[i]);
                    il.Neurons[j].AddSynapsis(temp);
                    Neurons[i].AddSynapsis(temp, false);
                    k++;
                }
            }
        }
        public void CreateSynapsisNetwork(HiddenLayer hl)
        {
            int k = 0;
            for (int i = 0; i < Neurons.Length; i++)
            {
                for (int j = 0; j < hl.Neurons.Length; j++)
                {
                    Synapsis temp = new Synapsis(WeightRecords[k], hl.Neurons[j], Neurons[i]);
                    hl.Neurons[j].AddSynapsis(temp, true);
                    Neurons[i].AddSynapsis(temp, false);
                    k++;
                }
            }
        }
        public void FeedForward()
        {
            foreach (HiddenLNeuron hn in Neurons)
                hn.UpdateActivation();
            //DataStream.Instance.DebugWriteStringOnFile("Debug/HLacts.txt", "HIDDEN LAYER");
            //for (int i = 0; i < Neurons.Length; i++)
            //{
            //    DataStream.Instance.DebugWriteStringOnFile("Debug/HLacts.txt", "Neuron " + i + Environment.NewLine);
            //    Neurons[i].UpdateActivation();
            //}
        }

        public List<float> DebugActivations()
        {
            List<float> temp = new List<float>();
            for (int i = 0; i < Neurons.Length; i++)
                temp.Add(Neurons[i].Activation);

            return temp;
        }
        public void BackPropagation(float Cost)
        {
            foreach (HiddenLNeuron n in Neurons)
                n.BackPropagation(Cost);
        }
        public void NodgeWB(float Eta, string layer)
        {
            WeightRecords.Clear();
            BiasRecords.Clear();
            for (int i = 0; i < NeuronNumber; i++)
            {
                Neurons[i].NodgeWB(Eta);
                for (int j = 0; j < Neurons[i].LeftS.Count; j++)
                    WeightRecords.Add(Neurons[i].LeftS[j].Weight);

                BiasRecords.Add(Neurons[i].Bias);
            }
            DataStream.Instance.WriteWBOnFile(WeightRecords, "Weights/s_" + layer + ".json");
            DataStream.Instance.WriteWBOnFile(BiasRecords, "Biases/b_" + layer + ".json");
        }
    }
}
