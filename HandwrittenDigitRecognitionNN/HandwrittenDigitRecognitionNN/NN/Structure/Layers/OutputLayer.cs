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
        public List<float> WeightRecords { get; set; }
        public List<float> BiasRecords { get; set; }
        public float Cost;

        private string SynapsesFile;
        private string BiasesFile;

        public OutputLayer() { }
        public OutputLayer(int size, string SynapsesFile, string BiasesFile)
        {
            NeuronNumber = size;
            this.SynapsesFile = SynapsesFile;
            this.BiasesFile = BiasesFile;
            Neurons = new OutputNeuron[NeuronNumber];
            WeightRecords = new List<float>();
            BiasRecords = new List<float>();

            for (int i = 0; i < NeuronNumber; i++)
                Neurons[i] = new OutputNeuron();

            WeightRecords = DataStream.Instance.ReadSynapseWeightsOfLayer(SynapsesFile);
            BiasRecords = DataStream.Instance.ReadBiasesOfLayer(BiasesFile);
            SetBiases();

            //Init();
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
                float temp = rn.Next(-20, 20);
                Neurons[i].Bias = temp;
                BiasRecords.Add(temp);
            }
            DataStream.Instance.WriteBiasesOfLayer(BiasRecords, BiasesFile);
        }
        public void OutputsAsDigits(int from, int to)
        {
            for (int i = from; i <= to; i++)
                Neurons[i].RepresentingValue = i;

        }
        public void Init_CreateSynapsisNetwork(HiddenLayer hl)
        {
            Random rn = new Random();
            for (int i = 0; i < Neurons.Length; i++)
            {
                for (int j = 0; j < hl.Neurons.Length; j++)
                {
                    float weightTemp = ((float)rn.Next(-200, 200)) / 10.0f;
                    Synapsis temp = new Synapsis(weightTemp, hl.Neurons[j], Neurons[i]);
                    hl.Neurons[j].AddSynapsis(temp, true);
                    Neurons[i].AddSynapsis(temp);
                    WeightRecords.Add(weightTemp);
                }
            }
            DataStream.Instance.WriteSynapseWeightsOfLayer(WeightRecords, SynapsesFile);
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
                    Neurons[i].AddSynapsis(temp);
                    k++;
                }
            }
        }
        public void FeedForward()
        {
            foreach (OutputNeuron on in Neurons)
                on.UpdateActivation();

            UpdateCost();
        }
        private void UpdateCost()
        {
            Cost = 0;
            foreach (OutputNeuron n in Neurons)
                Cost += n.DelCost;
                
        }
        public void BackPropagation()
        {

        }
        public int BrightestNeuron()
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

        public List<float> DebugActivations()
        {
            List<float> temp = new List<float>();
            for (int i = 0; i < Neurons.Length; i++)
                temp.Add(Neurons[i].Activation);

            return temp;
        }
        public void SetY(int value)
        {
            foreach (OutputNeuron n in Neurons)
            {
                if (n.RepresentingValue == value)
                    n.Y = 1;
                else
                    n.Y = 0;
            }
        }
    }
}
