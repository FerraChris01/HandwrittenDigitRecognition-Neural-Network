﻿using System;
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

            WeightRecords = DataStream.Instance.ReadWBFromFile(SynapsesFile);
            BiasRecords = DataStream.Instance.ReadWBFromFile(BiasesFile);
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
            DataStream.Instance.WriteWBOnFile(BiasRecords, BiasesFile);
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
            DataStream.Instance.WriteWBOnFile(WeightRecords, SynapsesFile);
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

            string act = "";
            for (int i = 0; i < Neurons.Length; i++)
                act += "Activation of " + " neuron " + i + ": " + Neurons[i].Activation + Environment.NewLine;

            DataStream.Instance.DebugWriteStringOnFile("Debug/debugOutputActs.txt", act);
            UpdateCost();
        }
        public float UpdateCost()
        {
            float Cost = 0;
            string str = "";            
            foreach (OutputNeuron n in Neurons)
            {
                str += "Single: " + n.DelCost + Environment.NewLine + "---------------" + Environment.NewLine;
                Cost += n.DelCost;
            }
            str += "-------------------- SINGLE FINISHED -----------------" + Environment.NewLine;
            DataStream.Instance.DebugWriteStringOnFile("Debug/debugSingleCost.txt", str);    

            return Cost;
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
        public void BackPropagation(float Cost)
        {
            foreach (OutputNeuron n in Neurons)
                n.BackPropagation(Cost);
        }
        public void NodgeWB(float Eta)
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
            DataStream.Instance.WriteWBOnFile(WeightRecords, "Weights/s_outputL.json");
            DataStream.Instance.WriteWBOnFile(BiasRecords, "Biases/b_outputL.json");
        }

    }
}
