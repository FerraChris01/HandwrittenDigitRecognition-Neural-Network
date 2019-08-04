using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HandwrittenDigitRecognitionNN.NN
{
    class InputLayer : Layer
    {
        public InputNeuron[] Neurons { get; set; }
        public InputLayer() { }
        
        public InputLayer(int size)
        {
            NeuronNumber = size;
            Neurons = new InputNeuron[size];
            Init();
        }
        private void Init()
        {
            for (int i = 0; i < NeuronNumber; i++)
                Neurons[i] = new InputNeuron();

        }
        public void Feed(float []inputs)
        {
            for (int i = 0; i < Neurons.Length; i++)
                Neurons[i].Activation = inputs[i];
        }
    }
}
