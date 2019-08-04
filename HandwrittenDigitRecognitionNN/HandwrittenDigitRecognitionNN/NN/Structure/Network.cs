using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HandwrittenDigitRecognitionNN.NN.Structure
{
    class Network
    {
        private InputLayer ILayer;
        private List<HiddenLayer> HLayers;
        private OutputLayer OLayer;

        public Network() { }
        public Network(List<int> layers)
        {
            ILayer = new InputLayer(layers[0]);
            OLayer = new OutputLayer(layers[layers.Count - 1]);
            for (int i = 1; i < layers.Count - 1; i++)
                HLayers.Add(new HiddenLayer(layers[i]));

            OLayer.OutputsAsDigits(0, 9);
            Init_CreateSynapseNetworks();
        }
        private void Init_CreateSynapseNetworks()
        {
            HLayers[0].Init_CreateSynapsisNetwork(ILayer);
            int i = 0;
            do
            {
                i++;
                HLayers[i].Init_CreateSynapsisNetwork(HLayers[i - 1]);
            } while (i < HLayers.Count - 1) ;
            OLayer.Init_CreateSynapsisNetwork(HLayers[i]);                
        }
        public void ForwardPropagation(float []inputs)  //needs to be 784
        {
            ILayer.Feed(inputs);
            foreach (HiddenLayer hl in HLayers)
                hl.ForwardPropagation();

            OLayer.ForwardPropagation();
        }

    }
}
