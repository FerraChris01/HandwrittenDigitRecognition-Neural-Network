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
        private int NumberOfLayers;

        private DataStream IOStreamer;

        public Network() { }
        public Network(List<int> layers)
        {
            IOStreamer = new DataStream();
            ILayer = new InputLayer(layers[0]);
            OLayer = new OutputLayer(layers[layers.Count - 1]);
            for (int i = 1; i < layers.Count - 1; i++)
                HLayers.Add(new HiddenLayer(layers[i]));

            NumberOfLayers = HLayers.Count + 2;

            //SetValuesOfLayers();

            OLayer.OutputsAsDigits(0, 9);

            Init_CreateSynapseNetworks(); // CreateSynapseNetworks();
        }
        private void SetValuesOfLayers()
        {
            for (int i = 0; i < HLayers.Count; i++)
            {
                HLayers[i].SRecords = IOStreamer.ReadSynapsesOfLayer(i + 1);
                HLayers[i].BiasRecords = IOStreamer.ReadBiasesOfLayer(i + 1);
            }
            OLayer.SRecords = IOStreamer.ReadSynapsesOfLayer(NumberOfLayers - 1);
            OLayer.BiasRecords = IOStreamer.ReadBiasesOfLayer(NumberOfLayers - 1);

        }
        private void CreateSynapseNetworks()
        {
            HLayers[0].CreateSynapsisNetwork(ILayer);
            int i = 0;
            do
            {
                i++;
                HLayers[i].CreateSynapsisNetwork(HLayers[i - 1]);
            } while (i < HLayers.Count - 1);
            OLayer.CreateSynapsisNetwork(HLayers[i]);
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
