using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HandwrittenDigitRecognitionNN.NN
{
    class Network
    {
        private InputLayer ILayer;
        private List<HiddenLayer> HLayers;
        private OutputLayer OLayer;
        private int NumberOfLayers;
        private float eta;

        public Network(List<int> layers, float eta)
        {
            this.eta = eta;
            ILayer = new InputLayer(layers[0]);
            OLayer = new OutputLayer(layers[layers.Count - 1], "Weights/s_outputL.json", "Biases/b_outputL.json");
            HLayers = new List<HiddenLayer>();
            for (int i = 1; i < layers.Count - 1; i++)
                HLayers.Add(new HiddenLayer(layers[i], "Weights/s_layer" + i + ".json", "Biases/b_layer" + i + ".json"));

            NumberOfLayers = HLayers.Count + 2;

            OLayer.OutputsAsDigits(0, 9);

            CreateSynapseNetworks();

            //Init_CreateSynapseNetworks();
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
        public void FeedForward(float []inputs)  //needs to be 784
        {
            ILayer.Feed(inputs);

            //foreach (HiddenLayer hl in HLayers)
            //   hl.FeedForward();
            for (int i = 0; i < HLayers.Count; i++)
            {
                DataStream.Instance.DebugWriteStringOnFile("Debug/debugActivations.txt", Environment.NewLine + "Hidden layer number " + i);
                HLayers[i].FeedForward();
            }

            OLayer.FeedForward();

            BackPropagation();
        }
        public int NetworkGuess()
        {
            return OLayer.BrightestNeuron();
        }
        private void BackPropagation()
        {

        }
        public void DebugValues()
        {
            string temp = "";
            foreach (float n in OLayer.DebugActivations())
                temp += n + Environment.NewLine;

            temp += "The network guess is: " + OLayer.BrightestNeuron();
            DataStream.Instance.DebugWriteStringOnFile("Debug/debug.txt", temp);
        }
        public string DebugActivationsOfLayers()
        {
            string str = "";
            str += "INPUT LAYER" + Environment.NewLine;

            foreach (float v in ILayer.DebugActivations())
                str += v + Environment.NewLine;

            foreach (HiddenLayer l in HLayers)
            {
                str += "HIDDEN LAYER" + Environment.NewLine;
                foreach (float v in l.DebugActivations())
                    str += v + Environment.NewLine;
            }
            str += Environment.NewLine + "OUTPUT LAYER" + Environment.NewLine;
            foreach (float v in OLayer.DebugActivations())
                str += v + Environment.NewLine;
            
            return str;
        }

    }
}
