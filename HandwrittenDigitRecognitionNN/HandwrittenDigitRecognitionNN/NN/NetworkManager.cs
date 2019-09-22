using HandwrittenDigitRecognitionNN.NN;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HandwrittenDigitRecognitionNN
{
    class NetworkManager
    {
        private Network NN;
        private int NEpochs;
        private int NBatches;
        private int MiniBatchSize;
        public List<LabeledImage> MnistImages { get; set; }
        private MNISTmanager MM;
        private int GuessedImages;
         
        public NetworkManager(float Eta, int NEpochs, int NBatches, int MiniBachSize, int StartIndex)
        {
            this.NEpochs = NEpochs;
            this.NBatches = NBatches;
            this.MiniBatchSize = MiniBachSize;
            NN = new Network(new List<int> { 784, 200, 10 }, true);
            NN.Eta = Eta;

            MM = new MNISTmanager("train-labels-idx1-ubyte.gz", "train-images-idx3-ubyte.gz", StartIndex);
            TrainTheNetwork();            
        }
        public NetworkManager(int StartIndex)
        {
            GuessedImages = 0;
            NN = new Network(new List<int> { 784, 200, 10 }, false);
            MM = new MNISTmanager("t10k-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", StartIndex);
            ValidateNN();
        }
        private void TrainTheNetwork()
        {
            for (int j = 0; j < NEpochs; j++)
            {
                for (int i = 0; i < NBatches; i++)
                {
                    DataStream.Instance.DebugWriteStringOnFile("Debug/Batch.txt", i.ToString());
                    MnistImages = MM.ReadLabeledImages(MiniBatchSize);
                    foreach (LabeledImage l in MnistImages)
                    {
                        NN.FeedForward(l.PixelValues, l.Label);

                        DataStream.Instance.DebugWriteStringOnFile("Debug/activations.txt",
                            NN.DebugActivationsOfLayers() + Environment.NewLine + "----------------------------" +
                            Environment.NewLine);

                        NN.BackPropagation();
                    }
                    DataStream.Instance.DebugWriteStringOnFile("Debug/Cost.txt", NN.Costs.Average().ToString());
                    NN.Costs.Clear();
                    NN.NodgeWB();
                }
                MM.ResetClusters();
                DataStream.Instance.DebugWriteStringOnFile("Debug/Epoch.txt", "Epoch " + (j + 1) + " finished" + Environment.NewLine);
            }

        }
        private void ValidateNN()
        {
            int i = 0;
            for (int j = 0; j < 1; j++)
            {
                MnistImages = MM.ReadLabeledImages(100);
                foreach (LabeledImage l in MnistImages)
                {
                    DataStream.Instance.DebugWriteStringOnFile("Debug/progress.txt", i.ToString());                    
                    NN.FeedForward(l.PixelValues, l.Label);
                    if (NN.NetworkGuess() == l.Label)
                        GuessedImages++;

                    string temp = NN.DebugValues() + Environment.NewLine + "Solution: " + l.Label + Environment.NewLine;
                    DataStream.Instance.DebugWriteStringOnFile("Debug/Oacts.txt", temp);
                    i++;
                }
            }
            
            DataStream.Instance.DebugWriteStringOnFile("Debug/Guess.txt", GuessedImages.ToString() + "/100");
        }
        public LabeledImage DebugReadSample(int index)
        {
            LabeledImage temp = MM.DebugGetPixelValues(index);
            return temp;
        }
    }
}
