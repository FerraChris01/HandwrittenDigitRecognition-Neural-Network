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
        private int EpochSize;
        public List<LabeledImage> MnistImages { get; set; }
        private MNISTmanager MM;
        private int GuessedImages;
         
        public NetworkManager(float Eta, int NEpochs, int EpochSize, int StartIndex)
        {            
            this.NEpochs = NEpochs;
            this.EpochSize = EpochSize;
            NN = new Network(new List<int> { 784, 16, 16, 10 }, true);
            NN.Eta = Eta;

            MM = new MNISTmanager("train-labels-idx1-ubyte.gz", "train-images-idx3-ubyte.gz", StartIndex);
            TrainTheNetwork();            
        }
        public NetworkManager(int StartIndex)
        {
            GuessedImages = 0;
            NN = new Network(new List<int> { 784, 16, 16, 10 }, false);
            MM = new MNISTmanager("t10k-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", StartIndex);
            ValidateNN();
        }
        private void TrainTheNetwork()
        {
            for (int i = 0; i < NEpochs; i++)
            {
                DataStream.Instance.DebugWriteStringOnFile("Debug/Epoch.txt", i.ToString());                
                MnistImages = MM.ReadLabeledImages(EpochSize);
                foreach (LabeledImage l in MnistImages)
                {
                    NN.FeedForward(l.PixelValues, l.Label);

                    DataStream.Instance.DebugWriteStringOnFile("Debug/activations.txt",
                        NN.DebugActivationsOfLayers() + Environment.NewLine + "----------------------------" + 
                        Environment.NewLine);
                    
                    NN.BackPropagation();                    
                }
                NN.NodgeWB();
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
