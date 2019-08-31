using HandwrittenDigitRecognitionNN.NN;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;

namespace HandwrittenDigitRecognitionNN
{
    class MNISTmanager
    {
        private List<TestCase> data;
        private int ClusterImageIndex;
        private int ClusterLabelIndex;
        public MNISTmanager(string LPath, string IPath, int startIndex)
        {
            ClusterImageIndex = startIndex;
            ClusterLabelIndex = startIndex;
            data = Shuffle(FileReaderMNIST.LoadImagesAndLables(LPath, IPath).ToList());
        }      
        public List<LabeledImage> ReadLabeledImages(int amount)
        {
            List<HWImage> imagesTemp = GetPixelValues(amount);
            List<int> labelsTemp = GetLabels(amount);

            List<LabeledImage> ret = new List<LabeledImage>();
            for (int i = 0; i < amount; i++)
                ret.Add(new LabeledImage(imagesTemp[i], labelsTemp[i]));
            
            return ret;
        }
        private List<HWImage> GetPixelValues(int amount)
        {
            List<HWImage> ret = new List<HWImage>();
            int i = ClusterImageIndex;
            while (i < ClusterImageIndex + amount)
            {
                HWImage temp = new HWImage(28, 28);
                temp.AddPixelImage(data.ElementAt(i++).Image);
                ret.Add(temp);
            }
            ClusterImageIndex = i;
            return ret;
        }
        private List<int> GetLabels(int amount)
        {
            List<int> ret = new List<int>();
            int i = ClusterLabelIndex;
            while (i < ClusterLabelIndex + amount)
                ret.Add(data.ElementAt(i++).Label);

            ClusterLabelIndex = i;
            return ret;
        }
        private static List<T> Shuffle<T>(List<T> list)
        {
            Random rnd = new Random();
            for (int i = 0; i < list.Count; i++)
            {
                int k = rnd.Next(0, i);
                T value = list[k];
                list[k] = list[i];
                list[i] = value;
            }
            return list;
        }
        public LabeledImage DebugGetPixelValues(int index)
        {
            HWImage temp = new HWImage(28, 28);
            temp.AddPixelImage(data.ElementAt(index).Image);
            return new LabeledImage(temp, data.ElementAt(index).Label);
        }
        

    }
}
