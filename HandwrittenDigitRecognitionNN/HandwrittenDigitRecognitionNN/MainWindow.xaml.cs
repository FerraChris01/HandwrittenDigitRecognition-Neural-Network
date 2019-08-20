using HandwrittenDigitRecognitionNN.NN;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace HandwrittenDigitRecognitionNN
{
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
            Network nw = new Network(new List<int> { 784, 16, 16, 10 });
            nw.Eta = 2.0f;

            float[] inputValues = new float[784];
            Random rn = new Random();
            string str = "";
            for (int i = 0; i < inputValues.Length; i++)
            {
                float temp = (float)(rn.Next(100)) / 100.0f;
                inputValues[i] = temp;
                str += temp.ToString() + Environment.NewLine;
            }
            nw.FeedForward(inputValues, rn.Next(10));

            DataStream.Instance.DebugWriteStringOnFile("Debug/debugOutputL.txt", nw.DebugActivationsOfLayers());
            DataStream.Instance.DebugWriteStringOnFile("Debug/debugOutputL.txt", "The network guess is: " + nw.NetworkGuess().ToString());
        }
    }
}
