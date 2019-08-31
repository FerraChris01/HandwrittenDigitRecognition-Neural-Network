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
        private NetworkManager NM;
        private int mnistIndex;
        public MainWindow()
        {
            InitializeComponent();
            NM = new NetworkManager(1.0f, 300, 100, 0);
            //NM = new NetworkManager(0);
            mnistIndex = 0;
            SwitchToImage();

        }
        private void SwitchToImage()
        {
            index.Text = mnistIndex.ToString();
            mnistImage.Source = NM.DebugReadSample(mnistIndex).Image;
            solution.Content = NM.DebugReadSample(mnistIndex).Label;
        }
        private void next_Click(object sender, RoutedEventArgs e)
        {
            mnistIndex++;
            SwitchToImage();
        }

        private void previous_Click(object sender, RoutedEventArgs e)
        {
            mnistIndex--;
            SwitchToImage();
        }

        private void index_TextChanged(object sender, TextChangedEventArgs e)
        {
            mnistIndex = Int32.Parse(index.Text);
            SwitchToImage();
        }
    }
}
