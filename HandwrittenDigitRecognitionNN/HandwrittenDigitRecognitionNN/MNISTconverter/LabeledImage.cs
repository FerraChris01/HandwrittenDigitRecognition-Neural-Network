using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;

namespace HandwrittenDigitRecognitionNN
{
    public class LabeledImage
    {
        private const int Height = 28;
        private const int Width = 28;
        public BitmapImage Image { get; set; }
        public float[] PixelValues { get; set; }
        public int Label { get; set; }
        public LabeledImage() { }
        public LabeledImage(HWImage img, int Label)
        {
            this.PixelValues = new float[Height * Width];
            Array.Copy(img.PixelImageToArray(), this.PixelValues, Height * Width);
            this.Image = img.ToBitMapImage();
            this.Label = Label;            
        }
    }
}
