using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media.Imaging;

namespace HandwrittenDigitRecognitionNN
{
    public class HWImage
    {
        public float[,] Pixels { get; set; }
        private int MaxHeight;
        private int MaxWidth;
        
        public HWImage()
        {

        }
        public HWImage(int MaxHeight, int MaxWidth)
        {
            this.MaxHeight = MaxHeight;
            this.MaxWidth = MaxWidth;
            Pixels = new float[MaxHeight, MaxWidth];
        }
        public void AddPixelImage(byte[,] values)
        {
            for (int i = 0; i < MaxHeight; i++)
            {
                for (int j = 0; j < MaxWidth; j++)                
                    Pixels[i, j] = (float)values[i, j];
            }
        }
        public float[] PixelImageToArray()
        {
            float[] temp = new float[MaxHeight * MaxWidth];
            int k = 0;
            for (int i = 0; i < MaxHeight; i++)
            {
                for (int j = 0; j < MaxWidth; j++)
                    temp[k++] = (float)Math.Round((float)(Pixels[i, j] / 255f), 1);

            }
            return temp;
        }
        public BitmapImage ToBitMapImage()
        {
            Bitmap bt = new Bitmap(MaxWidth, MaxHeight);
            for (int i = 0; i < MaxHeight; i++)
            {
                for (int j = 0; j < MaxWidth; j++)
                    bt.SetPixel(j, i, PixelToColor((int)Pixels[i, j]));
            }
            return ConvertToBitmapImage(bt);
        }
        private Color PixelToColor(int pixel)
        {
            Color c = Color.FromArgb(pixel, pixel, pixel);
            //if (pixel <= 255 && pixel >= 235)
            //    c = Color.White;
            //else if (pixel >= 0 && pixel <= 16)
            //    c = Color.Black;
            //else
            //    c = Color.Gray;

            return c;
        }
        private BitmapImage ConvertToBitmapImage(Bitmap bitmap)
        {
            using (var memory = new MemoryStream())
            {
                bitmap.Save(memory, ImageFormat.Png);
                memory.Position = 0;

                var bitmapImage = new BitmapImage();
                bitmapImage.BeginInit();
                bitmapImage.StreamSource = memory;
                bitmapImage.CacheOption = BitmapCacheOption.OnLoad;
                bitmapImage.EndInit();
                bitmapImage.Freeze();

                return bitmapImage;
            }
        }
    }
}
