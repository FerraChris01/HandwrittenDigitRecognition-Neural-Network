using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HandwrittenDigitRecognitionNN.NN
{
    class MyMath
    {
        private static MyMath instance;
        private MyMath() { }
        public static MyMath Instance
        {
            get
            {
                if (instance == null)
                {
                    instance = new MyMath();
                }
                return instance;
            }
        }
        public float Sigmoid(float value)
        {
            float k = (float)Math.Exp(value);
            return k / (1.0f + k);
        }
    }
}
