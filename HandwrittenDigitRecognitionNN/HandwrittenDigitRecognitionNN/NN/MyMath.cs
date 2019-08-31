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
        public float ReLU(float value)
        {
            float ret = 0f;
            if (value >= 0)
                ret = value;

            return ret;
        }
        public float DelRelu(float value)
        {
            float ret = 0f;
            if (value >= 0)
                ret = 1;

            return ret;
        }
        public float Sigmoid(double value)
        {
            double k = Math.Exp(value);
            double res = k / (1.0f + k);
            return (float)Math.Round(res, 2);
        }
        public float DelSigmoid(double value)
        {
            return Sigmoid(value) * (1 - Sigmoid(value));
        }
    }
}
