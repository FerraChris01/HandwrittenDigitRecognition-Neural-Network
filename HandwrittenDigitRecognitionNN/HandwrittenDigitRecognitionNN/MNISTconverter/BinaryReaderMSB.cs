using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HandwrittenDigitRecognitionNN
{
    public class BinaryReaderMSB : BinaryReader
    {
        public BinaryReaderMSB(Stream stream) : base(stream)
        {

        }

        public int ReadInt32MSB()
        {
            var s = ReadBytes(4);
            Array.Reverse(s);
            return BitConverter.ToInt32(s, 0);
        }
    }
    
}
