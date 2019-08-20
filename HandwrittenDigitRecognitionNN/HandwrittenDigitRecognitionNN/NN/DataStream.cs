using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace HandwrittenDigitRecognitionNN.NN
{
    class DataStream
    {
        private static DataStream instance;
        private DataStream() { }
        public static DataStream Instance
        {
            get
            {
                if (instance == null)
                {
                    instance = new DataStream();
                }
                return instance;
            }
        }
        public List<float> ReadWBFromFile(string fileName)
        {
            List<float> records = new List<float>();
            using (StreamReader r = new StreamReader(fileName))
            {
                string json = r.ReadToEnd();
                records = JsonConvert.DeserializeObject<List<float>>(json);
            }
            return records;
        }

        public void WriteWBOnFile(List<float> wb, string fileName)
        {
            using (StreamWriter file = new StreamWriter(fileName))
            {
                JsonSerializer serializer = new JsonSerializer();
                serializer.Serialize(file, wb);
            }
        }
       
        public void DebugWriteStringOnFile(string fileName, string txt)
        {
            using (StreamWriter file = new StreamWriter(fileName, true))
            {
                file.WriteLine(txt);
            }
        }
    }
}