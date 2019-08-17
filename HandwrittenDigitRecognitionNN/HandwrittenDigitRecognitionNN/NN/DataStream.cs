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

        public List<float> ReadSynapseWeightsOfLayer(string fileName)
        {
            List<float> records = new List<float>();
            using (StreamReader r = new StreamReader(fileName))
            {
                string json = r.ReadToEnd();
                records = JsonConvert.DeserializeObject<List<float>>(json);
            }
            return records;
        }

        public List<float> ReadBiasesOfLayer(string fileName)
        {
            List<float> records = new List<float>();
            using (StreamReader r = new StreamReader(fileName))
            {
                string json = r.ReadToEnd();
                records = JsonConvert.DeserializeObject<List<float>>(json);
            }
            return records;
        }

        public void WriteSynapseWeightsOfLayer(List<float> weights, string fileName)
        {
            using (StreamWriter file = new StreamWriter(fileName))
            {
                JsonSerializer serializer = new JsonSerializer();
                serializer.Serialize(file, weights);
            }
        }

        public void WriteBiasesOfLayer(List<float> records, string fileName)
        {
            using (StreamWriter file = new StreamWriter(fileName))
            {
                JsonSerializer serializer = new JsonSerializer();
                serializer.Serialize(file, records);
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