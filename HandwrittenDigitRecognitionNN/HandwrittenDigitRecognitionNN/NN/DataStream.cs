using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;
using System.Web.Script.Serialization;

namespace HandwrittenDigitRecognitionNN.NN
{
    class DataStream
    {
        private readonly string SynapsisFile;
        private readonly string BiasFile;

        public DataStream()
        {
            this.SynapsisFile = "s_layer";
            this.BiasFile = "b_layer";
        }
        public List<Synapsis> ReadSynapsesOfLayer(int layerIndex)
        {
            List<Synapsis> records = new List<Synapsis>();
            using (StreamReader r = new StreamReader(SynapsisFile + layerIndex + ".json"))
            {
                string json = r.ReadToEnd();
                records = JsonConvert.DeserializeObject<List<Synapsis>>(json);
            }
            return records;
        }

        public List<float> ReadBiasesOfLayer(int layerIndex)
        {
            List<float> records = new List<float>();
            using (StreamReader r = new StreamReader(BiasFile + layerIndex + ".json"))
            {
                string json = r.ReadToEnd();
                records = JsonConvert.DeserializeObject<List<float>>(json);
            }
            return records;
        }

        public void WriteSynapsesOfLayer(List<Synapsis> records, int layerIndex)
        {
            using (StreamWriter file = new StreamWriter(SynapsisFile + layerIndex + ".json"))
            {
                JsonSerializer serializer = new JsonSerializer();
                serializer.Serialize(file, records);
            }
        }

        public void WriteBiasesOfLayer(List<float> records, int layerIndex)
        {
            using (StreamWriter file = new StreamWriter(BiasFile + layerIndex + ".json"))
            {
                JsonSerializer serializer = new JsonSerializer();
                serializer.Serialize(file, records);
            }
        }

    }
}