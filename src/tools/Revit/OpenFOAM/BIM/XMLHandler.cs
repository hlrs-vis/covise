/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
using System;
using System.Collections.Generic;
using System.IO;
using System.Xml;
using System.Xml.Linq;

namespace BIM.OpenFOAMExport
{
    /// <summary>
    /// This class is in use for handling the xml-config file.
    /// </summary>
    public class XMLHandler
    {
        Data m_Data;

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="settings">Data-object for current project.</param>
        public XMLHandler(Data settings)
        {
            m_Data = settings;
            CreateConfig();
        }

        /// <summary>
        /// Read the config file and add entries to settings.
        /// </summary>
        /// <param name="path"></param>
        private void ReadConfig(string path)
        {
            if (File.Exists(path))
            {
                XmlDocument doc = new XmlDocument();
                doc.Load(path);

                Dictionary<string, object> dict = m_Data.SimulationDefault;
                List<string> keyPath = new List<string>();
                //keyPath.Add("OpenFOAMConfig");
                //keyPath.Add("DefaultParameter");
                UpdateData(doc, keyPath, dict);

                //XmlTextReader reader = new XmlTextReader(path);
                //while (reader.Read())
                //{
                //    switch (reader.NodeType)
                //    {
                //        case XmlNodeType.Element:
                //            {
                //                // The node is an element.
                //                Console.Write("<" + reader.Name);

                //                while (reader.MoveToNextAttribute())
                //                {
                //                    // Read the attributes.
                //                    Console.Write(" " + reader.Name + "='" + reader.Value + "'");
                //                }

                //                Console.WriteLine(">");
                //                break;
                //            }
                //        case XmlNodeType.Text:
                //            {
                //                //Display the text in each element.
                //                Console.WriteLine(reader.Value);
                //                break;
                //            }
                //        case XmlNodeType.EndElement:
                //            {
                //                //Display the end of the element.
                //                Console.Write("</" + reader.Name);
                //                Console.WriteLine(">");
                //                break;
                //            }
                //    }

                //}
                //reader.Close();
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="doc"></param>
        /// <param name="dict"></param>
        private void UpdateData(XmlDocument doc, List<string> keyPath, Dictionary<string, object> dict)
        {
            foreach (var entry in dict)
            {
                if (entry.Value is Dictionary<string, object> newLevel)
                {
                    keyPath.Add(entry.Key);
                    UpdateData(doc, keyPath, newLevel);
                    break;
                }
                else if (entry.Value is System.Collections.ArrayList newLevelArray)
                {
                }
                else if (entry.Value is FOAMParameterPatch<dynamic> patch)
                {
                    keyPath.Add(entry.Key);
                    UpdateData(doc, keyPath, patch.Attributes);
                }
                else
                {
                    UpdateDataBasedOnXmlEntry(doc, keyPath, entry.Key, entry.Value);
                }
            }
        }

        /// <summary>
        /// Update settings.
        /// </summary>
        /// <param name="doc">Xml document.</param>
        /// <param name="keyPath">Path to attribute in SimulationDefault in Data.</param>
        /// <param name="entryName">Name of node in settings.</param>
        /// <param name="entryValue">Value in settings.</param>
        private void UpdateDataBasedOnXmlEntry(XmlDocument doc, List<string> keyPath, string entryName, object entryValue)
        {
            keyPath.Add(entryName);
            string xmlPath = GetXmlPath(keyPath);
            XmlNode node = doc.SelectSingleNode(xmlPath);
            object value = null;
            if (entryValue is double)
            {
                value = Convert.ToDouble(node.FirstChild.Value, System.Globalization.CultureInfo.GetCultureInfo("en-US"));
            }
            else if (entryValue is int)
            {
                value = Convert.ToInt32(node.FirstChild.Value);
            }
            else if (entryValue is System.Windows.Media.Media3D.Vector3D)
            {
                List<double> vec = new List<double>();
                foreach (var vecEntry in node.FirstChild.Value.Split(';'))
                {
                    vec.Add(Convert.ToDouble(vecEntry, System.Globalization.CultureInfo.GetCultureInfo("en-US")));
                }
                value = new System.Windows.Media.Media3D.Vector3D(vec[0], vec[1], vec[2]);
            }
            else if (entryValue is System.Windows.Vector)
            {
                List<double> vec = new List<double>();
                foreach (var vecEntry in node.FirstChild.Value.Split(';'))
                {
                    vec.Add(Convert.ToDouble(vecEntry, System.Globalization.CultureInfo.GetCultureInfo("en-US")));
                }
                value = new System.Windows.Vector(vec[0], vec[1]);
            }
            else if (entryValue is bool)
            {
                value = Convert.ToBoolean(node.FirstChild.Value);
            }
            else if (entryValue is Enum e)
            {
                foreach(var enu in Enum.GetValues(e.GetType()))
                {
                    if(node.FirstChild.Value.Equals(enu))
                    {
                        value = enu;
                    }
                }
            }
            else if (entryValue is string s)
            {
                value = node.FirstChild.Value;
            }
            if(value != null)
            {
                m_Data.UpdateDataEntry(keyPath, value);
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="keyPath"></param>
        /// <returns></returns>
        private string GetXmlPath(List<string> keyPath)
        {
            string xmlPath = "OpenFOAMConfig[1]/DefaultParameter[1]";
            foreach (string s in keyPath)
            {
                xmlPath += "/" + s + "[1]";
            }
            return xmlPath;
        }

        /// <summary>
        /// Create config file if it doesn't exist.
        /// </summary>
        private void CreateConfig()
        {
            //remove file:///
            string assemblyDir = System.Reflection.Assembly.GetExecutingAssembly().GetName().CodeBase.Substring(8);

            //remove name of dll from string
            string assemblyDirCorrect = assemblyDir.Remove(assemblyDir.IndexOf("OpenFOAMExport.dll"), 18).Replace("/", "\\");

            //configname
            string configPath = assemblyDirCorrect + "openFOAMExporter.config";
            if (!File.Exists(configPath))
            {

                var config = new XDocument();
                var elements = new XElement("OpenFOAMConfig",
                    new XElement("OpenFOAMEnv"),
                    new XElement("SSH")
                );

                var defaultElement = new XElement("DefaultParameter");
                Dictionary<string, object> dict = m_Data.SimulationDefault;
                CreateXMLTree(defaultElement, dict);
                elements.Add(defaultElement);

                config.Add(elements);

                XElement ssh = config.Root.Element("SSH");
                ssh.Add(
                        new XElement("user", m_Data.SSH.User),
                        new XElement("host", m_Data.SSH.ServerIP),
                        new XElement("serverCasePath", m_Data.SSH.ServerCaseFolder),
                        new XElement("ofAlias", m_Data.SSH.OfAlias),
                        new XElement("port", m_Data.SSH.Port.ToString()),
                        new XElement("tasks", m_Data.SSH.SlurmCommand.ToString()),
                        new XElement("download", m_Data.SSH.Download),
                        new XElement("delete", m_Data.SSH.Delete),
                        new XElement("slurm", m_Data.SSH.Slurm)
                );
                config.Save(configPath);
            }
            else
            {
                ReadConfig(configPath);
            }
        }

        /// <summary>
        /// Creates a XML-tree from given dict.
        /// </summary>
        /// <param name="e">XElement xml will be attached to.</param>
        /// <param name="dict">Source for XML-tree.</param>
        private void CreateXMLTree(XElement e, Dictionary<string, object> dict)
        {
            foreach (var element in dict)
            {
                string nameNode = element.Key;
                nameNode = PrepareXMLString(nameNode);
                if (nameNode.Equals("null"))
                    continue;
                var elem = new XElement(nameNode);
                if (element.Value is Dictionary<string, object>)
                {
                    CreateXMLTree(elem, element.Value as Dictionary<string, object>);
                }
                else
                {
                    if(element.Value is FOAMParameterPatch<dynamic> patch)
                    {
                        CreateXMLTree(elem, patch.Attributes);
                    }
                    else
                    {
                        elem.Value = element.Value.ToString();
                    }
                }
                e.Add(elem);
            }
        }

        /// <summary>
        /// Removes critical strings for xml.
        /// </summary>
        /// <param name="nameNode">String which will be prepared.</param>
        /// <returns>Prepared string.</returns>
        private static string PrepareXMLString(string nameNode)
        {
            if (nameNode.Equals("0"))
            {
                nameNode = "null";
                return nameNode;
            }

            var criticalXMLCharacters = new Dictionary<string, string>()
            {
                { "(", "lpar" },
                { ")", "rpar" },
                { ",", "comma" },
                { "*", "ast" },
                { " ", "nbsp" }
            };

            foreach (var critical in criticalXMLCharacters)
            {
                nameNode = nameNode.Replace(critical.Key, critical.Value);
            }

            return nameNode;
        }
    }
}
