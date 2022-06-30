/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
using System.Collections.Generic;
using System.Windows;

namespace OpenFOAMInterface.BIM.OpenFOAM
{
    using TransportModel = Enums.TransportModel;
    
    /// <summary>
    /// TransportProperties-Dictionary in Constant-Folder.
    /// </summary>
    public class TransportProperties : FOAMDict
    {
        /// <summary>
        /// Transportmodel that has to be set in Data.
        /// </summary>
        private TransportModel m_TransportModel;

        /// <summary>
        /// Contructor.
        /// </summary>
        /// <param name="version">Version-object.</param>
        /// <param name="path">Path to this File.</param>
        /// <param name="attributes">Additional attributes.</param>
        /// <param name="format">Ascii or Binary.</param>
        /// <param name="settings">Data-objects</param>
        public TransportProperties(Version version, string path, Dictionary<string, object> attributes, SaveFormat format)
            : base("transportProperties", "dictionary", version, path, attributes, format)
        {
            InitAttributes();
        }

        /// <summary>
        /// Initialize Attributes.
        /// </summary>
        public override void InitAttributes()
        {
            m_TransportModel = FOAMInterface.Singleton.Data.TransportModel;
            Dictionary<string, object> transportModelParameterData = m_DictFile["transportModelParameter"] as Dictionary<string, object>;

            //nu-Unit = default
            int[] m_Unit = new int[] { 0, 2, -1, 0, 0, 0, 0 };
            FoamFile.Attributes.Add("transportModel", m_TransportModel);
            string modelParameterValue = string.Empty;

            if (m_TransportModel != TransportModel.Newtonian)
            {
                Dictionary<string, object> transportModelParemeter = new Dictionary<string, object>();
                foreach (var v in transportModelParameterData)
                {
                    if (v.Key.Equals("function polynomial"))
                    {
                        List<Vector> vectors = v.Value as List<Vector>;
                        modelParameterValue = "(";
                        foreach (Vector vec in vectors)
                        {
                            modelParameterValue += "( " + vec.ToString().Replace(";", " ");
                        }
                        modelParameterValue = ");";
                    }
                    else
                    {
                        m_Unit = ChangeDimension(v.Key);
                    }
                    modelParameterValue = AddUnit(m_Unit, v.Value);
                    transportModelParemeter.Add(v.Key, modelParameterValue);
                }
                FoamFile.Attributes.Add(m_TransportModel.ToString(), transportModelParemeter);
            }
            else
            {
                foreach (var obj in transportModelParameterData)
                {
                    m_Unit = ChangeDimension(obj.Key);
                    if (m_Unit != null)
                    {
                        modelParameterValue = AddUnit(m_Unit, transportModelParameterData[obj.Key]);
                        FoamFile.Attributes.Add(obj.Key + " " + obj.Key, modelParameterValue);
                    }
                    else
                    {
                        modelParameterValue = obj.Value.ToString();
                        FoamFile.Attributes.Add(obj.Key, modelParameterValue);
                    }
                }
            }
        }

        /// <summary>
        /// Change dimension vector in specified metric.
        /// </summary>
        /// <param name="tag">Specifier-Tag for changing</param>
        /// <returns>Dim-Vector in specified metric.</returns>
        private int[] ChangeDimension(string tag)
        {
            int[] dim = new int[7];
            if (tag.Contains("tau"))
            {
                dim = new int[] { 0, 2, -2, 0, 0, 0, 0 };
            }
            else if (tag.Equals("m"))
            {
                dim = new int[] { 0, 0, 1, 0, 0, 0, 0 };
            }
            else if (tag.Equals("n"))
            {
                dim = new int[] { 0, 0, 0, 0, 0, 0, 0 };
            }
            else if (tag.Equals("k"))
            {
                dim = ChangeDimension("m");
            }
            else if (tag.Equals("beta"))
            {
                dim = new int[] { 0, 0, 0, -1, 0, 0, 0 };
            }
            else if (tag.Equals("TRef"))
            {
                dim = new int[] { 0, 0, 0, 1, 0, 0, 0 };
            }
            else if (tag.Equals("Pr"))
            {
                dim = new int[] { 0, 0, 0, 0, 0, 0, 0 };
            }
            else if (tag.Equals("Prt"))
            {
                dim = ChangeDimension("Pr");
            }
            else if (tag.Equals("nu"))
            {
                dim = new int[] { 0, 2, -1, 0, 0, 0, 0 };
            }
            else
            {
                return null;
            }
            return dim;
        }

        /// <summary>
        /// Returns a string that is build from the given unit and value.
        /// </summary>
        /// <param name="unit">Interger array.</param>
        /// <param name="value">Value as object</param>
        /// <returns>Unit + Value as string.</returns>
        private string AddUnit(int[] unit, object _value)
        {
            string modelParameterValue = string.Empty;
            string entry = string.Empty;
            string value = string.Empty;
            if (_value.GetType() == typeof(double))
            {
                double v = (double)_value;
                value = v.ToString(System.Globalization.CultureInfo.GetCultureInfo("en-US").NumberFormat);
            }
            else
            {
                value = _value.ToString();
            }

            for (int i = 0; i < unit.Length; i++)
            {
                if (i == unit.Length - 1)
                {
                    entry += unit[i];
                    break;
                }
                entry += unit[i] + " ";
            }
            modelParameterValue = "[ " + entry + "] " + value;
            return modelParameterValue;
        }
    }
}
