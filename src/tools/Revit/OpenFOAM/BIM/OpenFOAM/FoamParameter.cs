/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
using System.Collections.Generic;

namespace OpenFOAMInterface.BIM.OpenFOAM
{
    using Structs.FOAM;

    /// <summary>
    /// Abstract base class for simulation parameter that vary with used simulation-model.
    /// </summary>
    /// <typeparam name="T">Type of value.</typeparam>
    public abstract class FOAMParameter<T> : FOAMDict
    {
        /// <summary>
        /// String- list of all patch-names of the walls
        /// </summary>
        protected List<string> m_WallNames;

        /// <summary>
        /// ValueType for entries in boundaryField
        /// </summary>
        protected string m_Uniform;

        /// <summary>
        /// String-array with all patch-names of the inlets
        /// </summary>
        protected List<string> m_InletNames;

        /// <summary>
        /// String-array with all patchnames of the outlets
        /// </summary>
        protected List<string> m_OutletNames;

        /// <summary>
        /// String-array with all patch-names of the slip type walls
        /// </summary>
        protected List<string> m_SlipNames;

        /// <summary>
        /// Struct for internalField-entry
        /// </summary>
        protected struct InternalField<K>
        {
            K m_Value;

            public K Value
            {
                get
                {
                    return m_Value;
                }
                set
                {
                    m_Value = value;
                }
            }
        }

        /// <summary>
        /// InternalField-entry.
        /// </summary>
        protected InternalField<T> m_InternalField;

        /// <summary>
        /// Internalfield as string.
        /// </summary>
        protected string m_InternalFieldString;

        /// <summary>
        /// Dimension-entry
        /// </summary>
        protected int[] m_Dimensions;

        /// <summary>
        /// Dictionary which specify the different patch handlings.
        /// </summary>
        protected Dictionary<string, object> m_BoundaryField;

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="version">Version-Object</param>
        /// <param name="path">Path to this file.</param>
        /// <param name="attributes">Additional attributs.</param>
        /// <param name="format">Format of this file.</param>
        /// <param name="settings">Data-object</param>
        /// <param name="_class">Specify class of Parameter.</param>
        /// <param name="_name">Name of the FoamParameter.</param>
        /// <param name="_wallName">Name of the patch wall.</param>
        /// <param name="_InletNames">Patchnames of the inlets as string-array.</param>
        /// <param name="_OutletNames">Patchnames of the outlets as string-array.</param>
        public FOAMParameter(Version version, string path, Dictionary<string, object> attributes, SaveFormat format, Data settings, string _name, string _class, List<string> _wallNames,
            List<string> _InletNames, List<string> _OutletNames, List<string> _SlipNames)
            : base(_name, _class, version, path, attributes, format)
        {
            m_WallNames = _wallNames;
            m_InletNames = _InletNames;
            m_OutletNames = _OutletNames;
            m_SlipNames = _SlipNames;
            m_Uniform = "uniform";
            m_Dimensions = new int[7];
            m_BoundaryField = new Dictionary<string, object>();
            m_InternalField.Value = (T)m_DictFile["internalField"];
            InitAttributes();
        }

        /// <summary>
        /// Initialize Attributes.
        /// </summary>
        public override void InitAttributes()
        {
            FOAMParameterPatch<dynamic> patch = (FOAMParameterPatch<dynamic>)m_DictFile["wall"];
            foreach (string s in m_WallNames)
            {
                m_BoundaryField.Add(s, patch.Attributes);
            }

            //TODO Add SlipWalls
            foreach (string s in m_OutletNames)
            {
                AddPatchToBoundary(s, 2);
            }
            foreach (string s in m_InletNames)
            {
                AddPatchToBoundary(s, 1);
            }
            if (FOAMInterface.Singleton.Data.WindAroundBuildings)
                IncludeEtc("\"caseDicts/setContraintTypes\"");

            FoamFile.Attributes.Add("dimensions", m_Dimensions);
            FoamFile.Attributes.Add("internalField", m_InternalFieldString);
            FoamFile.Attributes.Add("boundaryField", m_BoundaryField);
        }

        /// <summary>
        /// Add OpenFOAM-Module.
        /// </summary>
        /// <param name="moduleLocation">Location of the OpenFOAM module to include in root dir of OpenFOAM.</param>
        private void IncludeEtc(in string moduleLocation)
        {
            m_BoundaryField.Add("#includeEtc", moduleLocation);
        }

        /// <summary>
        /// Add patch to BoundaryField.
        /// </summary>
        /// <param name="s">Patch-name.</param>
        private void AddPatchToBoundary(in string s, int inletOutlet)
        {
            object patch;
            if (inletOutlet == 1)
                s = "inlet";
            else if (inletOutlet == 2)
                s ="outlet";
            if (m_DictFile.TryGetValue(s, out patch))
                m_BoundaryField.Add(s, ((FOAMParameterPatch<dynamic>)patch).Attributes);
        }
    }
}
