/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
using System.Collections.Generic;
using Autodesk.Revit.DB;
using utils;

namespace OpenFOAMInterface.BIM.OpenFOAM
{
    /// <summary>
    /// The class SurfaceFeatureExtract represents the Dictionary for extracting eMesh.
    /// </summary>
    public class SurfaceFeatureExtract : FOAMDict
    {
        /// <summary>
        /// Dictionary for SurfaceFeatures
        /// </summary>
        //private Dictionary<string, object> m_SurfaceFeature;
        private readonly Dictionary<KeyValuePair<string, Document>, KeyValuePair<List<Face>, Transform>> m_Faces;

        /// <summary>
        /// Name of the STL
        /// </summary>
        private string m_STLName;

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="version">Version-object.</param>
        /// <param name="path">Path to this File.</param>
        /// <param name="attributes">Additional attributes.</param>
        /// <param name="format">Ascii or Binary.</param>
        /// <param name="settings">Data-object</param>
        /// <param name="stlName">Name of the stl</param>
        public SurfaceFeatureExtract(in Version version, in string path, in Dictionary<string, object> attributes, in Dictionary<KeyValuePair<string, Document>, KeyValuePair<List<Face>, Transform>> faces, in SaveFormat format, in string stlName)
            : base("surfaceFeatureExtractDict", "dictionary", version, path, attributes, format)
        {
            //m_SurfaceFeature = new Dictionary<string, object>();
            //m_SurfaceFeature = m_DictFolder;
            m_STLName = stlName;
            m_Faces = faces;
            InitAttributes();
        }

        /// <summary>
        /// Initialize Attributes.
        /// </summary>
        public override void InitAttributes()
        {
            FoamFile.Attributes.Add(m_STLName + ".stl", m_DictFile);
            foreach(var face in m_Faces)
                FoamFile.Attributes.Add(face.Key.Key + ".stl", m_DictFile);

            var s = FOAMInterface.Singleton.Data;
            foreach (var entry in s.MeshResolution)
            {
                var name = AutodeskHelperFunctions.GenerateNameFromElement(entry.Key);
                FoamFile.Attributes.Add(name + ".stl", m_DictFile);
            }

            //TO-DO: Dont set in this class
            //BIM.OpenFOAMExport.Exporter.Instance.settings.Features.Add("{file \"" + m_STLName + ".eMesh\"; level 3;}");
        }
    }
}
