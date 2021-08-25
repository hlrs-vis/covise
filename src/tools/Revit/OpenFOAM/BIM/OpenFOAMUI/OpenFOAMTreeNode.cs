/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
using System.Collections.Generic;
using System.Runtime.Serialization;
using System.Windows.Forms;

namespace OpenFOAMInterface.BIM.OpenFOAMUI
{
    /// <summary>
    /// A treeNode that is used to store values inside the node in a OpenFOAMTreeView.
    /// </summary>
    /// <typeparam name="T">Type of Value.</typeparam>
    public abstract class OpenFOAMTreeNode<T> : TreeNode
    {
        /// <summary>
        /// List of strings that leads to the stored value in the settings-dictionary m_SimulationDefault.
        /// </summary>
        protected List<string> m_KeyPath;


        /// <summary>
        /// Stored Value.
        /// </summary>
        protected T m_Value;

        #region Constructor
        /// <summary>
        /// Initializes a new instance of the <see cref="T:OpenFOAMDropDownTreeNode"/> class.
        /// </summary>
        public OpenFOAMTreeNode()
            : base()
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="T:OpenFOAMDropDownTreeNode"/> class.
        /// </summary>
        /// <param name="text">The text.</param>
        public OpenFOAMTreeNode(string text)
            : base(text)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="T:OpenFOAMDropDownTreeNode"/> class.
        /// </summary>
        /// <param name="text">The text.</param>
        /// <param name="children">The children.</param>
        public OpenFOAMTreeNode(string text, TreeNode[] children)
            : base(text, children)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="T:OpenFOAMDropDownTreeNode"/> class.
        /// </summary>
        /// <param name="serializationInfo">A <see cref="T:System.Runtime.Serialization.SerializationInfo"></see> containing the data to deserialize the class.</param>
        /// <param name="context">The <see cref="T:System.Runtime.Serialization.StreamingContext"></see> containing the source and destination of the serialized stream.</param>
        public OpenFOAMTreeNode(SerializationInfo serializationInfo, StreamingContext context)
            : base(serializationInfo, context)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="T:OpenFOAMDropDownTreeNode"/> class.
        /// </summary>
        /// <param name="text">The text.</param>
        /// <param name="imageIndex">Index of the image.</param>
        /// <param name="selectedImageIndex">Index of the selected image.</param>
        public OpenFOAMTreeNode(string text, int imageIndex, int selectedImageIndex)
            : base(text, imageIndex, selectedImageIndex)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="T:OpenFOAMDropDownTreeNode"/> class.
        /// </summary>
        /// <param name="text">The text.</param>
        /// <param name="imageIndex">Index of the image.</param>
        /// <param name="selectedImageIndex">Index of the selected image.</param>
        /// <param name="children">The children.</param>
        public OpenFOAMTreeNode(string text, int imageIndex, int selectedImageIndex, TreeNode[] children)
            : base(text, imageIndex, selectedImageIndex, children)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="T:OpenFOAMTreeNode"/> class.
        /// </summary>
        /// <param name="text">Text for node.</param>
        /// <param name="_settings">Settings-object.</param>
        /// <param name="_keyPath">Path to value in dictionary in settings.</param>
        /// <param name="_value">Stored value in node.</param>
        public OpenFOAMTreeNode(string text, ref Settings _settings, List<string> _keyPath, T _value)
            : base(text)
        {
            Exporter.Instance.settings = _settings;
            m_KeyPath = new List<string>();
            foreach (string s in _keyPath)
            {
                m_KeyPath.Add(s);
            }
            m_Value = _value;
        }
        #endregion

        ///// <summary>
        ///// This method is in use to change the corresponding attribute of the value
        ///// that is stored in this node in settings.
        ///// </summary>
        //public virtual void UpdateSettingsEntry()
        //{
        //    Dictionary<string, object> att = BIM.OpenFOAMExport.Exporter.Instance.settings.SimulationDefault;
        //    foreach (string s in m_KeyPath)
        //    {
        //        if (att[s] is Dictionary<string, object>)
        //        {
        //            Dictionary<string, object> newLevel = att[s] as Dictionary<string, object>;
        //            att = newLevel;
        //        }
        //        else if (att[s] is FOAMParameterPatch<dynamic> patch)
        //        {
        //            att = patch.Attributes;
        //        }
        //        else
        //        {
        //            if (att.ContainsKey(s))
        //            {
        //                att[s] = m_Value;
        //            }
        //        }
        //    }
        //}

        /// <summary>
        /// Getter-Setter for value that is stored in the node.
        /// </summary>
        public T Value
        {
            get
            {
                return m_Value;
            }
            set
            {
                m_Value = value;
                Exporter.Instance.settings.UpdateSettingsEntry(m_KeyPath, m_Value);
            }
        }
    }
}
