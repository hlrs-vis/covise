/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
// Original Source Code: https://www.codeproject.com/Articles/14544/A-TreeView-Control-with-ComboBox-Dropdown-Nodes
using System.Collections.Generic;
using System.Windows;
using System.Windows.Forms;
using System.Windows.Media.Media3D;

namespace OpenFOAMInterface.BIM.OpenFOAMUI
{
    /// <summary>
    /// The class OpenFOAMTextBoxTreeNode is an child class from TreeNode and offers a additional Textbox.
    /// </summary>
    /// <typeparam name="T">Generic Value inside Textbox.</typeparam>
    public class OpenFOAMTextBoxTreeNode<T> : OpenFOAMTreeNode<T>
    {
        /// <summary>
        /// Format of type as string.
        /// </summary>
        private readonly string m_Format;

        /// <summary>
        /// Textbox-Object.
        /// </summary>
        private TextBox m_TxtBox = new TextBox();

        #region Constructor
        /// <summary>
        /// Initializes a new instance of the <see cref="T:OpenFOAMDropDownTreeNode"/> class.
        /// </summary>
        /// <param name="_txtBoxValue">Stored value in node.</param>
        /// <param name="_settings">Settings-object.</param>
        /// <param name="_keyPath">Path to value in dictionary in settings.</param>
        public OpenFOAMTextBoxTreeNode(T _txtBoxValue, ref Settings _settings, List<string> _keyPath)
            : base(_txtBoxValue.ToString().Replace(';', ' '), ref _settings, _keyPath, _txtBoxValue)
        {
            m_TxtBox.Text = Text;
            if (Value is Vector3D)
            {
                m_Format = "x y z -> (x,y,z) ∊ ℝ";
            }
            else if (Value is Vector)
            {
                m_Format = "x y -> (x,y) ∊ ℝ";
            }
            else if (Value is int || Value is double)
            {
                m_Format = "int/double";
            }
            else if (Value is string)
            {
                m_Format = "string";
            }
            else
            {
                m_Format = "Please initialize format for this type in OpenFOAMTextBoxTreeNode.";
            }
        }
        #endregion

        /// <summary>
        /// Getter-Setter for TxtBox.
        /// </summary>
        public TextBox TxtBox
        {
            get
            {
                return m_TxtBox;
            }
            set
            {
                m_TxtBox = value;
            }
        }

        /// <summary>
        /// Getter for format-string.
        /// </summary>
        public string Format { get => m_Format; }
    }
}
