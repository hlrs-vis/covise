/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
 
using System;
using System.Collections.Generic;
using System.Windows.Forms;

namespace BIM.OpenFOAMExport.OpenFOAMUI
{
    /// <summary>
    /// This class is in use for list OpenFOAM-Parameter as a dropdownlist.
    /// </summary>
    public class OpenFOAMDropDownTreeNode<T> : OpenFOAMTreeNode<T>
    {
        /// <summary>
        /// ComboBox-Object
        /// </summary>
        private ComboBox m_ComboBox = new ComboBox();

        #region Constructors

        /// <summary>
        /// Initializes a new instance of the <see cref="T:OpenFOAMDropDownTreeNode"/> class.
        /// </summary>
        /// <param name="enum">Stored value in node.</param>
        /// <param name="_settings">Settings-object.</param>
        /// <param name="_keyPath">Path to value in dictionary in settings.</param>
        public OpenFOAMDropDownTreeNode(T _value, ref Settings _settings, List<string> _keyPath)
            : base(_value.ToString(), ref _settings, _keyPath, _value)
        {
            if(_value is Enum)
            {
                var @enum = _value as Enum;
                foreach (var value in Enum.GetValues(@enum.GetType()))
                {
                    m_ComboBox.Items.Add(value);
                }
            }
            else if(_value is bool)
            {
                //bool? => nullable bool
                bool? _bool = _value as bool?;
                if(_bool != null)
                {
                    m_ComboBox.Items.Add(_bool);
                    m_ComboBox.Items.Add(!_bool);
                }
            }
            else
            {
                m_ComboBox.Items.Add("Not initialized in OpenFOAMDropDownTreeNode");
            }

        }
        #endregion

        /// <summary>
        /// Getter-Setter for ComboBox.
        /// </summary>
        public ComboBox ComboBox
        {
            get
            {
                m_ComboBox.DropDownStyle = ComboBoxStyle.DropDownList;
                return m_ComboBox;
            }
            set
            {
                m_ComboBox = value;
                m_ComboBox.DropDownStyle = ComboBoxStyle.DropDownList;
            }
        }
    }
}
