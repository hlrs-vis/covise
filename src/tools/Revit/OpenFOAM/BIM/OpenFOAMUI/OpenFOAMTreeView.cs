/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using System.Windows;
using System.Windows.Forms;
using System.Windows.Media.Media3D;

namespace BIM.OpenFOAMExport.OpenFOAMUI
{
    /// <summary>
    /// Use this class for list <see cref="T:OpenFOAMTreeNode"/> and <see cref="T:OpenFOAMDropDownTreeNode"/> in a <see cref="T:TreeView"/>.
    /// </summary>
    public class OpenFOAMTreeView : TreeView
    {

        /// <summary>
        /// We'll use this variable to keep track of the current node that is being edited.
        /// This is set to something (non-null) only if the node's ComboBox is being displayed.
        /// </summary>
        private OpenFOAMDropDownTreeNode<dynamic> m_CurrentOFDropDownNode = null;

        /// <summary>
        /// This variable is in use for track current textbox-node that is being edited.
        /// </summary>
        private OpenFOAMTextBoxTreeNode<dynamic> m_CurrentOFTxtBoxTreeNode = null;

        /// <summary>
        /// True if the current node (OpenFOAMDropDown-/OpenFOAMTextBox-Node) has been changed.
        /// </summary>
        private bool m_ChangeValue = false;

        /// <summary>
        /// Negative and positive decimal number regular expression.
        /// </summary>
        private const string d = @"(-?\d*\.)?(-?\d+)";

        /// <summary>
        /// Regular Expresion for Vector3D.
        /// </summary>
        private readonly Regex m_Vector3DReg = new Regex("^" + d + "\\s+" + d + "\\s+" + d + "$");

        /// <summary>
        /// Regular Expression for Vector.
        /// </summary>
        private readonly Regex m_VectorReg = new Regex("^" + d + "\\s+" + d + "$");

        /// <summary>
        /// Regular Expression for number.
        /// </summary>
        private readonly Regex m_SingleReg = new Regex("^" + d);

        #region Constructors
        /// <summary>
        /// Initializes a new instance of the <see cref="T:OpenFOAMTreeView"/> class.
        /// </summary>
        public OpenFOAMTreeView()
            : base()
        {

        }
        #endregion

        /// <summary>
        /// Occurs when the <see cref="E:System.Windows.Forms.TreeView.NodeMouseClick"></see> event is fired
        /// -- that is, when a node in the tree view is clicked.
        /// </summary>
        /// <param name="e">A <see cref="T:System.Windows.Forms.TreeNodeMouseClickEventArgs"></see> that contains the event data.</param>
        protected override void OnNodeMouseClick(TreeNodeMouseClickEventArgs e)
        {
            // Are we dealing with a dropdown node?
            if (e.Node is OpenFOAMDropDownTreeNode<dynamic>)
            {
                m_CurrentOFDropDownNode = (OpenFOAMDropDownTreeNode<dynamic>)e.Node;

                // Need to add the node's ComboBox to the TreeView's list of controls for it to work
                Controls.Add(m_CurrentOFDropDownNode.ComboBox);

                // Set the bounds of the ComboBox, with a little adjustment to make it look right
                m_CurrentOFDropDownNode.ComboBox.SetBounds(
                    m_CurrentOFDropDownNode.Bounds.X - 1,
                    m_CurrentOFDropDownNode.Bounds.Y - 2,
                    m_CurrentOFDropDownNode.Bounds.Width + 25,
                    m_CurrentOFDropDownNode.Bounds.Height);

                // Listen to the SelectedValueChanged event of the node's ComboBox
                m_CurrentOFDropDownNode.ComboBox.SelectedValueChanged += new EventHandler(ComboBox_SelectedValueChanged);
                m_CurrentOFDropDownNode.ComboBox.DropDownClosed += new EventHandler(ComboBox_DropDownClosed);

                // Now show the ComboBox
                m_CurrentOFDropDownNode.ComboBox.Show();
                m_CurrentOFDropDownNode.ComboBox.DroppedDown = true;
            }
            else if (e.Node is OpenFOAMTextBoxTreeNode<dynamic>)
            {
                if (m_CurrentOFTxtBoxTreeNode != (OpenFOAMTextBoxTreeNode<dynamic>)e.Node)
                {
                    HideTextBox();
                }

                m_CurrentOFTxtBoxTreeNode = (OpenFOAMTextBoxTreeNode<dynamic>)e.Node;

                Controls.Add(m_CurrentOFTxtBoxTreeNode.TxtBox);

                //Set location of the TextBox.
                m_CurrentOFTxtBoxTreeNode.TxtBox.SetBounds(
                    m_CurrentOFTxtBoxTreeNode.Bounds.X - 1,
                    m_CurrentOFTxtBoxTreeNode.Bounds.Y - 2,
                    m_CurrentOFTxtBoxTreeNode.Bounds.Width + 25,
                    m_CurrentOFTxtBoxTreeNode.Bounds.Height);

                //Listen to Click/Leave and TextChanged Event of the TextBox.
                m_CurrentOFTxtBoxTreeNode.TxtBox.TextChanged += new EventHandler(TextBox_TextChanged);
                m_CurrentOFTxtBoxTreeNode.TxtBox.Click += new EventHandler(TextBox_Click);
                m_CurrentOFTxtBoxTreeNode.TxtBox.MouseLeave += new EventHandler(TextBox_Leave);

                m_CurrentOFTxtBoxTreeNode.TxtBox.Show();
            }
            else
            {
                HideTextBox();
                HideComboBox();
            }
            base.OnNodeMouseClick(e);
        }

        /// <summary>
        /// Handles the SelectedValueChanged event of the ComboBox control.
        /// Hides the ComboBox if an item has been selected in it.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="T:System.EventArgs"/> instance containing the event data.</param>
        void ComboBox_SelectedValueChanged(object sender, EventArgs e)
        {
            m_ChangeValue = true;
            HideComboBox();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="sende"></param>
        /// <param name="e"></param>
        void TextBox_TextChanged(object sender, EventArgs e)
        {
            m_CurrentOFTxtBoxTreeNode.TxtBox.MouseLeave += new EventHandler(TextBox_Leave);
            m_ChangeValue = true;
        }

        /// <summary>
        /// Handles Leave event from Textbox.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        void TextBox_Leave(object sender, EventArgs e)
        {
            HideTextBox();
        }

        /// <summary>
        /// Handles Click event of Textbox.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        void TextBox_Click(object sender, EventArgs e)
        {
            m_CurrentOFTxtBoxTreeNode.TxtBox.MouseLeave -= TextBox_Leave;
        }

        /// <summary>
        /// Handles the DropDownClosed event of the ComboBox control.
        /// Hides the ComboBox if the user clicks anywhere else on the TreeView or adjusts the scrollbars, or scrolls the mouse wheel.
        /// </summary>
        /// <param name="sender">The source of the event.</param>
        /// <param name="e">The <see cref="T:System.EventArgs"/> instance containing the event data.</param>
        void ComboBox_DropDownClosed(object sender, EventArgs e)
        {
            m_ChangeValue = true;
            HideComboBox();
        }


        /// <summary>
        /// Handles the <see cref="E:System.Windows.Forms.Control.MouseWheel"></see> event.
        /// Hides the ComboBox if the user scrolls the mouse wheel.
        /// </summary>
        /// <param name="e">A <see cref="T:System.Windows.Forms.MouseEventArgs"></see> that contains the event data.</param>
        protected override void OnMouseWheel(MouseEventArgs e)
        {
            HideComboBox();
            HideTextBox();
            base.OnMouseWheel(e);
        }


        /// <summary>
        /// Method to hide the currently-selected node's ComboBox
        /// </summary>
        private void HideComboBox()
        {
            if (m_CurrentOFDropDownNode != null)
            {
                ComboBox_ChangeValue();
                m_ChangeValue = false;

                // Unregister the event listener
                m_CurrentOFDropDownNode.ComboBox.SelectedValueChanged -= ComboBox_SelectedValueChanged;
                m_CurrentOFDropDownNode.ComboBox.DropDownClosed -= ComboBox_DropDownClosed;

                // Copy the selected text from the ComboBox to the TreeNode
                m_CurrentOFDropDownNode.Text = m_CurrentOFDropDownNode.ComboBox.Text;

                // Hide the ComboBox
                m_CurrentOFDropDownNode.ComboBox.Hide();
                m_CurrentOFDropDownNode.ComboBox.DroppedDown = false;

                // Remove the control from the TreeView's list of currently-displayed controls
                Controls.Remove(m_CurrentOFDropDownNode.ComboBox);

                // And return to the default state (no ComboBox displayed)
                m_CurrentOFDropDownNode = null;
            }

        }

        /// <summary>
        /// Method to hide the currently-selected node TextBox
        /// </summary>
        private void HideTextBox()
        {
            if(m_CurrentOFTxtBoxTreeNode != null)
            {
                TextBox_ChangeValue();
                m_ChangeValue = false;
                m_CurrentOFTxtBoxTreeNode.TxtBox.TextChanged -= TextBox_TextChanged;
                m_CurrentOFTxtBoxTreeNode.Text = m_CurrentOFTxtBoxTreeNode.TxtBox.Text;
                m_CurrentOFTxtBoxTreeNode.TxtBox.Hide();
                Controls.Remove(m_CurrentOFTxtBoxTreeNode.TxtBox);
                m_CurrentOFTxtBoxTreeNode = null;
            }
        }

        /// <summary>
        /// Changes value of current OpenFOAMDropDownTreeNode.
        /// </summary>
        private void ComboBox_ChangeValue()
        {
            if(!m_ChangeValue)
            {
                return;
            }
            m_CurrentOFDropDownNode.Value = m_CurrentOFDropDownNode.ComboBox.SelectedItem as dynamic;
        }

        /// <summary>
        /// Changes value of current OpenFOAMTextBoxTreeNode.
        /// </summary>
        private void TextBox_ChangeValue()
        {
            if(!m_ChangeValue)
            {
                return;
            }

            string valueString = m_CurrentOFTxtBoxTreeNode.TxtBox.Text;
            try
            {
                //Vector3D ( d d d )
                if (m_Vector3DReg.IsMatch(valueString) && m_CurrentOFTxtBoxTreeNode.Value is Vector3D)
                {
                    List<double> entries = GetListFromVector3DString(valueString);
                    if(entries.Count != 0)
                        m_CurrentOFTxtBoxTreeNode.Value = new Vector3D(entries[0], entries[1], entries[2]);
                }
                //Vector ( d d )
                else if (m_VectorReg.IsMatch(valueString) && m_CurrentOFTxtBoxTreeNode.Value is Vector)
                {
                    List<double> entries = GetListFromVector3DString(valueString);
                    if(entries.Count != 0)
                        m_CurrentOFTxtBoxTreeNode.Value = new Vector(entries[0], entries[1]);
                }
                //double / integer ( d )
                else if (m_SingleReg.IsMatch(valueString))
                {
                    string value = valueString.Trim();
                    if(m_CurrentOFTxtBoxTreeNode.Value is int)
                    {
                        int j = Convert.ToInt32(value);
                        m_CurrentOFTxtBoxTreeNode.Value = j;
                    }
                    else if(m_CurrentOFTxtBoxTreeNode.Value is double)
                    {
                        double j;
                        if(!double.TryParse(value, System.Globalization.NumberStyles.Any, System.Globalization.CultureInfo.InvariantCulture, out j))
                        {
                            FormatException format = new FormatException();
                            throw format;
                        } 
                        m_CurrentOFTxtBoxTreeNode.Value = j;
                    }
                    else
                    {
                        FormatException format = new FormatException();
                        throw format;
                    }
                }
                else if(m_CurrentOFTxtBoxTreeNode.Value is string)
                {
                    m_CurrentOFTxtBoxTreeNode.Value = valueString;
                }
                else
                {
                    FormatException format = new FormatException();
                    throw format;
                }
            }
            catch (Exception)
            {
                System.Windows.Forms.MessageBox.Show(OpenFOAMExportResource.ERR_FORMAT +
                    " " + valueString + "\nFormat: " + m_CurrentOFTxtBoxTreeNode.Format +
                    ". Could be a Format-Error in class OpenFOAMTreeView in methode TextBox_ChangeValue()",
                    OpenFOAMExportResource.MESSAGE_BOX_TITLE,
                            MessageBoxButtons.OK, MessageBoxIcon.Exclamation);

                //m_CurrentOFTxtBoxTreeNode.TxtBox.Text = m_CurrentOFTxtBoxTreeNode.Value;
            }
        }

        /// <summary>
        /// This method extract entries from a vector that is given as string, convert them to double and 
        /// return them as List.
        /// </summary>
        /// <param name="vecString">Vector-String</param>
        /// <returns>Double-List</returns>
        public static List<double> GetListFromVector3DString(string vecString)
        {
            List<double> entries = new List<double>();
            if(vecString.Equals("")|| vecString == string.Empty)
            {
                return entries;
            }
            double j = 0;
            foreach (string s in vecString.Split(' '))
            {
                s.Trim(' ');
                if (s.Equals(""))
                {
                    continue;
                }
                try
                {
                    j = Convert.ToDouble(s, System.Globalization.CultureInfo.GetCultureInfo("en-US"));
                }
                catch (FormatException)
                {
                    System.Windows.Forms.MessageBox.Show(OpenFOAMExportResource.ERR_VECTOR_FORMAT,
                        OpenFOAMExportResource.MESSAGE_BOX_TITLE,
                        MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
                    return entries;
                }
                
                entries.Add(j);
            }
            return entries;
        }
    }
}
