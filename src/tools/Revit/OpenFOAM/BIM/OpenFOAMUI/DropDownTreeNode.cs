#region (c) 2006 Matt Valerio
// DropDownTreeView Control
//
// 6.21.06 Matt Valerio
// (matt.valerio@gmail.com)
//
// Filename: DropDownTreeNode.cs
//
// Description: Provides a DropDownTreeNode that extends TreeNode for use in the
// DropDownTreeView control.
//
// ============================================================================
// This software is free software; you can modify and/or redistribute it, provided
// that the author is credited with the work.
//
// This library is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
// PARTICULAR PURPOSE.
// ============================================================================
//
// Revisions:
//   1  6.21.06 MDV  Initial CodeProject article
//   2  7.21.06 MDV  Insisted that the internal ComboBox be ComboBoxStyle.DropDownList.
#endregion


using System;
using System.Collections.Generic;
using System.Text;
using System.Windows.Forms;
using System.Runtime.Serialization;


namespace DropDownTreeView
{
    /// <summary>
    /// A class that inherits from TreeNode that lets you specify a ComboBox to be shown
    /// at the TreeNode's position
    /// </summary>
    public class DropDownTreeNode : TreeNode
    {
        #region Constructors
        /// <summary>
        /// Initializes a new instance of the <see cref="T:DropDownTreeNode"/> class.
        /// </summary>
        public DropDownTreeNode()
            : base()
        {            
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="T:DropDownTreeNode"/> class.
        /// </summary>
        /// <param name="text">The text.</param>
        public DropDownTreeNode(string text)
            : base(text)
        {            
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="T:DropDownTreeNode"/> class.
        /// </summary>
        /// <param name="text">The text.</param>
        /// <param name="children">The children.</param>
        public DropDownTreeNode(string text, TreeNode[] children)
            : base(text, children)
        {            
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="T:DropDownTreeNode"/> class.
        /// </summary>
        /// <param name="serializationInfo">A <see cref="T:System.Runtime.Serialization.SerializationInfo"></see> containing the data to deserialize the class.</param>
        /// <param name="context">The <see cref="T:System.Runtime.Serialization.StreamingContext"></see> containing the source and destination of the serialized stream.</param>
        public DropDownTreeNode(SerializationInfo serializationInfo, StreamingContext context)
            : base(serializationInfo, context)
        {            
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="T:DropDownTreeNode"/> class.
        /// </summary>
        /// <param name="text">The text.</param>
        /// <param name="imageIndex">Index of the image.</param>
        /// <param name="selectedImageIndex">Index of the selected image.</param>
        public DropDownTreeNode(string text, int imageIndex, int selectedImageIndex)
            : base(text, imageIndex, selectedImageIndex)
        {            
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="T:DropDownTreeNode"/> class.
        /// </summary>
        /// <param name="text">The text.</param>
        /// <param name="imageIndex">Index of the image.</param>
        /// <param name="selectedImageIndex">Index of the selected image.</param>
        /// <param name="children">The children.</param>
        public DropDownTreeNode(string text, int imageIndex, int selectedImageIndex, TreeNode[] children)
            : base(text, imageIndex, selectedImageIndex, children)
        {            
        }
        #endregion


        #region Property - ComboBox
        private ComboBox m_ComboBox = new ComboBox();
        /// <summary>
        /// Gets or sets the ComboBox.  Lets you access all of the properties of the internal ComboBox.
        /// </summary>
        /// <example>
        /// For example,
        /// <code>
        /// DropDownTreeNode node1 = new DropDownTreeNode("Some text");
        /// node1.ComboBox.Items.Add("Some text");
        /// node1.ComboBox.Items.Add("Some more text");
        /// node1.IsDropDown = true; 
        /// </code>
        /// </example>
        /// <value>The combo box.</value>
        public ComboBox ComboBox
        {
            get
            {
                this.m_ComboBox.DropDownStyle = ComboBoxStyle.DropDownList;
                return this.m_ComboBox;
            }
            set
            {
                this.m_ComboBox = value;
                this.m_ComboBox.DropDownStyle = ComboBoxStyle.DropDownList;
            }
        } 
        #endregion  
    }
}
