using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Windows.Forms;
using System.Diagnostics;
using Microsoft.Surface.Core;
using Microsoft.Surface;
using Microsoft.Surface.Core.Manipulations;

namespace SurfaceCOVER
{
    static class Program
    {

        /// <summary>
        /// The main entry point for the application.
        /// </summary>
        [STAThread]
        static void Main(string[] args)
        {
            // Disable the WinForms unhandled exception dialog.
            // SurfaceShell will notify the user.
            Application.SetUnhandledExceptionMode(UnhandledExceptionMode.ThrowException);
            //Application.Run(new Form1());
            Form1 form = new Form1();
            form.startCOVER(args);
            form.Show();

            // the loop is here to keep app running if non-fatal exception is caught.
            do
            {
                Application.DoEvents();
                form.Frame();
            }
            while (!form.IsDisposed);

        }
        public class Form1 : Form
        {
            private ManagedOpenCOVER.coOpenCOVERWindow openCOVER;
            private ContactTarget contactTarget;

            private ReadOnlyContactCollection previousContacts;
            protected ContactTarget ContactTarget
            {
                get { return contactTarget; }
            }
            // application state: Activated, Previewed, Deactivated,
            // start in Activated state
            //private bool isApplicationActivated = true;
            //private bool isApplicationPreviewed;

            private Affine2DManipulationProcessor manipulationProcessor;
            private Affine2DInertiaProcessor inertiaProcessor;
            private bool manipulating;
            private bool extrapolating;
            

            private List<Manipulator> currentManipulators = new List<Manipulator>();
            private List<Manipulator> removedManipulators = new List<Manipulator>();


            public Form1()
            {
                InitializeSurfaceInput();

                Affine2DManipulations supportedManipulations =
                    Affine2DManipulations.TranslateX | Affine2DManipulations.TranslateY | Affine2DManipulations.Scale;

                manipulationProcessor = new Affine2DManipulationProcessor(supportedManipulations);
                manipulationProcessor.Affine2DManipulationStarted += OnAffine2DManipulationStarted;
                manipulationProcessor.Affine2DManipulationDelta += OnAffine2DDelta;
                manipulationProcessor.Affine2DManipulationCompleted += OnAffine2DManipulationCompleted;

                inertiaProcessor = new Affine2DInertiaProcessor();
                inertiaProcessor.Affine2DInertiaCompleted += OnAffine2DInertiaCompleted;
                inertiaProcessor.Affine2DInertiaDelta += OnAffine2DDelta;

                Visible = true;
                InteractiveSurface interactiveSurface = InteractiveSurface.DefaultInteractiveSurface;
                if (interactiveSurface != null)
                {
                    FormBorderStyle = FormBorderStyle.None;
                }
                UpdateWindowPosition();

                // Set the application's orientation based on the current launcher orientation
                //currentOrientation = ApplicationLauncher.Orientation;

                // Subscribe to surface application activation events
                ApplicationLauncher.ApplicationActivated += OnApplicationActivated;
                ApplicationLauncher.ApplicationPreviewed += OnApplicationPreviewed;
                ApplicationLauncher.ApplicationDeactivated += OnApplicationDeactivated;
            }
            public void startCOVER(String[] args)
            {
                openCOVER = null;
                openCOVER = new ManagedOpenCOVER.coOpenCOVERWindow();
                openCOVER.init(Handle, args);

                ApplicationLauncher.SignalApplicationLoadComplete();
            }
            public void Frame()
            {
                if (openCOVER != null)
                {
                    // Want to identify all the contacts added or removed since the last update.
                    List<Contact> addedContacts = new List<Contact>();
                    List<Contact> removedContacts = new List<Contact>();
                    List<Contact> changedContacts = new List<Contact>();

                    // Get a list of the current contacts
                    ReadOnlyContactCollection currentContacts = contactTarget.GetState();

                    // Compare the contacts in the current list to the list saved from the last update
                    if (previousContacts != null)
                    {
                        foreach (Contact contact in previousContacts)
                        {
                            Contact c = null;
                            currentContacts.TryGetContactFromId(contact.Id, out c);
                            if (c == null)
                            {
                                removedContacts.Add(contact);
                                if(contact.IsFingerRecognized)
                                removedManipulators.Add(new Manipulator(contact.Id, contact.CenterX, contact.CenterY));
                            }
                        }
                        foreach (Contact contact in currentContacts)
                        {
                            Contact c = null;
                            previousContacts.TryGetContactFromId(contact.Id, out c);
                            if (c != null)
                            {
                                changedContacts.Add(contact);
                                if (contact.IsFingerRecognized)
                                currentManipulators.Add(new Manipulator(contact.Id, contact.CenterX, contact.CenterY));
                            }
                            else
                            {
                                addedContacts.Add(contact);
                                if (contact.IsFingerRecognized)
                                currentManipulators.Add(new Manipulator(contact.Id, contact.CenterX, contact.CenterY));
                            }
                        }
                    }
                    else
                    {
                        foreach (Contact c in currentContacts)
                        {
                            addedContacts.Add(c);
                            if (c.IsFingerRecognized)
                            currentManipulators.Add(new Manipulator(c.Id, c.CenterX, c.CenterY));
                        }
                    }


                    manipulationProcessor.ProcessManipulators(currentManipulators, removedManipulators);

                    currentManipulators.Clear();
                    removedManipulators.Clear();

                    previousContacts = currentContacts;

                    // Hit test and assign all new contacts
                    foreach (Contact c in addedContacts)
                    {
                        openCOVER.addedContact(c);
                    }

                    // Update the captors of all the pre-existing contacts
                    foreach (Contact c in changedContacts)
                    {
                        openCOVER.changedContact(c);
                    }

                    // Clean up all old contacts
                    foreach (Contact co in removedContacts)
                    {
                        openCOVER.removedContact(co);
                    }
                    openCOVER.frame();
                    
                }
            }


            #region Manipulation and Inertia Processor Events

            //==========================================================//
            /// <summary>
            /// Event handler for the manipulation processor's delta event. 
            /// Occurs whenever the first time that the manipulation processor processes a 
            /// group of manipulators
            /// </summary>
            /// <param name="sender">The manipulation processor that raised the event</param>
            /// <param name="e">The event args for the event</param>
            private void OnAffine2DManipulationStarted(object sender, Affine2DOperationStartedEventArgs e)
            {
                //Debug.Assert(!extrapolating);
                manipulating = true;

                //    manipulationProcessor.PivotX = transformedCenter.X;
                //    manipulationProcessor.PivotY = transformedCenter.Y;
                //    manipulationProcessor.PivotRadius = Math.Max(Width, Height) / 2.0f;
                
            }

            //==========================================================//
            /// <summary>
            /// Event handler for the manipulation and inertia processor's delta events. 
            /// Occurs whenever the manipulation or inertia processors processes or extrapolate 
            /// manipulator data
            /// </summary>
            /// <param name="sender">The manipulation or inertia processor that raised the event</param>
            /// <param name="e">The event args for the event</param>
            private void OnAffine2DDelta(object sender, Affine2DOperationDeltaEventArgs e)
            {
                //Debug.Assert(manipulating && sender is Affine2DManipulationProcessor ||
                //    extrapolating && sender is Affine2DInertiaProcessor);

                openCOVER.manipulation(e);
                //Vector2 manipulationOrigin = new Vector2(e.ManipulationOriginX, e.ManipulationOriginY);
                //Vector2 manipulationDelta = new Vector2(e.DeltaX, e.DeltaY);
                //Vector2 previousOrigin = new Vector2(manipulationOrigin.X - manipulationDelta.X, manipulationOrigin.Y - manipulationDelta.Y);
                //float restrictedOrientation = RestrictOrientation(e.RotationDelta);
                //float restrictedScale = RestrictScale(e.ScaleDelta);

                // Adjust the position of the item based on change in rotation
              /*  if (restrictedOrientation != 0.0f)
                {
                    Vector2 manipulationOffset = transformedCenter - previousOrigin;
                    Vector2 rotatedOffset = GeometryHelper.RotatePointVector(manipulationOffset, restrictedOrientation);
                    Vector2 compensation = rotatedOffset - manipulationOffset;
                    transformedCenter += compensation;
                }

                // Adjust the position of the item based on change in scale
                if (restrictedScale != 1.0f)
                {
                    Vector2 manipulationOffset = manipulationOrigin - transformedCenter;
                    Vector2 scaledOffset = manipulationOffset * restrictedScale;
                    Vector2 compensation = manipulationOffset - scaledOffset;
                    transformedCenter += compensation;
                }

                // Rotate the item if it is allowed
                if (canRotate || canRotateFlick)
                {
                    orientation += restrictedOrientation;
                }

                // Scale the item if it is allowed
                if (canScale || canScaleFlick)
                {
                    scaleFactor *= restrictedScale;
                }

                // Translate the item if it is allowed
                if (canTranslate || canTranslateFlick)
                {
                    transformedCenter += new Vector2(e.DeltaX, e.DeltaY);
                }

                RestrictCenter();

                if (canRotate || canRotateFlick)
                {
                    manipulationProcessor.PivotX = transformedCenter.X;
                    manipulationProcessor.PivotY = transformedCenter.Y;
                    manipulationProcessor.PivotRadius = Math.Max(Width, Height) / 2.0f;
                }*/
            }

            //==========================================================//
            /// <summary>
            /// Event handler for the manipulation processor's completed event. 
            /// Occurs whenever the manipulation processor processes manipulator 
            /// data where all remaining contacts have been removed.
            /// Check final deltas and start the inertia processor if they are high enough
            /// </summary>
            /// <param name="sender">The manipulation processor that raised the event</param>
            /// <param name="e">The event args for the event</param>
            private void OnAffine2DManipulationCompleted(object sender, Affine2DOperationCompletedEventArgs e)
            {
                // manipulation completed
                manipulating = false;

                // Get the inital inertia values
                //Vector2 initialVelocity = new Vector2(e.VelocityX, e.VelocityY);
                float angularVelocity = e.AngularVelocity;
                float expansionVelocity = e.ExpansionVelocity;

                // Calculate the deceleration rates

                // 4 inches/second squared (4 inches * 96 pixels per inch / (1000 per millisecond)^2)
                const float deceleration = 4.0f * 96.0f / (1000.0f * 1000.0f);
                const float expansionDeceleration = 4.0f * 96.0f / (1000.0f * 1000.0f);

                // 90 degrees/second squared, specified in radians (180 * pi / (1000 per miliseconds)^2)
                const float angularDeceleration = 90.0f / 180.0f * (float)Math.PI / (1000.0f * 1000.0f);

                // Rotate around the center of the item
                //inertiaProcessor.InitialOriginX = TransformedCenter.X;
                //inertiaProcessor.InitialOriginY = TransformedCenter.Y;
                inertiaProcessor.InitialOriginX = e.ManipulationOriginX;
                inertiaProcessor.InitialOriginY = e.ManipulationOriginY;

                // set initial velocity if translate flicks are allowed
                //if (canTranslateFlick)
                //{
                    inertiaProcessor.InitialVelocityX = e.VelocityX;
                    inertiaProcessor.InitialVelocityY = e.VelocityY;
                    inertiaProcessor.DesiredDeceleration = deceleration;
                //}
                /*else
                {
                    inertiaProcessor.InitialVelocityX = 0.0f;
                    inertiaProcessor.InitialVelocityY = 0.0f;
                    inertiaProcessor.DesiredDeceleration = 0.0f;
                }*/


                // set expansion velocity if scale flicks are allowed
                if (Math.Abs(expansionVelocity)> 0.00001 /*&& canScaleFlick*/)
                {
                    inertiaProcessor.InitialExpansionVelocity = expansionVelocity;
                    //inertiaProcessor.InitialRadius = (AxisAlignedBoundingRectangle.Width / 2 + AxisAlignedBoundingRectangle.Height / 2) / 2;
                    inertiaProcessor.InitialRadius = 100;
                    inertiaProcessor.DesiredExpansionDeceleration = expansionDeceleration;
                }
                /*else
                {
                    inertiaProcessor.InitialExpansionVelocity = 0.0f;
                    inertiaProcessor.InitialRadius = 1.0f;
                    inertiaProcessor.DesiredExpansionDeceleration = 0.0f;
                }*/


                // set angular velocity if rotation flicks are allowed
                if (Math.Abs(angularVelocity)> 0.00001 /*&& canRotateFlick*/)
                {
                    inertiaProcessor.InitialAngularVelocity = angularVelocity;
                    inertiaProcessor.DesiredAngularDeceleration = angularDeceleration;
                }
                /*else
                {
                    inertiaProcessor.InitialAngularVelocity = 0.0f;
                    inertiaProcessor.DesiredAngularDeceleration = 0.0f;
                }*/


                // Set the boundaries in which manipulations can occur
                /*inertiaProcessor.LeftBoundary = parent.Left - parent.BoundaryThreshold;
                inertiaProcessor.RightBoundary = parent.Right + parent.BoundaryThreshold;
                inertiaProcessor.TopBoundary = parent.Top - parent.BoundaryThreshold;
                inertiaProcessor.BottomBoundary = parent.Bottom + parent.BoundaryThreshold;
                */
                extrapolating = true;
            }

            //==========================================================//
            /// <summary>
            /// Event handler for the inertia processor's complete event.
            /// Occurs whenever the item comes to rest after being flicked
            /// </summary>
            /// <param name="sender">The inertia processor that raised the event</param>
            /// <param name="e">The event args for the event</param>
            private void OnAffine2DInertiaCompleted(object sender, Affine2DOperationCompletedEventArgs e)
            {
                extrapolating = false;
            }

            #endregion

            /// <summary>
            /// Initializes the surface input system. This should be called after any window
            /// initialization is done, and should only be called once.
            /// </summary>
            private void InitializeSurfaceInput()
            {
                System.Diagnostics.Debug.Assert(Handle != System.IntPtr.Zero,
                    "Window initialization must be complete before InitializeSurfaceInput is called");
                if (Handle == System.IntPtr.Zero)
                    return;
                System.Diagnostics.Debug.Assert(contactTarget == null,
                    "Surface input already initialized");
                if (contactTarget != null)
                    return;

                // Create a target for surface input.
                contactTarget = new ContactTarget(Handle, EventThreadChoice.OnBackgroundThread);
                contactTarget.EnableInput();

            }
            /// <summary>
            /// Use the Desktop bounds to update the position of the Window correctly.
            /// </summary>
            private void UpdateWindowPosition()
            {
                if (InteractiveSurface.DefaultInteractiveSurface != null)
                {
                    SetDesktopBounds(InteractiveSurface.DefaultInteractiveSurface.Left,
                                            InteractiveSurface.DefaultInteractiveSurface.Top, InteractiveSurface.DefaultInteractiveSurface.Width, InteractiveSurface.DefaultInteractiveSurface.Height);
                }
            }

            /// <summary>
            /// This is called when application has been activated.
            /// </summary>
            /// <param name="sender"></param>
            /// <param name="e"></param>
            private void OnApplicationActivated(object sender, EventArgs e)
            {
                // update application state
                //isApplicationActivated = true;
                //isApplicationPreviewed = false;

                //TODO: Enable audio, animations here

                //TODO: Optionally enable raw image here
            }

            /// <summary>
            /// This is called when application is in preview mode.
            /// </summary>
            /// <param name="sender"></param>
            /// <param name="e"></param>
            private void OnApplicationPreviewed(object sender, EventArgs e)
            {
                // update application state
                //isApplicationActivated = false;
                //isApplicationPreviewed = true;

                //TODO: Disable audio here if it is enabled

                //TODO: Optionally enable animations here
            }

            /// <summary>
            ///  This is called when application has been deactivated.
            /// </summary>
            /// <param name="sender"></param>
            /// <param name="e"></param>
            private void OnApplicationDeactivated(object sender, EventArgs e)
            {
                // update application state
                //isApplicationActivated = false;
                //isApplicationPreviewed = false;

                //TODO: Disable audio, animations here

                //TODO: Disable raw image if it's enabled
            }
        }
    }
}

