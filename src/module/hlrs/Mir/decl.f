!=====================================================================
!	This module includes the definiton of matrices
!	and vectors as well as databases for viscoelastic
!	calculations in a regular mesh.
!	Original author: Michael M. Resch
!	First released: 1. November 1994
!	
!	Third revision:
!			date:
!			author: (MR)
!			reason: include an information array
!				that states which results should
!				be written to the output file
!			
!=====================================================================
	module visco_declaration

	use declaration
        use arithmetic

!=====================================================================
!	The variables that provide global information
!
!	today		= todays date
!	problem		= description of problem
!=====================================================================
	character (len = 8)	:: today
	character (len = 120)	:: problem

!=====================================================================
!	A field to store corner cell information
!	corners		= integer filed of indices of corner cells
!=====================================================================
	integer, dimension(48)	:: corners
	integer			:: num_corners
!=====================================================================
!	Some global variables
!	delta_t		  = time_step_size
!	number_time_steps = number of time steps to be calculated
!	relax		  = relaxation parameter
!	unull		  = reference velocity
!	dim 		  = overall number of cells, dimension of 
!			    the problem
!	info		  = just an information integer used as
!			    an error flag
!	output_flags	  = output_flags that specify which 
!			    of the five results to be written
!			    onto the output file
!	p_min,p_max	    = minimum and maximu value of pressure
!	u_min,u_max	    =      -------"---------      u-vel.
!	v_min,v_max	    =      -------"---------      v-vel.
!	sigma_min,sigma_max =      -------"---------      sigma
!	tau_min,tau_max     =      -------"---------      tau
!	gamma_min,gamma_max =      -------"---------      gamma
!       cot_min,cot_max     =      -------"---------      change of type
!	res1_min,res1_max   =      -------"---------      residuum u_vel
!	res2_min,res2_max   =      -------"---------      residuum v_vel
!	res3_min,res3_max   =      -------"---------      residuum sigma
!	res4_min,res4_max   =      -------"---------      residuum tau
!	res5_min,res5_max   =      -------"---------      residuum gamma
!
!	x_max,x_min	  = maximum and minimum x-values of
!			    the geometry described
!	y_max,y_min	  = maximum and minimum y-values of 
!                           the geometry described
!=====================================================================
	real(double)	:: delta_t
	integer 	:: number_time_steps
	real(double)	:: relax
	real(double)	:: unull
	integer		:: dim
	integer		:: info
        integer,dimension(7)	:: output_flags
	real(double)	:: p_min,p_max
	real(double)	:: u_min,u_max
	real(double)	:: v_min,v_max
	real(double)	:: sigma_min,sigma_max
	real(double)	:: tau_min,tau_max
        real(double)	:: gamma_min,gamma_max
        real(double)    :: cot_min,cot_max
	real(double)	:: res1_min,res1_max
	real(double)	:: res2_min,res2_max
	real(double)	:: res3_min,res3_max
	real(double)	:: res4_min,res4_max
	real(double)	:: res5_min,res5_max
	real(double)	:: x_max,x_min
	real(double)	:: y_max,y_min

!=====================================================================
!       added heikex 980206
!       this change_type field is nescessary to calculate
!       cot_min (change of type) and cot_max
!       allocate with allocate(dim) in init.f90 in function visco_read_data  
!       deallocate in visco_write_results after writing type_change to file
!=====================================================================
 
        real(double), allocatable, dimension (:)        :: type_change
        
!=====================================================================
!       definition of a profile for an initialisation of the
!       calculation. this is values and derivatives
!=====================================================================
        
!=====================================================================
!	definition of matrices needed
!	A = matrix of divergence
!	C = gradient matrix
!	AC = product of AC
!=====================================================================
!******* Uwe Laun: Hilfsfelder zur Strukturwandlung der matrix fuer arithmetic-Modul
        integer,      allocatable, dimension(:)         :: col
        integer,      allocatable, dimension(:)         :: row
!******* Uwe Laun: A wird nur zeitweise allokiert; eigentliche Matrix A
!                  wird fuer das arithm.-Modul in A2 abgespeichert
	real(double), allocatable, dimension(:) 	:: A
        type(sparse_matrix)                             :: A2
!******* Uwe Laun: C wird nur zeitweise allokiert; eigentliche Matrix C
!                  wird fuer das arithm.-Modul in C2 abgespeichert
	real(double), allocatable, dimension(:)	        :: C
        type(sparse_matrix)                             :: C2
!******* Uwe Laun: AC wird nicht allokiert; eigentliche Matrix AC
!                  wird fuer das arithm.-Modul in AC2 abgespeichert
	real(double), allocatable, dimension(:,:)	:: AC
        type(sparse_matrix)                             :: AC2

!******* Uwe Laun: Jedem Vektor wird auch ein Typ Vektor des arithmetic-Muduls
!                  (Type vector) zugeordnet und dessen Feldpointer auf das
!                  entsprechende Feld gesetzt; notwendig zur Verwendung der
!                  Vektoren in arithmetic. Die Vektoren wurden ausserdem
!                  zu Pointern umdefiniert, damit man sie auf andere
!                  Felder zeigen lassen kann.

!=====================================================================
!	definition of all vectors needed
!=====================================================================
!	vectors of results
!=====================================================================
!	q vector of velocities and stresses at time level n+1
!	q_old vector of velocities and stresses at time level n
!=====================================================================
	real(double),dimension(:),pointer           :: q,q_old
        type(vector)                                :: q2,q_old2
!=====================================================================
!	vector of pressure
!=====================================================================
!        real(double),dimension(:),pointer           :: p,p_old
        real(double),dimension(:),pointer           :: p
        type(vector)                                :: p2
!=====================================================================
!	vector of B times q
!=====================================================================
	real(double),dimension(:),pointer           :: Bq
        type(vector)                                :: Bq2
!=====================================================================
!	vector of right hand side 
!=====================================================================
	real(double),dimension(:),pointer           :: r
        type(vector)                                :: r2
!=====================================================================
!	auxiliary vectors for the iterative process
!=====================================================================
!	S,T,U	see main program for description
!=====================================================================
	real(double),pointer,dimension(:)           :: S,T,U
        type(vector)                                :: S2,T2,U2
!=====================================================================
!	Vector for pivoting in LU decomposition
!=====================================================================
	integer, pointer, dimension(:)              :: pivot
        
!=====================================================================
!	definition of derived type for one 
!	array of unknowns. these unknowns should
!	be split in the cell to be sure that access is on
!	velocities and stresses rather than unknowns.
!	this makes the program a little bit more clear.
!=====================================================================
	type vec_of_unknowns
	  sequence
	  real(double), dimension(2)	:: velocities
	  real(double), dimension(3)	:: stresses
	end type vec_of_unknowns
!=====================================================================
!	definition of derived type for one 
!	finite volume cell
!=====================================================================
        type cell
	  integer			        :: num_sides    
	  integer, pointer,dimension(:)		:: neighbours
	  real(double), pointer,dimension(:,:)	:: coordinates
	  real(double), dimension(2)	        :: centre
	  real(double), pointer,dimension(:,:)  :: n_vectors
          real(double), pointer,dimension(:)    :: alphas
          real(double)                          :: volume
	  type(vec_of_unknowns)		        :: unknowns
	  real(double)			        :: pressure
          real(double), dimension(4)            :: sum
	end type cell
!=====================================================================
!	definition of an array of derived type for
!	the mesh in use
!=====================================================================
	type(cell), allocatable, dimension(:)	:: cell_data
	type(cell), allocatable, dimension(:)	:: cell_data_1
	integer, allocatable, dimension(:)	:: c_to_c_1
	integer, allocatable, dimension(:)	:: c_1_to_c
!=====================================================================
!	pointer indicating the point in cell_data
!	were volumes of different types start.
!	inflow		inflow volumes
!	c_inflow_2	corner volume at inflow boundary
!	c_inflow_4	corner volume at inflow boundary
!	outflow		outflow volumes
!	c_outflow_2	corner volume at outflow boundary
!	c_outflow_1	corner volume at outflow boundary
!	wall_1		boundary volumes at fixed wall side 1
!	wall_2		boundary volumes at fixed wall side 2
!	wall_3		boundary volumes at fixed wall side 3
!	wall_4		boundary volumes at fixed wall side 4
!	c_wall_12	corner volumes at fixed wall side 1/2
!	c_wall_23	corner volumes at fixed wall side 2/3
!	c_wall_34	corner volumes at fixed wall side 3/4
!	c_wall_41	corner volumes at fixed wall side 4/1
!
!				      /c_wall_12 /c_wall23
!	       		wall_2	     /		/
!		---------------------		|-------|
!  c_inflow_2	|		    |		|	| c_outflow_2
!      	 	|                   |-wall_1	|	|
!      	 	|                   |		|	|
!		|                   |-----------|	|
!		|					|
!		|					|
!	inflow  | 					| outflow
!		|		      			|
!		|		     			|
!		|		    |-----------|	|
!		|		    |  		|	|
!		|		    |  		|	|
!      	 	|                   |    wall_3-|	|
!      	 	|                   |		|	|
!  c_inflow_4	|		    |		|	| c_outflow_4
!		|-------------------|		|-------|
!		   	  wall_4     \           \
!				      \c_wall_41  \c_wall_34
!
!=====================================================================
	integer 	:: inflow, c_inflow_2, c_inflow_4
	integer 	:: outflow, c_outflow_2, c_outflow_4
	integer		:: wall_1,wall_2,wall_3,wall_4
	integer		:: c_wall_12,c_wall_23,c_wall_34,c_wall_41
!=====================================================================
!	definition of several variables to describe the structure
!	of the blocks in use.
!=====================================================================
	integer         :: num_blocks
!=====================================================================
!	definition of a type called block
!	includes the description of a block
!	b_number		internal number of the block
!	b_coordinates		coordinates of the four corner
!				points of the block. is a
!				(2,4) array.
!	number_cells		#of cells in the two directions.
!				is a (2) array.
!	enlargement		the factor that describes how
!				the size of the last cell depends
!				on the size of the first one by
!				s_l = s_1*2^enlargement.
!				is a (2) array.
!	factor			the factor that describes how
!				the size of consecutive cells changes
!	neighbours		neighbouring blocks.
!				if no neighbour is given value is set
!				to -1.
!				is a (4) array.
!	initial_cell		the initial number of the cells 
!				inside the block.
!	init_cell_size		size of the first cell on one side
!=====================================================================
	type block
	  integer			:: b_number
	  real(double), dimension(2,4)	:: b_coordinates
	  integer, dimension(2)		:: number_cells
	  real(double), dimension(2)	:: enlargement
	  real(double), dimension(2)	:: factor
	  integer, dimension(4)		:: neighbours
	  integer			:: initial_cell
	  real(double), dimension(2)	:: init_cell_size
	end type block
!=====================================================================
!	now define an array of blocks
!	the array is allocatable to allow a flexible
!	number of blocks in use
!=====================================================================
	type(block), dimension(:), allocatable	:: blocks
!=====================================================================
!	some problem parameters that are used
!	lambda    = relaxation time
!	viscosity = viscoelastic viscosity
!=====================================================================
	real(double)	:: lambda
	real(double)	:: viscosity
!=====================================================================
!       a checking parameter for initialisation of A
!=====================================================================
        integer :: check

	end module visco_declaration
