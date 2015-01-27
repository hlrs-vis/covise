!=======================================================================
!	this module will create a mesh by reading some
!	information about blocks that the mesh is
!	made up of.
!	Original author: Michael M. Resch (MR)
!	First released: 1. November 1994
!	First revision: 
!			date: 15. November 1994
!			author: (MR)
!			reason: modification of calculation of cells by
!				incorporating an enlargement factor that
!				describes how the size of a cell changes
!				from one end of the side to the other.
!	Second revision:
!			date: 20. November 1994
!			author: (MR)
!			reason:	connecting several blocks together
!				taking into account the way the cells are
!				calculated by enlargement factors.
!	Third revision:
!			date: 1. December 1994
!			author: (MR)
!			reason: reorganisation of boundary cells
!			        necessary when using several blocks
!	Fourth Revision:
!			date: 1. February 1995
!			author: (MR)
!			reason: output according to Andreas Baric
!				and Wolfgang Lohr
!
!=======================================================================
	module visco_mesh
	contains

!=======================================================================
	subroutine visco_read_mesh()
!=======================================================================
!	this subroutine reads the information about
!	the mesh in blocks
!=======================================================================
	use visco_declaration
	use visco_control

	implicit none

!=======================================================================
!	i,j,k.l			running indices for loops
!	counter			running index for counting cells
!	x_coord,y_coord 	reference point for one cell
!				to calculate coordinates of points
!				when running through an x-row.
!	x_coord_old,y_coord_old reference point for finding the 
!				starting point for a y-column.
!	x_abs,y_abs		length of the block in x,y-direction
!				respectively
!	x_factor,y_factor	factor describing the difference in size
!				of consecutive cells.
!	x_vec,y_vec		vector giving the direction of
!				x,y-side of the block respectively.
!	init,x_dim,y_dim	auxiliary variables to store
!				information on neighbouring blocks.
!				they are just used to better understand
!				teh written code.
!=======================================================================
	integer		            :: i,j,k,l,counter
	real(double)	            :: delta_x, delta_y, delta_x_old
	real(double)	            :: x_coord, y_coord, x_coord_old, y_coord_old
	real(double)	            :: x_abs, y_abs, x_factor, y_factor
	real(double), dimension(2)  :: x_vec,y_vec
	integer		            :: init, x_dim, y_dim

!=======================================================================
!	First read the description of the problem
!=======================================================================
	read(70,*) problem
        read(70,'(A)') problem
        write(*,*) '========= description of problem ============'
        write(*,*) problem
        write(*,*) '============================================='
!=======================================================================
!	read number of blocks that shall be used.
!=======================================================================
	write(*,*) 'number of blocks wanted'
	read(70,*) num_blocks
	write(*,*) num_blocks
	allocate(blocks(num_blocks),STAT=info)
!=======================================================================
!	read number of cells for each block and add up total
!	number of cells in use.
!=======================================================================
	dim = 0
	read_info: do i = 1,num_blocks
!=======================================================================
!	read block number, topology and size of block
!=======================================================================
	  write(*,*) 'number of block'
	  read(70,*) blocks(i)%b_number
	  write(*,*) blocks(i)%b_number
	  write(*,*) '4 neighbours of the block'
	  read(70,*) (blocks(i)%neighbours(j),j=1,4)
	  write(*,*) (blocks(i)%neighbours(j),j=1,4)
	  write(*,*) 'number of cells in x/y-direction for block',i
	  read(70,*) blocks(i)%number_cells(1),				&
		    blocks(i)%number_cells(2)
	  write(*,*) blocks(i)%number_cells(1),                          &
                    blocks(i)%number_cells(2)
	  dim = dim + blocks(i)%number_cells(1) *			&
		      blocks(i)%number_cells(2)
!=======================================================================
!	read enlargement factor for calculation of
!	size of cells along one side of the block
!=======================================================================
	  write(*,*) 'Enlargement factor for x_direction'
	  read(70,*) blocks(i)%enlargement(1)
	  write(*,*) blocks(i)%enlargement(1)
	  if (blocks(i)%neighbours(2) .ne. -1) then
	    if (blocks(i)%neighbours(2) .lt. i)				&
	    blocks(i)%enlargement(1) = 					&
		blocks(blocks(i)%neighbours(2))%enlargement(1)
	  else if (blocks(i)%neighbours(4).ne. -1) then                     
	    if (blocks(i)%neighbours(4) .lt. i)				&
            blocks(i)%enlargement(1) =                                  &
                blocks(blocks(i)%neighbours(4))%enlargement(1)
	  end if
	  write(*,*) 'Enlargement factor for y_direction'
	  read(70,*) blocks(i)%enlargement(2)
	  write(*,*) blocks(i)%enlargement(2)
	  if (blocks(i)%neighbours(1) .ne. -1) then
	    if (blocks(i)%neighbours(1) .lt. i)				&
	    blocks(i)%enlargement(2) = 					&
		blocks(blocks(i)%neighbours(1))%enlargement(2)
	  else if (blocks(i)%neighbours(3).ne. -1) then                     
	    if (blocks(i)%neighbours(3) .lt. i)				&
            blocks(i)%enlargement(2) =                                  &
                blocks(blocks(i)%neighbours(3))%enlargement(2)
	  end if
!=======================================================================
!	read coordinates of each block.
!=======================================================================
	  write(*,*) 'coordinates of left lower node for block',i
	  read(70,*) (blocks(i)%b_coordinates(j,1),j=1,2)
	  write(*,*) (blocks(i)%b_coordinates(j,1),j=1,2)
	  write(*,*) 'coordinates of right lower node for block',i
	  read(70,*) (blocks(i)%b_coordinates(j,2),j=1,2)
	  write(*,*) (blocks(i)%b_coordinates(j,2),j=1,2)
	  write(*,*) 'coordinates of right upper node for block',i
	  read(70,*) (blocks(i)%b_coordinates(j,3),j=1,2)
	  write(*,*) (blocks(i)%b_coordinates(j,3),j=1,2)
	end do read_info
!=======================================================================
!	according to number of cells calculated allocate
!	memory for mesh-array.
!=======================================================================
	write(*,*) 'dimension of overall problem'
	write(*,*) dim
	allocate(cell_data(-1:dim),STAT=info)
	if (info .ne. 0) then
	  write(*,*) 'cell_data could not be allocated'
	end if
	allocate(cell_data_1(-1:dim),STAT=info)
	if (info .ne. 0) then
	  write(*,*) 'cell_data_1 could not be allocated'
	end if
	allocate(c_to_c_1(dim),STAT=info)
	if (info .ne. 0) then
	  write(*,*) 'c_to_c_1 could not be allocated'
	end if
	allocate(c_1_to_c(dim),STAT=info)
	if (info .ne. 0) then
	  write(*,*) 'c_1_to_c could not be allocated'
	end if
!=======================================================================
!	initialize the dummy cell -1 and 0
!=======================================================================
        cell_data_1(0)%num_sides = 0
	allocate(cell_data_1(0)%neighbours(4),STAT=info)
        cell_data_1(0)%neighbours(1) = -1
        cell_data_1(0)%neighbours(2) = -1
        cell_data_1(0)%neighbours(3) = -1
        cell_data_1(0)%neighbours(4) = -1
	allocate(cell_data_1(0)%coordinates(2,4),STAT=info)
	allocate(cell_data_1(0)%n_vectors(2,4),STAT=info)
        cell_data_1(0)%centre(1) = 0.0d0
        cell_data_1(0)%centre(2) = 0.0d0
        cell_data_1(0)%volume = 0.0d0
        cell_data_1(0)%unknowns%velocities(1) = 0.0d0
        cell_data_1(0)%unknowns%velocities(2) = 0.0d0
        cell_data_1(0)%unknowns%stresses(1)   = 0.0d0
        cell_data_1(0)%unknowns%stresses(2)   = 0.0d0
        cell_data_1(0)%unknowns%stresses(3)   = 0.0d0
        cell_data_1(-1)%num_sides = 0
	allocate(cell_data_1(-1)%neighbours(4),STAT=info)
        cell_data_1(-1)%neighbours(1) = -1
        cell_data_1(-1)%neighbours(2) = -1
        cell_data_1(-1)%neighbours(3) = -1
        cell_data_1(-1)%neighbours(4) = -1
	allocate(cell_data_1(-1)%coordinates(2,4),STAT=info)
	allocate(cell_data_1(-1)%n_vectors(2,4),STAT=info)
        cell_data_1(-1)%centre(1) = 0.0d0
        cell_data_1(-1)%centre(2) = 0.0d0
        cell_data_1(-1)%volume = 0.0d0
        cell_data_1(-1)%unknowns%velocities(1) = 0.0d0
        cell_data_1(-1)%unknowns%velocities(2) = 0.0d0
        cell_data_1(-1)%unknowns%stresses(1)   = 0.0d0
        cell_data_1(-1)%unknowns%stresses(2)   = 0.0d0
        cell_data_1(-1)%unknowns%stresses(3)   = 0.0d0
!=======================================================================
!	for each block now generate a rectangular mesh
!=======================================================================
	counter = 1
	generate: do i = 1,num_blocks
!=======================================================================
!	at the moment we will only allow parallelograms.
!	thus the fourth point is a fixed one and can
!	be calculated using points 2 and 3.
!=======================================================================
	  y_vec(1) = blocks(i)%b_coordinates(1,3) -		&
		     blocks(i)%b_coordinates(1,2)
	  y_vec(2) = blocks(i)%b_coordinates(2,3) -		&
		     blocks(i)%b_coordinates(2,2)
	  blocks(i)%b_coordinates(1,4) = blocks(i)%b_coordinates(1,1) + &
					y_vec(1)
	  blocks(i)%b_coordinates(2,4) = blocks(i)%b_coordinates(2,1) + &
					y_vec(2)
!=======================================================================
!	initialise starting point as the lower left corner
!	of the block.
!=======================================================================
	  x_coord_old = blocks(i)%b_coordinates(1,1)
	  y_coord_old = blocks(i)%b_coordinates(2,1)
!=======================================================================
!	calculate directions of lines of the grid.
!	and calculate stepsize for the mesh (delta_x, delta_y)
!=======================================================================
	  x_vec(1) = blocks(i)%b_coordinates(1,2) -		&
		     blocks(i)%b_coordinates(1,1)
	  x_vec(2) = blocks(i)%b_coordinates(2,2) -		&
		     blocks(i)%b_coordinates(2,1)
!=======================================================================
!	normalize x_vector
!=======================================================================
	  x_abs = sqrt(x_vec(1)**2 + x_vec(2)**2)
	  x_vec(1) = x_vec(1)/x_abs
	  x_vec(2) = x_vec(2)/x_abs
!=======================================================================
!	normalize x_vector
!=======================================================================
	  y_abs = sqrt(y_vec(1)**2 + y_vec(2)**2)
	  y_vec(1) = y_vec(1)/y_abs
	  y_vec(2) = y_vec(2)/y_abs
!=======================================================================
!	calculate the multiplication factor for the calculation
!	of varying cell-lengths. additionally calculate
!	the initial cell size.
!=======================================================================
	  if ((blocks(i)%neighbours(2) .ne. -1) .and. 			&
	    (blocks(i)%neighbours(2) .lt. i)) then
	    blocks(i)%factor(1) =					&
                blocks(blocks(i)%neighbours(2))%factor(1)
	    x_factor = blocks(i)%factor(1)
	    blocks(i)%init_cell_size(1) = 				&
		blocks(blocks(i)%neighbours(2))%init_cell_size(1)
	    delta_x = blocks(i)%init_cell_size(1)
	    blocks(i)%number_cells(1) = 				&
	    	 blocks(blocks(i)%neighbours(2))%number_cells(1)
!	    if (x_factor .ne. 1.0) then
!              blocks(i)%number_cells(1) = 				&
!			log(1-x_abs*(1-x_factor)/delta_x)/		&
!			log(x_factor)
!	    else
!	      blocks(i)%number_cells(1) = x_abs/delta_x
!	    end if
	  else if ((blocks(i)%neighbours(4) .ne. -1) .and.		&
                   (blocks(i)%neighbours(4) .lt. i)) then
            blocks(i)%factor(1) =					&
                blocks(blocks(i)%neighbours(4))%factor(1)
            x_factor = blocks(i)%factor(1)
	    blocks(i)%init_cell_size(1) = 				&
		blocks(blocks(i)%neighbours(4))%init_cell_size(1)
	    delta_x = blocks(i)%init_cell_size(1)
	    blocks(i)%number_cells(1) = 				&
	    	 blocks(blocks(i)%neighbours(4))%number_cells(1)
!	    if (x_factor .ne. 1.0) then
!              blocks(i)%number_cells(1) = 				&
!			log(1-x_abs*(1-x_factor)/delta_x)/		&
!			log(x_factor)
!	    else
!	      blocks(i)%number_cells(1) = x_abs/delta_x
!	    end if
!=======================================================================
!	if there is no neighbour available where this block
!	could copy his x_factor from
!=======================================================================
	  else 
	    x_factor = 2**(blocks(i)%enlargement(1)/			&
			 (blocks(i)%number_cells(1)-1))
	    blocks(i)%factor(1) = x_factor
	    if (x_factor .ne. 1.0) then
	      delta_x = x_abs*(1-x_factor)/				&
			  (1 - x_factor**blocks(i)%number_cells(1))
	      blocks(i)%init_cell_size(1) = delta_x
	    else
	      delta_x = x_abs/blocks(i)%number_cells(1)
	      blocks(i)%init_cell_size(1) = delta_x
	    end if
!=======================================================================
!	endif of checking for existing neighbours 
!	and their x_factors
!=======================================================================
	  end if
	  delta_x_old = delta_x
	  if ((blocks(i)%neighbours(1) .ne. -1) .and. 			&
	      (blocks(i)%neighbours(1) .lt. i)) then
	    blocks(i)%factor(2) =					&
                blocks(blocks(i)%neighbours(1))%factor(2)
	    y_factor = blocks(i)%factor(2)
	    blocks(i)%init_cell_size(2) = 				&
		blocks(blocks(i)%neighbours(1))%init_cell_size(2)
	    delta_y = blocks(i)%init_cell_size(2) 
	    blocks(i)%number_cells(2) = 				&
	    	 blocks(blocks(i)%neighbours(1))%number_cells(2)
!	    if (y_factor .ne. 1.0) then
!              blocks(i)%number_cells(2) = 				&
!			log(1-y_abs*(1-y_factor)/delta_y)/		&
!			log(y_factor)
!	    else
!	      blocks(i)%number_cells(2) = y_abs/delta_y
!	    end if
	  else if ((blocks(i)%neighbours(3) .ne. -1) .and.		&
                   (blocks(i)%neighbours(3) .lt. i)) then
            blocks(i)%factor(2) =                                     &
                blocks(blocks(i)%neighbours(3))%factor(2)
            y_factor = blocks(i)%factor(2)
	    blocks(i)%init_cell_size(2) = 				&
		blocks(blocks(i)%neighbours(3))%init_cell_size(2)
	    delta_y = blocks(i)%init_cell_size(2) 
	    blocks(i)%number_cells(2) = 				&
	    	 blocks(blocks(i)%neighbours(3))%number_cells(2)
!	    if (y_factor .ne. 1.0) then
!              blocks(i)%number_cells(2) = 				&
!			log(1-y_abs*(1-y_factor)/delta_y)/		&
!			log(y_factor)
!	    else
!	      blocks(i)%number_cells(2) = y_abs/delta_y
!	    end if
!=======================================================================
!	if there is no neighbour available where this block
!	could copy his y_factor from
!=======================================================================
	  else
	    y_factor = 2**(blocks(i)%enlargement(2)/			&
			 (blocks(i)%number_cells(2)-1))
	    blocks(i)%factor(2) = y_factor
	    if (y_factor .ne. 1.0) then
	      delta_y = y_abs*(1-y_factor)/				&
			  (1 - y_factor**blocks(i)%number_cells(2))
	      blocks(i)%init_cell_size(2) = delta_y
	    else
	      delta_y = y_abs/blocks(i)%number_cells(2)
	      blocks(i)%init_cell_size(2) = delta_y
	    end if
!=======================================================================
!	endif of checking for existing neighbours 
!	and their x_factors
!=======================================================================
	  end if
	  blocks(i)%initial_cell = counter
!=======================================================================
!	Loop over the two directions of the block for calculation of
!	cell sizes.
!=======================================================================
	  y_direction: do j=1,blocks(i)%number_cells(2)
	    x_coord = x_coord_old
	    y_coord = y_coord_old
	    delta_x = delta_x_old
	    x_direction: do k = 1,blocks(i)%number_cells(1)
!=======================================================================
!	calculate coordinates of four cell nodes
!=======================================================================
	      allocate(cell_data(counter)%neighbours(4),STAT=info)
	      allocate(cell_data(counter)%alphas(4),STAT=info)
	      allocate(cell_data(counter)%coordinates(2,4),STAT=info)
	      allocate(cell_data(counter)%n_vectors(2,4),STAT=info)
	      cell_data(counter)%num_sides = 4
	      cell_data(counter)%coordinates(1,1) =			&
			x_coord + delta_x*x_vec(1)
	      cell_data(counter)%coordinates(2,1) =			&
			y_coord + delta_x*x_vec(2)

	      cell_data(counter)%coordinates(1,2) =			&
			x_coord + delta_x*x_vec(1) +			&
				  delta_y*y_vec(1)
	      cell_data(counter)%coordinates(2,2) =			&
			y_coord + delta_x*x_vec(2) +			&
				  delta_y*y_vec(2)

	      cell_data(counter)%coordinates(1,3) =			&
			x_coord + delta_y*y_vec(1)
	      cell_data(counter)%coordinates(2,3) =			&
			y_coord + delta_y*y_vec(2)
!=======================================================================
!	for the first cell in one x-row the coordinates of the 
!	3rd point mark the beginning of the next x-row.
!=======================================================================
	      if (k .eq. 1) then
		x_coord_old = cell_data(counter)%coordinates(1,3)
		y_coord_old = cell_data(counter)%coordinates(2,3)
	      end if

	      cell_data(counter)%coordinates(1,4) = x_coord
	      cell_data(counter)%coordinates(2,4) = y_coord

	      x_coord = cell_data(counter)%coordinates(1,1)
	      y_coord = cell_data(counter)%coordinates(2,1)

	      delta_x = delta_x*x_factor
!=======================================================================
!	calculation of centre point coordinates 
!=======================================================================
	      cell_data(counter)%centre(1) 	  =	&
	       (cell_data(counter)%coordinates(1,1) +	&
		cell_data(counter)%coordinates(1,2) +	&
		cell_data(counter)%coordinates(1,3) +	&
		cell_data(counter)%coordinates(1,4))/4.0
	      cell_data(counter)%centre(2) 	  =	&
	       (cell_data(counter)%coordinates(2,1) +	&
		cell_data(counter)%coordinates(2,2) +	&
		cell_data(counter)%coordinates(2,3) +	&
		cell_data(counter)%coordinates(2,4))/4.0
!=======================================================================
!	calculate neighbours of each cell
!	neighbour = -1	no neighbour exists/ cell is boundary cell
!=======================================================================
	      cell_data(counter)%neighbours(4) = &
			counter - blocks(i)%number_cells(1) 
	      cell_data(counter)%neighbours(1) = counter+1
	      cell_data(counter)%neighbours(2) = &
			counter + blocks(i)%number_cells(1) 
	      cell_data(counter)%neighbours(3) = counter-1
!=======================================================================
!	take into account boundary cells if they are not
!	near to another block.
!=======================================================================
!=======================================================================
!	lower block boundary
!=======================================================================
	      if (j .eq. 1) then
		if ((blocks(i)%neighbours(4) .eq. -1) .or.		&
		    (blocks(i)%neighbours(4) .gt. i)) then
	          cell_data(counter)%neighbours(4) = -1
		else
	          init	= blocks(blocks(i)%neighbours(4))%initial_cell
		  x_dim	= blocks(blocks(i)%neighbours(4))%number_cells(1)
		  y_dim	= blocks(blocks(i)%neighbours(4))%number_cells(2)
		  cell_data(counter)%neighbours(4) =			&
			init+x_dim*(y_dim-1)+k-1
		  cell_data(init+x_dim*(y_dim-1)+k-1)%neighbours(2) = counter
		end if
	      end if
!=======================================================================
!	upper block boundary
!=======================================================================
	      if(j .eq. blocks(i)%number_cells(2)) then
		if ((blocks(i)%neighbours(2) .eq. -1) .or.		&
		    (blocks(i)%neighbours(2) .gt. i)) then
	          cell_data(counter)%neighbours(2) = -1
		else
	          init	= blocks(blocks(i)%neighbours(2))%initial_cell
		  x_dim	= blocks(blocks(i)%neighbours(2))%number_cells(1)
		  y_dim	= blocks(blocks(i)%neighbours(2))%number_cells(2)
		  cell_data(counter)%neighbours(2) =			&
			init+k-1
		  cell_data(init+k-1)%neighbours(4) = counter
		end if
	      end if
!=======================================================================
!	left block boundary
!=======================================================================
	      if(k .eq. 1) then
		if ((blocks(i)%neighbours(3) .eq. -1) .or.		&
		    (blocks(i)%neighbours(3) .gt. i)) then
	          cell_data(counter)%neighbours(3) = -1
		else
	          init	= blocks(blocks(i)%neighbours(3))%initial_cell
		  x_dim	= blocks(blocks(i)%neighbours(3))%number_cells(1)
		  y_dim	= blocks(blocks(i)%neighbours(3))%number_cells(2)
		  cell_data(counter)%neighbours(3) =			&
			init+j*x_dim-1
		  cell_data(init+j*x_dim-1)%neighbours(1) = counter
		end if
	      end if
!=======================================================================
!	right block boundary
!=======================================================================
	      if(k .eq. blocks(i)%number_cells(1)) then
		if ((blocks(i)%neighbours(1) .eq. -1) .or.		&
		    (blocks(i)%neighbours(1) .gt. i)) then
	          cell_data(counter)%neighbours(1) = -1
		else
	          init	= blocks(blocks(i)%neighbours(1))%initial_cell
		  x_dim	= blocks(blocks(i)%neighbours(1))%number_cells(1)
		  y_dim	= blocks(blocks(i)%neighbours(1))%number_cells(2)
	          cell_data(counter)%neighbours(1) =			&
			init+(j-1)*x_dim
		  cell_data(init+(j-1)*x_dim)%neighbours(3) = counter
		end if
	      end if
!=======================================================================
!	calculate normal vectors
!=======================================================================
	      do l = 1,3
		cell_data(counter)%n_vectors(1,l) =		&
		cell_data(counter)%coordinates(2,l+1) - 	&
		cell_data(counter)%coordinates(2,l)
		cell_data(counter)%n_vectors(2,l) =		&
		cell_data(counter)%coordinates(1,l) - 		&
		cell_data(counter)%coordinates(1,l+1)
	      end do
	      cell_data(counter)%n_vectors(1,4) =		&
	      cell_data(counter)%coordinates(2,1) - 	&
	      cell_data(counter)%coordinates(2,4)
	      cell_data(counter)%n_vectors(2,4) =		&
	      cell_data(counter)%coordinates(1,4) - 		&
	      cell_data(counter)%coordinates(1,1)
!=======================================================================
!	calculate volume
!=======================================================================
              call calc_cell_volume(counter)
!=======================================================================
!	increment counter
!=======================================================================
	      counter = counter + 1
	    end do x_direction
	    delta_y = delta_y*y_factor
	  end do y_direction
	end do generate
		
	end subroutine visco_read_mesh

!=======================================================================
	subroutine visco_sort_data()
!=======================================================================
!	this subroutine sorts the volumes to make sure
!	that boundary-volumes are stored first.
!=======================================================================
	use visco_declaration
	use visco_control

	implicit none

	integer		:: i, j
	integer		:: counter,boundary
	integer		:: loc_counter
	integer		:: help
	type(cell)	:: cell_help

!=======================================================================
!	initialize corner information
!=======================================================================
	corners = -1
	num_corners = 0
!=======================================================================
!	assume that the first volume is always the lower
!	edge volume of the inflow boundary
!=======================================================================
	if (cell_data(1)%neighbours(3)*cell_data(1)%neighbours(4) .eq. 1) &
	cell_data_1(1) 	= cell_data(1)
	c_to_c_1(1) = 1
	c_1_to_c(1) = 1
	i = 1
	c_inflow_4 	= 1
	i 		= cell_data(1)%neighbours(2)
	counter 	= 1
!=======================================================================
!	now we asume that there is one single inflow boundary
!=======================================================================
	do while (cell_data(i)%neighbours(2) .ne. -1)
	  counter 		= counter + 1
	  cell_data_1(counter) 	= cell_data(i)
	  c_to_c_1(i) 		= counter
	  c_1_to_c(counter) 	= i
	  i 			= cell_data(i)%neighbours(2)
        end do
!=======================================================================
!	we may now set the c_inflow counter since no more
!	inflow volumes have to be expected.
!=======================================================================
	inflow 			= counter
	counter 		= counter + 1
	cell_data_1(counter) 	= cell_data(i)
	c_to_c_1(i) 		= counter
	c_1_to_c(counter) 	= i
	c_inflow_2 		= counter

!=======================================================================
!	now right of this corner volume there has to be 
!	a number of wall volumes of type wall_2
!=======================================================================
	i 		= cell_data(i)%neighbours(1)
	counter 		= counter+1
	cell_data_1(counter) 	= cell_data(i)
	c_to_c_1(i) 		= counter
	c_1_to_c(counter) 	= i
	i 			= cell_data(i)%neighbours(1)
	x_max = -1e+05
	x_min = 1e+05
	y_max = -1e+05
	y_min = 1e+05

	boundary_search: do while(i .ne. 1) 
          if ((cell_data(i)%neighbours(1) .eq. -1) .and. 		&
	      (cell_data(i)%neighbours(4) .ne. -1)) then
!------------------------------------------------------------------------
!  found a right upper corner. move down.
!------------------------------------------------------------------------
	    counter 			= counter + 1
	    cell_data_1(counter) 	= cell_data(i)
	    c_1_to_c(counter) 		= i
	    c_to_c_1(i) 		= counter
	    i = cell_data(i)%neighbours(4)
	  else if (cell_data(i)%neighbours(2) .eq. -1) then
!------------------------------------------------------------------------
!  upper boundary. move on to the right
!------------------------------------------------------------------------
	    counter 		= counter + 1
	    cell_data_1(counter) 	= cell_data(i)
	    c_1_to_c(counter) 	= i
	    c_to_c_1(i) 		= counter
	    i = cell_data(i)%neighbours(1)
	  else if (cell_data(i)%neighbours(3) .eq. -1) then
!------------------------------------------------------------------------
!  left boundary. move up.
!------------------------------------------------------------------------
	    counter 		= counter + 1
	    cell_data_1(counter) 	= cell_data(i)
	    c_1_to_c(counter) 	= i
	    c_to_c_1(i) 		= counter
	    i = cell_data(i)%neighbours(2)
	  else if (cell_data(i)%neighbours(4) .eq. -1) then
!------------------------------------------------------------------------
!  lower boundary. move left.
!------------------------------------------------------------------------
	    counter 		= counter + 1
	    cell_data_1(counter) 	= cell_data(i)
	    c_1_to_c(counter) 	= i
	    c_to_c_1(i) 		= counter
	    i = cell_data(i)%neighbours(3)
	  else
	    if (cell_data(cell_data(i)%neighbours(1))%neighbours(2)	&
		.eq. -1) then
!------------------------------------------------------------------------
!  angle from right to upper boundary. 
!  register actual cell, upper cell and right hand cell
!  as part of a corner configuration. then move right.
!------------------------------------------------------------------------
              num_corners = num_corners+1
              corners(num_corners) = i
	      i = cell_data(i)%neighbours(1)
	    else if (cell_data(cell_data(i)%neighbours(2))%neighbours(3)&
	        .eq. -1) then
!------------------------------------------------------------------------
!  angle from upper to left boundary. 
!  register actual cell, upper cell and left hand cell
!  as part of corner configuration. then move up.
!------------------------------------------------------------------------
              num_corners = num_corners+1
              corners(num_corners) = i
	      i =  cell_data(i)%neighbours(2)
	    else if (cell_data(cell_data(i)%neighbours(3))%neighbours(4)&
	        .eq. -1) then
!------------------------------------------------------------------------
!  angle from left to lower boundary.
!  register actual cell, lower and left cell
!  as part of corner configuration. then move left
!------------------------------------------------------------------------
              num_corners = num_corners+1
              corners(num_corners) = i
              i =  cell_data(i)%neighbours(3)
	    else if (cell_data(cell_data(i)%neighbours(4))%neighbours(1)&
                .eq. -1) then
!------------------------------------------------------------------------
!  angle from lower to right boundary. 
!  register actual cell, right and lower cell
!  as part of corner configuration. then move down
!------------------------------------------------------------------------
              num_corners = num_corners+1
              corners(num_corners) = i
	      i =  cell_data(i)%neighbours(4)
	    else
	      write(*,*) '(I) there is a bug you should kill urgently!!!'
	    end if
	  end if
	  x_max = max(x_max,cell_data(i)%centre(1))
	  x_min = min(x_min,cell_data(i)%centre(1))
	  y_max = max(y_max,cell_data(i)%centre(2))
	  y_min = min(y_min,cell_data(i)%centre(2))
	end do boundary_search


!=======================================================================
!	now we would like to collect all boundary cells
!	of the same type
!=======================================================================
	loc_counter = c_inflow_2
!=======================================================================
!	find outflow boundaries
!	starting with upper corner cell
!=======================================================================
	loop_out_2: do i = c_inflow_2,counter-1
	  if ((cell_data_1(i+1)%neighbours(1) .eq. -1) .and.		&
	      (cell_data_1(i+1)%neighbours(2) .eq. -1) .and.		&
	      (cell_data_1(i+1)%centre(1) .eq. x_max)) then
	    loc_counter = loc_counter+1
	    if (loc_counter .ne. i+1) then
	      help = c_1_to_c(loc_counter) 
	      c_1_to_c(loc_counter) = c_1_to_c(i+1)
	      c_1_to_c(i+1) = help
	      help = c_to_c_1(c_1_to_c(loc_counter)) 
	      c_to_c_1(c_1_to_c(loc_counter)) = c_to_c_1(c_1_to_c(i+1))
	      c_to_c_1(c_1_to_c(i+1)) = help
	      cell_help = cell_data_1(loc_counter) 
	      cell_data_1(loc_counter) = cell_data_1(i+1)
	      cell_data_1(i+1) = cell_help
	    end if
	  end if
	end do loop_out_2
	c_outflow_2 = loc_counter
!=======================================================================
!	now find all outflow nodes
!=======================================================================
	loop_outflow: do i = c_outflow_2,counter-1
	  if ((cell_data_1(i+1)%neighbours(1) .eq. -1) .and.		&
	      (cell_data_1(i+1)%neighbours(2) .ne. -1) .and.		&
	      (cell_data_1(i+1)%neighbours(4) .ne. -1) .and.		&
	      (cell_data_1(i+1)%centre(1) .eq. x_max)) then
	    loc_counter = loc_counter+1
	    if (loc_counter .ne. i+1) then
	      help = c_1_to_c(loc_counter) 
	      c_1_to_c(loc_counter) = c_1_to_c(i+1)
	      c_1_to_c(i+1) = help
	      help = c_to_c_1(c_1_to_c(loc_counter)) 
	      c_to_c_1(c_1_to_c(loc_counter)) = c_to_c_1(c_1_to_c(i+1))
	      c_to_c_1(c_1_to_c(i+1)) = help
	      cell_help = cell_data_1(loc_counter) 
	      cell_data_1(loc_counter) = cell_data_1(i+1)
	      cell_data_1(i+1) = cell_help
	    end if
	  end if
	end do loop_outflow
	outflow = loc_counter
!=======================================================================
!	now find lower corner cell
!	this may be improved!!!!
!=======================================================================
	loop_out_4: do i = outflow,counter-1
	  if ((cell_data_1(i+1)%neighbours(1) .eq. -1) .and.		&
	      (cell_data_1(i+1)%neighbours(4) .eq. -1) .and.		&
	      (cell_data_1(i+1)%centre(1) .eq. x_max)) then
	    loc_counter = loc_counter+1
	    if (loc_counter .ne. i+1) then
	      help = c_1_to_c(loc_counter) 
	      c_1_to_c(loc_counter) = c_1_to_c(i+1)
	      c_1_to_c(i+1) = help
	      help = c_to_c_1(c_1_to_c(loc_counter)) 
	      c_to_c_1(c_1_to_c(loc_counter)) = c_to_c_1(c_1_to_c(i+1))
	      c_to_c_1(c_1_to_c(i+1)) = help
	      cell_help = cell_data_1(loc_counter) 
	      cell_data_1(loc_counter) = cell_data_1(i+1)
	      cell_data_1(i+1) = cell_help
	    end if
	  end if
	end do loop_out_4
	c_outflow_4 = loc_counter
!=======================================================================
!	find all wall_2 cells
!=======================================================================
	loop_wall_2: do i = c_outflow_4,counter-1
	  if ((cell_data_1(i+1)%neighbours(2) .eq. -1) .and.		&
	      (cell_data_1(i+1)%neighbours(1) .ne. -1) .and.		&
	      (cell_data_1(i+1)%neighbours(3) .ne. -1)) then
	    loc_counter = loc_counter+1
	    if (loc_counter .ne. i+1) then
	      help = c_1_to_c(loc_counter) 
	      c_1_to_c(loc_counter) = c_1_to_c(i+1)
	      c_1_to_c(i+1) = help
	      help = c_to_c_1(c_1_to_c(loc_counter)) 
	      c_to_c_1(c_1_to_c(loc_counter)) = c_to_c_1(c_1_to_c(i+1))
	      c_to_c_1(c_1_to_c(i+1)) = help
	      cell_help = cell_data_1(loc_counter) 
	      cell_data_1(loc_counter) = cell_data_1(i+1)
	      cell_data_1(i+1) = cell_help
	    end if
	  end if
	end do loop_wall_2
	wall_2 = loc_counter
!=======================================================================
!	find c_wall_12: the corner between a wall_1 and
!	a real wall_2 (not outflow!!)
!=======================================================================
	loop_c_wall_12: do i = wall_2,counter-1
	  if ((cell_data_1(i+1)%neighbours(1) .eq. -1) .and.		&
	      (cell_data_1(i+1)%neighbours(2) .eq. -1) .and.		&
	      (cell_data_1(i+1)%centre(1) .lt. x_max)) then
	    loc_counter = loc_counter+1
	    if (loc_counter .ne. i+1) then
	      help = c_1_to_c(loc_counter) 
	      c_1_to_c(loc_counter) = c_1_to_c(i+1)
	      c_1_to_c(i+1) = help
	      help = c_to_c_1(c_1_to_c(loc_counter)) 
	      c_to_c_1(c_1_to_c(loc_counter)) = c_to_c_1(c_1_to_c(i+1))
	      c_to_c_1(c_1_to_c(i+1)) = help
	      cell_help = cell_data_1(loc_counter) 
	      cell_data_1(loc_counter) = cell_data_1(i+1)
	      cell_data_1(i+1) = cell_help
	    end if
	  end if
	end do loop_c_wall_12
	c_wall_12 = loc_counter
!=======================================================================
!	now find all wall_1 cells (not outflow!!)
!=======================================================================
	loop_wall_1: do i = c_wall_12,counter-1
	  if ((cell_data_1(i+1)%neighbours(1) .eq. -1) .and.		&
	      (cell_data_1(i+1)%neighbours(2) .ne. -1) .and.		&
	      (cell_data_1(i+1)%neighbours(4) .ne. -1) .and.		&
	      (cell_data_1(i+1)%centre(1) .lt. x_max)) then
	    loc_counter = loc_counter+1
	    if (loc_counter .ne. i+1) then
	      help = c_1_to_c(loc_counter) 
	      c_1_to_c(loc_counter) = c_1_to_c(i+1)
	      c_1_to_c(i+1) = help
	      help = c_to_c_1(c_1_to_c(loc_counter)) 
	      c_to_c_1(c_1_to_c(loc_counter)) = c_to_c_1(c_1_to_c(i+1))
	      c_to_c_1(c_1_to_c(i+1)) = help
	      cell_help = cell_data_1(loc_counter) 
	      cell_data_1(loc_counter) = cell_data_1(i+1)
	      cell_data_1(i+1) = cell_help
	    end if
	  end if
	end do loop_wall_1
	wall_1 = loc_counter
!=======================================================================
!	find c_wall_41: the corner between a wall_1 and
!	a real wall_4 (not outflow!!)
!=======================================================================
	loop_c_wall_41: do i = wall_1,counter-1
	  if ((cell_data_1(i+1)%neighbours(1) .eq. -1) .and.		&
	      (cell_data_1(i+1)%neighbours(4) .eq. -1) .and.		&
	      (cell_data_1(i+1)%centre(1) .lt. x_max)) then
	    loc_counter = loc_counter+1
	    if (loc_counter .ne. i+1) then
	      help = c_1_to_c(loc_counter) 
	      c_1_to_c(loc_counter) = c_1_to_c(i+1)
	      c_1_to_c(i+1) = help
	      help = c_to_c_1(c_1_to_c(loc_counter)) 
	      c_to_c_1(c_1_to_c(loc_counter)) = c_to_c_1(c_1_to_c(i+1))
	      c_to_c_1(c_1_to_c(i+1)) = help
	      cell_help = cell_data_1(loc_counter) 
	      cell_data_1(loc_counter) = cell_data_1(i+1)
	      cell_data_1(i+1) = cell_help
	    end if
	  end if
	end do loop_c_wall_41
	c_wall_41 = loc_counter
!=======================================================================
!	now find all wall_4 cells
!=======================================================================
	loop_wall_4: do i = c_wall_41,counter-1
	  if ((cell_data_1(i+1)%neighbours(4) .eq. -1) .and.		&
	      (cell_data_1(i+1)%neighbours(1) .ne. -1) .and.		&
	      (cell_data_1(i+1)%neighbours(3) .ne. -1)) then
	    loc_counter = loc_counter+1
	    if (loc_counter .ne. i+1) then
	      help = c_1_to_c(loc_counter) 
	      c_1_to_c(loc_counter) = c_1_to_c(i+1)
	      c_1_to_c(i+1) = help
	      help = c_to_c_1(c_1_to_c(loc_counter)) 
	      c_to_c_1(c_1_to_c(loc_counter)) = c_to_c_1(c_1_to_c(i+1))
	      c_to_c_1(c_1_to_c(i+1)) = help
	      cell_help = cell_data_1(loc_counter) 
	      cell_data_1(loc_counter) = cell_data_1(i+1)
	      cell_data_1(i+1) = cell_help
	    end if
	  end if
	end do loop_wall_4
	wall_4 = loc_counter
!=======================================================================
!	find c_wall_34: the corner between a wall_4 and
!	a real wall_3 (not inflow!!)
!=======================================================================
	loop_c_wall_34: do i = wall_4,counter-1
	  if ((cell_data_1(i+1)%neighbours(3) .eq. -1) .and.		&
	      (cell_data_1(i+1)%neighbours(4) .eq. -1) .and.		&
	      (cell_data_1(i+1)%centre(1) .gt. x_min)) then
	    loc_counter = loc_counter+1
	    if (loc_counter .ne. i+1) then
	      help = c_1_to_c(loc_counter) 
	      c_1_to_c(loc_counter) = c_1_to_c(i+1)
	      c_1_to_c(i+1) = help
	      help = c_to_c_1(c_1_to_c(loc_counter)) 
	      c_to_c_1(c_1_to_c(loc_counter)) = c_to_c_1(c_1_to_c(i+1))
	      c_to_c_1(c_1_to_c(i+1)) = help
	      cell_help = cell_data_1(loc_counter) 
	      cell_data_1(loc_counter) = cell_data_1(i+1)
	      cell_data_1(i+1) = cell_help
	    end if
	  end if
	end do loop_c_wall_34
	c_wall_34 = loc_counter
!=======================================================================
!	now find all wall_3 cells (not inflow!!)
!=======================================================================
	loop_wall_3: do i = c_wall_34,counter-1
	  if ((cell_data_1(i+1)%neighbours(3) .eq. -1) .and.		&
	      (cell_data_1(i+1)%neighbours(2) .ne. -1) .and.		&
	      (cell_data_1(i+1)%neighbours(4) .ne. -1) .and.		&
	      (cell_data_1(i+1)%centre(1) .gt. x_min)) then
	    loc_counter = loc_counter+1
	    if (loc_counter .ne. i+1) then
	      help = c_1_to_c(loc_counter) 
	      c_1_to_c(loc_counter) = c_1_to_c(i+1)
	      c_1_to_c(i+1) = help
	      help = c_to_c_1(c_1_to_c(loc_counter)) 
	      c_to_c_1(c_1_to_c(loc_counter)) = c_to_c_1(c_1_to_c(i+1))
	      c_to_c_1(c_1_to_c(i+1)) = help
	      cell_help = cell_data_1(loc_counter) 
	      cell_data_1(loc_counter) = cell_data_1(i+1)
	      cell_data_1(i+1) = cell_help
	    end if
	  end if
	end do loop_wall_3
	wall_3 = loc_counter
!=======================================================================
!	find c_wall_23: the corner between a wall_3 and
!	a real wall_2 (not inflow!!)
!=======================================================================
	loop_c_wall_23: do i = wall_3,counter-1
	  if ((cell_data_1(i+1)%neighbours(3) .eq. -1) .and.		&
	      (cell_data_1(i+1)%neighbours(2) .eq. -1) .and.		&
	      (cell_data_1(i+1)%centre(1) .gt. x_min)) then
	    loc_counter = loc_counter+1
	    if (loc_counter .ne. i+1) then
	      help = c_1_to_c(loc_counter) 
	      c_1_to_c(loc_counter) = c_1_to_c(i+1)
	      c_1_to_c(i+1) = help
	      help = c_to_c_1(c_1_to_c(loc_counter)) 
	      c_to_c_1(c_1_to_c(loc_counter)) = c_to_c_1(c_1_to_c(i+1))
	      c_to_c_1(c_1_to_c(i+1)) = help
	      cell_help = cell_data_1(loc_counter) 
	      cell_data_1(loc_counter) = cell_data_1(i+1)
	      cell_data_1(i+1) = cell_help
	    end if
	  end if
	end do loop_c_wall_23
	c_wall_23 = loc_counter
!=======================================================================
!	now renumber all the inner nodes by first finding out
!	who is boundary cell and then renumbering the remaining ones.
!=======================================================================
	do i = 1,dim
	  boundary = 0
	  do j = 1,4
	    if (cell_data(i)%neighbours(j) .lt. 0 ) boundary = 1
	  end do
	  if (boundary .eq. 0) then
	    counter = counter+1
	    cell_data_1(counter) = cell_data(i)
	    c_to_c_1(i) 	= counter
	    c_1_to_c(counter) 	= i
	  end if
	end do

!=======================================================================
!	find all the new neighbour relations
!	c_to_c_1	gives the number in cell_data_1
!			for a cell in cell_data
!	c_1_to_c	the other way round
!	both together determine exactly relations between
!	old and new numbering.
!=======================================================================
	do i = 1,dim
	  if (cell_data(c_1_to_c(i))%neighbours(1) .ne. -1)		&
	  cell_data_1(i)%neighbours(1) = 				&
			c_to_c_1(cell_data(c_1_to_c(i))%neighbours(1))
	  if (cell_data(c_1_to_c(i))%neighbours(2) .ne. -1)		&
	  cell_data_1(i)%neighbours(2) = 				&
			c_to_c_1(cell_data(c_1_to_c(i))%neighbours(2))
	  if (cell_data(c_1_to_c(i))%neighbours(3) .ne. -1)		&
	  cell_data_1(i)%neighbours(3) = 				&
			c_to_c_1(cell_data(c_1_to_c(i))%neighbours(3))
	  if (cell_data(c_1_to_c(i))%neighbours(4) .ne. -1)		&
	  cell_data_1(i)%neighbours(4) = 				&
			c_to_c_1(cell_data(c_1_to_c(i))%neighbours(4))
	end do
!=======================================================================
! copy cell_data_1 to cell_data
!=======================================================================
	cell_data = cell_data_1
!=======================================================================
!  now throw away cell_data_1
!=======================================================================
	deallocate(cell_data_1,c_to_c_1,c_1_to_c,stat=info)
!=======================================================================
!  find the nodes at the boundary that are used
!  as additional points!
!=======================================================================
	counter = 1
	do i = 2,c_inflow_2
	  counter = counter+1
	end do
	counter = counter+1
	i = c_inflow_2
	i = cell_data(i)%neighbours(1)
	counter = counter+1
	bc_in_1: do while(i .ne. 1) 
          if ((cell_data(i)%neighbours(1) .eq. -1) .and. 		&
	      (cell_data(i)%neighbours(4) .ne. -1)) then
	    i = cell_data(i)%neighbours(4)
	    counter = counter+1
	    j = i
	  else if (cell_data(i)%neighbours(2) .eq. -1) then
	    i = cell_data(i)%neighbours(1)
	    counter = counter+1
	    j = i
	  else if (cell_data(i)%neighbours(3) .eq. -1) then
	    i = cell_data(i)%neighbours(2)
	    counter = counter+1
	    j = i
	  else if (cell_data(i)%neighbours(4) .eq. -1) then
	    i = cell_data(i)%neighbours(3)
	    counter = counter+1
	    j = i
	  else
	    if (cell_data(cell_data(j)%neighbours(1))%neighbours(2)	&
		.eq. -1) then
	      i = cell_data(i)%neighbours(1)
	    else if (cell_data(cell_data(i)%neighbours(2))%neighbours(3)&
	        .eq. -1) then
	      i =  cell_data(i)%neighbours(2)
	    else if (cell_data(cell_data(i)%neighbours(3))%neighbours(4)&
	        .eq. -1) then
              i =  cell_data(i)%neighbours(3)
	    else if (cell_data(cell_data(i)%neighbours(4))%neighbours(1)&
                .eq. -1) then
	      i =  cell_data(i)%neighbours(4)
	    else
	      write(*,*) '(II) there is a bug you should kill urgently!!!'
	    end if
	  end if
	end do bc_in_1
!=======================================================================
!  now write those boundaries to a file and keep the number
!  of boundary nodes in mind!
!=======================================================================
	
	open(99,file='boundary.in',status='unknown')
	write(99,*) c_wall_23
	counter = 0
	j = 1
	do i = 2,c_inflow_2
	  write(99,*) counter,-1,i-1,1,j-1
	  j = i
	  counter = counter+1
	end do
	i = c_inflow_2
	i = cell_data(i)%neighbours(1)
	write(99,*) counter,-1,i-1,1,j-1
	counter = counter+1
	j = i
	bc_in_2: do while(i .ne. 1) 
          if ((cell_data(i)%neighbours(1) .eq. -1) .and. 		&
	      (cell_data(i)%neighbours(4) .ne. -1)) then
	    i = cell_data(i)%neighbours(4)
	    write(99,*) counter,-1,i-1,1,j-1
	    counter = counter+1
	    j = i
	  else if (cell_data(i)%neighbours(2) .eq. -1) then
	    i = cell_data(i)%neighbours(1)
	    write(99,*) counter,-1,i-1,1,j-1
	    counter = counter+1
	    j = i
	  else if (cell_data(i)%neighbours(3) .eq. -1) then
	    i = cell_data(i)%neighbours(2)
	    write(99,*) counter,-1,i-1,1,j-1
	    counter = counter+1
	    j = i
	  else if (cell_data(i)%neighbours(4) .eq. -1) then
	    i = cell_data(i)%neighbours(3)
	    write(99,*) counter,-1,i-1,1,j-1
	    counter = counter+1
	    j = i
	  else
	    if (cell_data(cell_data(j)%neighbours(1))%neighbours(2)	&
		.eq. -1) then
	      i = cell_data(i)%neighbours(1)
	    else if (cell_data(cell_data(i)%neighbours(2))%neighbours(3)&
	        .eq. -1) then
	      i =  cell_data(i)%neighbours(2)
	    else if (cell_data(cell_data(i)%neighbours(3))%neighbours(4)&
	        .eq. -1) then
              i =  cell_data(i)%neighbours(3)
	    else if (cell_data(cell_data(i)%neighbours(4))%neighbours(1)&
                .eq. -1) then
	      i =  cell_data(i)%neighbours(4)
	    else
	      write(*,*) '(II) there is a bug you should kill urgently!!!'
	    end if
	  end if
	end do bc_in_2
	close (99)

!=======================================================================
!  write all points were values are given to a special
!  file. points are given both at the boundaries and at
!  the center of each cell. The number of points to be given is
!  thus
!  dim + number of boundary cells + 
!	 number of 1_2 corners    +
!        number of 2_3 corners    +
!        number of 3_4 corners    +
!	 number of 4_1 corners
!
!=======================================================================
	open(99,file='nodes.in',status='unknown')
        write(99,*) dim
!        write(99,*) dim + 				&
!		    c_wall_12 - wall_2 +		&
!		    c_wall_23 - wall_3 +		&
!		    c_wall_34 - wall_4 +		&
!		    c_wall_41 - wall_1
	do i = 1,dim
	  write(99,*) i-1,cell_data(i)%centre(1),cell_data(i)%centre(2)
	end do
	close(99)

	end subroutine visco_sort_data

!=======================================================================
        subroutine calc_cell_volume(i)
!=======================================================================
!   this subroiutine calculates the volume of a cell!
!   it only works for rectangular quadrilaterals!!!
!=======================================================================

        use visco_declaration
        implicit none

	integer      :: i
	real(double) :: delta_x, delta_y

	delta_x = sqrt((cell_data(i)%coordinates(1,2) -      &
                        cell_data(i)%coordinates(1,1))**2 +  &
                       (cell_data(i)%coordinates(2,2) -      &
			cell_data(i)%coordinates(2,1))**2)
	delta_y = sqrt((cell_data(i)%coordinates(1,3) -      &
                        cell_data(i)%coordinates(1,2))**2 +  &
                       (cell_data(i)%coordinates(2,3) -      &
			cell_data(i)%coordinates(2,2))**2)
        cell_data(i)%volume = delta_x * delta_y

	end subroutine calc_cell_volume

	end module visco_mesh
