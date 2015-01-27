
!	This program initialises everything for 
!	the calculation of a viscoelastic fluid flow
!	using a sort of pressure correction method
!
!	- information about block structure is read
!	  from a data file
!
!	- matrices are then initialised using allocate
!
!	Original author: Michael M. Resch (MR)
!	First released: 15. November 1994
!
!	First revision:
!			date:
!			author: (MR)
!			reason: additionally read output_flags
!				to determine which results to
!				write onto an output file
!=====================================================================
	module visco_init
	contains

!=====================================================================
	subroutine visco_read_data()
!=====================================================================
!       This subroutine reads all data necessary for
!       a viscoelastic calculation.
!=====================================================================

	use visco_declaration
	use visco_mesh
	
	implicit none
!=====================================================================
!	read mesh information and calculation parameters
!=====================================================================
        call visco_read_mesh()
        call visco_sort_data()
	call visco_allocate_vectors()
	call visco_allocate_matrices()

!====================================================================
!       added heikex 980206
!====================================================================  
        allocate (type_change(dim),STAT=info) 


        viscosity = 0.036
	write(*,*) 'lambda = ?'
	read(70,*) lambda
	write(*,*) lambda
	call visco_read_initial_unknowns()
	write(*,*) 'number_time_steps = ?'
	read(70,*) number_time_steps
	write(*,*) number_time_steps
	write(*,*) 'time_step_size =?'
	read(70,*) delta_t
	write(*,*) delta_t
	write(*,*) 'relaxation parameter'
	read(70,*) relax
	write(*,*) relax
	write(*,*) 'reference velocity'
	read(70,*) unull
	write(*,*) unull
        write(*,*) 'give me some output-flags 1=> print 0=> print not'
        write(*,*) 'pressure, u, v, sigma, tau, gamma'
        read(70,*) output_flags
	write(*,*) output_flags

	end subroutine visco_read_data

!=====================================================================
	subroutine visco_read_initial_unknowns()
!=====================================================================
!	This subroutine reads initial unknown values
!	velocity/stress and pressure
!=====================================================================

	use auxiliary
	use visco_declaration
	use visco_mesh
	
	implicit none

	integer		:: i,init
	character*80	:: filestring, dummy

	write(*,*) 'Is there initial values available? (0/1)'
	read(70,*) init
	write(*,*) init
	init_value: if (init==0) then
	  read(70,*) dummy		!! reads a dummy line
	  do i = 1,dim
!	    q((i-1)*5+1) = parabel(cell_data(i)%centre(2))
!	    q((i-1)*5+2) = 0.0d0
!	    q((i-1)*5+3) = 2*parabel_y(cell_data(i)%centre(2))**2 *  	&
!			   viscosity*lambda
!	    q((i-1)*5+4) = parabel_y(cell_data(i)%centre(2))*viscosity
!	    q((i-1)*5+5) = 0.0d0

	    q((i-1)*5+1) = 0.001d0
	    q((i-1)*5+2) = 0.0d0
	    q((i-1)*5+3) = 0.0d0
	    q((i-1)*5+4) = 0.0d0
	    q((i-1)*5+5) = 0.0d0

	    cell_data(i)%unknowns%velocities(1) = q((i-1)*5+1)
	    cell_data(i)%unknowns%velocities(2) = q((i-1)*5+2)
	    cell_data(i)%unknowns%stresses(1)   = q((i-1)*5+3)
	    cell_data(i)%unknowns%stresses(2)   = q((i-1)*5+4)
	    cell_data(i)%unknowns%stresses(3)   = q((i-1)*5+5)

            cell_data(i)%pressure = 2.0d0*parabel(0.0d0)*			&
	    			    (x_max - cell_data(i)%centre(1))
	  end do
	else
	  write(*,*) 'File with initial values?'
	  read(70,*) filestring
	  open(50,file=filestring)
!          do i = 1,4
!            read(50,*) dummy		!! reads 4 character strings
!	  end do
!	  read(50,*) i			!! reads the number of types of unknowns
! if (i <> 6) stop 'Not all values p,u,v,sigma,tau,gamma given'
!          do i = 1,7
!	    read(50,*) dummy		!! reads max/min values for each unknown
!	  end do
          read(50,*) i			!! reads number of cells
!	  if (i <> dim) stop 'num_vals and num_cells do not correspond'
	  do i = 1,dim
	    read(50,*) init,p(i),q((i-1)*5+1),q((i-1)*5+2),q((i-1)*5+3),	&
	    	       q((i-1)*5+4),q((i-1)*5+5)
	    cell_data(i)%unknowns%velocities(1) = q((i-1)*5+1)
	    cell_data(i)%unknowns%velocities(2) = q((i-1)*5+2)
	    cell_data(i)%unknowns%stresses(1)   = q((i-1)*5+3)
	    cell_data(i)%unknowns%stresses(2)   = q((i-1)*5+4)
	    cell_data(i)%unknowns%stresses(3)   = q((i-1)*5+5)

	    cell_data(i)%pressure = p(i)
	  end do
	  close(50)
	end if init_value
        q2 = q
        p2 = p

	end subroutine visco_read_initial_unknowns

!=====================================================================
	subroutine visco_allocate_matrices()
!=====================================================================
!       This subroutine initializes the matrices.
!=====================================================================
	use visco_declaration

	implicit none

!=====================================================================
!	allocate memory for the matrices needed
!=====================================================================
! Uwe Laun: A,C,col und row sind reine Hilfsfelder, welche die Nicht-Null-
!           Glieder der matrix aufnehmen und deshalb nicht quadratisch mit
!           der Matrixdimension wachsen, sondern linear.
	allocate(A(10*dim),STAT=info)
	allocate(col(10*dim),STAT=info)
	allocate(row(10*dim),STAT=info)
	allocate(C(10*dim),STAT=info)
!	allocate(AC(dim,dim),STAT=info)

        C = 0.0d0
        A = 0.0d0
        col = 0
        row = 0

	end subroutine visco_allocate_matrices

!=====================================================================
	subroutine visco_init_A()
!=====================================================================
!       This subroutine initializes matrix A.
!	THis matrix will remain constant during calculation.
!	It's the divergence matrix for velocity.
!=====================================================================
! Uwe Laun: A ist nun eine eindimensionales Feld in welches nacheinander alle
!           explizit gesetzten Feldelemante der Matrix eingetragen werden. 
!           row und col halyen dabei die entsprechenden Zeilen und Spalten.

	use visco_declaration
        use arithmetic

	implicit none

	integer		:: i,k
        integer         :: l
        real(double),dimension(:),allocatable:: ele3

!=====================================================================
!	initialize A correctly
!=====================================================================
        A = 0.0d0
        col=0
        row=0

!=====================================================================
!	inflow lower corner
!=====================================================================

	i = 1
        l = 1

        k = (cell_data(i)%neighbours(1)-1)*5+1
        A(l)            = cell_data(i)%n_vectors(1,1)/2.0d0
        col(l)=k
        row(l)=i
        l=l+1
        k = k + 1
        A(l)            = cell_data(i)%n_vectors(2,1)/2.0d0
        col(l)=k
        row(l)=i
        l=l+1

        k = (cell_data(i)%neighbours(2)-1)*5+1
        A(l)            = cell_data(i)%n_vectors(1,2)/2.0d0
        col(l)=k
        row(l)=i
        l=l+1
        k = k + 1
        A(l)            = cell_data(i)%n_vectors(2,2)/2.0d0
        col(l)=k
        row(l)=i
        l=l+1

        k = (i-1)*5+1
        A(l) =  (cell_data(i)%n_vectors(1,1) +		&
		   cell_data(i)%n_vectors(1,2))/2.0d0
        col(l)=k
        row(l)=i
        l=l+1
	k = k + 1
        A(l) =  (cell_data(i)%n_vectors(2,1) +		&
		   cell_data(i)%n_vectors(2,2))/2.0d0
        col(l)=k
        row(l)=i
        l=l+1

!=====================================================================
!	this is a loop over inflow boundaries
!=====================================================================
	do i = 2,inflow
	
          k = (cell_data(i)%neighbours(1)-1)*5+1
          A(l)            = cell_data(i)%n_vectors(1,1)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1
          k = k + 1
          A(l)            = cell_data(i)%n_vectors(2,1)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1

          k = (cell_data(i)%neighbours(2)-1)*5+1
          A(l)            = cell_data(i)%n_vectors(1,2)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1
	  k = k + 1
          A(l)            = cell_data(i)%n_vectors(2,2)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1

          k = (i-1)*5+1
          A(l) =  (cell_data(i)%n_vectors(1,1) +		&
		     cell_data(i)%n_vectors(1,2) +		&
		     cell_data(i)%n_vectors(1,4))/2.0d0
          col(l)=k
          row(l)=i
          l=l+1
          k = k + 1
          A(l) =  (cell_data(i)%n_vectors(2,1) +		&
		     cell_data(i)%n_vectors(2,2) +		&
		     cell_data(i)%n_vectors(2,4))/2.0d0
          col(l)=k
          row(l)=i
          l=l+1

          k = (cell_data(i)%neighbours(4)-1)*5+1
          A(l)            = cell_data(i)%n_vectors(1,4)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1
          k = k + 1
          A(l)            = cell_data(i)%n_vectors(2,4)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1

	end do

!=====================================================================
!	inflow upper corner
!=====================================================================

	i = c_inflow_2

        k = (cell_data(i)%neighbours(1)-1)*5+1
        A(l)            = cell_data(i)%n_vectors(1,1)/2.0d0
        col(l)=k
        row(l)=i
        l=l+1
        k = k + 1
        A(l)            = cell_data(i)%n_vectors(2,1)/2.0d0
        col(l)=k
        row(l)=i
        l=l+1

        k = (i-1)*5+1
        A(l) 		= (cell_data(i)%n_vectors(1,1) +	&
		  	   cell_data(i)%n_vectors(1,4))/2.0d0
        col(l)=k
        row(l)=i
        l=l+1
        k = k + 1
        A(l) 		= (cell_data(i)%n_vectors(2,1) +	&
		  	   cell_data(i)%n_vectors(2,4))/2.0d0
        col(l)=k
        row(l)=i
        l=l+1

        k = (cell_data(i)%neighbours(4)-1)*5+1
        A(l)            = cell_data(i)%n_vectors(1,4)/2.0d0
        col(l)=k
        row(l)=i
        l=l+1
        k = k + 1
        A(l)            = cell_data(i)%n_vectors(2,4)/2.0d0
        col(l)=k
        row(l)=i
        l=l+1

!=====================================================================
!	outflow upper corner
!=====================================================================

	i = c_outflow_2

        k = (i-1)*5+1
        A(l) 		=  cell_data(i)%n_vectors(1,1) +		&
			  (cell_data(i)%n_vectors(1,3) +	&
			   cell_data(i)%n_vectors(1,4))/2.0d0
        col(l)=k
        row(l)=i
        l=l+1
        k = k + 1
        A(l) 		=  cell_data(i)%n_vectors(2,1) +		&
			  (cell_data(i)%n_vectors(2,3) +	&
			   cell_data(i)%n_vectors(2,4))/2.0d0
        col(l)=k
        row(l)=i
        l=l+1

        k = (cell_data(i)%neighbours(3)-1)*5+1
        A(l)            = cell_data(i)%n_vectors(1,3)/2.0d0
        col(l)=k
        row(l)=i
        l=l+1
        k = k + 1
        A(l)            = cell_data(i)%n_vectors(2,3)/2.0d0
        col(l)=k
        row(l)=i
        l=l+1

        k = (cell_data(i)%neighbours(4)-1)*5+1
        A(l)            = cell_data(i)%n_vectors(1,4)/2.0d0
        col(l)=k
        row(l)=i
        l=l+1
        k = k + 1
        A(l)            = cell_data(i)%n_vectors(2,4)/2.0d0
        col(l)=k
        row(l)=i
        l=l+1

!=====================================================================
!	this is a loop over all outflow volumes
!=====================================================================

	do i = c_outflow_2+1,outflow

          k = (i-1)*5+1
          A(l)  	= (cell_data(i)%n_vectors(1,2) +	&
			   cell_data(i)%n_vectors(1,3) +	&
			   cell_data(i)%n_vectors(1,4))/2.0d0 + &
			   cell_data(i)%n_vectors(1,1)
          col(l)=k
          row(l)=i
          l=l+1
          k = k + 1
          A(l)  	= (cell_data(i)%n_vectors(2,2) +	&
			   cell_data(i)%n_vectors(2,3) +	&
			   cell_data(i)%n_vectors(2,4))/2.0d0 + &
			   cell_data(i)%n_vectors(2,1)
          col(l)=k
          row(l)=i
          l=l+1

          k = (cell_data(i)%neighbours(2)-1)*5+1
          A(l)             = cell_data(i)%n_vectors(1,2)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1
          k = k + 1
          A(l)            = cell_data(i)%n_vectors(2,2)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1

          k = (cell_data(i)%neighbours(3)-1)*5+1
          A(l)            = cell_data(i)%n_vectors(1,3)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1
          k = k + 1
          A(l)            = cell_data(i)%n_vectors(2,3)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1

          k = (cell_data(i)%neighbours(4)-1)*5+1
          A(l)            = cell_data(i)%n_vectors(1,4)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1
          k = k + 1
          A(l)            = cell_data(i)%n_vectors(2,4)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1

	end do

!=====================================================================
!	this is the lower corner outflow
!=====================================================================

	i = c_outflow_4

        k = (i-1)*5+1
        A(l) 		= (cell_data(i)%n_vectors(1,2) +	&
			   cell_data(i)%n_vectors(1,3))/2.0d0 +	&
			   cell_data(i)%n_vectors(1,1) 
        col(l)=k
        row(l)=i
        l=l+1
        k = k + 1
        A(l) 		= (cell_data(i)%n_vectors(2,2) +	&
			   cell_data(i)%n_vectors(2,3))/2.0d0 +	&
			   cell_data(i)%n_vectors(2,1)
        col(l)=k
        row(l)=i
        l=l+1

        k = (cell_data(i)%neighbours(2)-1)*5+1
        A(l)            = cell_data(i)%n_vectors(1,2)/2.0d0
        col(l)=k
        row(l)=i
        l=l+1
        k = k + 1
        A(l)            = cell_data(i)%n_vectors(2,2)/2.0d0
        col(l)=k
        row(l)=i
        l=l+1

        k = (cell_data(i)%neighbours(3)-1)*5+1
        A(l)            = cell_data(i)%n_vectors(1,3)/2.0d0
        col(l)=k
        row(l)=i
        l=l+1
        k = k + 1
        A(l)            = cell_data(i)%n_vectors(2,3)/2.0d0
        col(l)=k
        row(l)=i
        l=l+1

!=====================================================================
!	this is a loop over all wall_2 volumes
!=====================================================================

	A_WALL_2: do i = c_outflow_4+1,wall_2

          k = (cell_data(i)%neighbours(1)-1)*5+1
          A(l)            = cell_data(i)%n_vectors(1,1)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1
          k = k + 1
          A(l)            = cell_data(i)%n_vectors(2,1)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1

          k = (i-1)*5+1
          A(l)  	= (cell_data(i)%n_vectors(1,1) +	&
			   cell_data(i)%n_vectors(1,3) +	&
			   cell_data(i)%n_vectors(1,4))/2.0d0 
          col(l)=k
          row(l)=i
          l=l+1
          k = k + 1
          A(l)  	= (cell_data(i)%n_vectors(2,1) +	&
			   cell_data(i)%n_vectors(2,3) +	&
			   cell_data(i)%n_vectors(2,4))/2.0d0 
          col(l)=k
          row(l)=i
          l=l+1

          k = (cell_data(i)%neighbours(3)-1)*5+1
          A(l)            = cell_data(i)%n_vectors(1,3)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1
          k = k + 1
          A(l)            = cell_data(i)%n_vectors(2,3)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1

          k = (cell_data(i)%neighbours(4)-1)*5+1
          A(l)            = cell_data(i)%n_vectors(1,4)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1
          k = k + 1
          A(l)            = cell_data(i)%n_vectors(2,4)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1

	end do A_WALL_2

!=====================================================================
!	all the c_wall_12 cells (corner cells)
!=====================================================================

	do i = wall_2+1, c_wall_12

          k = (i-1)*5+1
          A(l)  	= (cell_data(i)%n_vectors(1,3) +	&
			   cell_data(i)%n_vectors(1,4))/2.0d0 
          col(l)=k
          row(l)=i
          l=l+1
          k = k + 1
          A(l)  	= (cell_data(i)%n_vectors(2,3) +	&
			   cell_data(i)%n_vectors(2,4))/2.0d0 
          col(l)=k
          row(l)=i
          l=l+1

          k = (cell_data(i)%neighbours(3)-1)*5+1
          A(l)            = cell_data(i)%n_vectors(1,3)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1
          k = k + 1
          A(l)            = cell_data(i)%n_vectors(2,3)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1

          k = (cell_data(i)%neighbours(4)-1)*5+1
          A(l)            = cell_data(i)%n_vectors(1,4)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1
          k = k + 1
          A(l)            = cell_data(i)%n_vectors(2,4)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1

	end do

!=====================================================================
!	this is a loop over all wall_1 volumes
!=====================================================================
	do i = c_wall_12+1,wall_1

	  k = (i-1)*5+1
	  A(l)  	    = (cell_data(i)%n_vectors(1,2) +	&
			       cell_data(i)%n_vectors(1,3) +	&
			       cell_data(i)%n_vectors(1,4))/2.0d0
          col(l)=k
          row(l)=i
          l=l+1
	  k = k + 1
	  A(l)  	    = (cell_data(i)%n_vectors(2,2) +	&
			       cell_data(i)%n_vectors(2,3) +	&
			       cell_data(i)%n_vectors(2,4))/2.0d0
          col(l)=k
          row(l)=i
          l=l+1

          k = (cell_data(i)%neighbours(2)-1)*5+1
          A(l)            = cell_data(i)%n_vectors(1,2)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1
          k = k + 1
          A(l)            = cell_data(i)%n_vectors(2,2)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1

          k = (cell_data(i)%neighbours(3)-1)*5+1
          A(l)            = cell_data(i)%n_vectors(1,3)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1
          k = k + 1
          A(l)            = cell_data(i)%n_vectors(2,3)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1

          k = (cell_data(i)%neighbours(4)-1)*5+1
          A(l)            = cell_data(i)%n_vectors(1,4)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1
          k = k + 1
          A(l)            = cell_data(i)%n_vectors(2,4)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1
	end do
!=====================================================================
!	this is a loop over all c_wall_41 corners
!=====================================================================

	do i = wall_1+1,c_wall_41

	  k = (i-1)*5+1
	  A(l)  	    = (cell_data(i)%n_vectors(1,2) +	&
			       cell_data(i)%n_vectors(1,3))/2.0d0
          col(l)=k
          row(l)=i
          l=l+1
	  k = k + 1
	  A(l)  	    = (cell_data(i)%n_vectors(2,2) +	&
			       cell_data(i)%n_vectors(2,3))/2.0d0
          col(l)=k
          row(l)=i
          l=l+1

          k = (cell_data(i)%neighbours(2)-1)*5+1
          A(l)            = cell_data(i)%n_vectors(1,2)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1
          k = k + 1
          A(l)            = cell_data(i)%n_vectors(2,2)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1

          k = (cell_data(i)%neighbours(3)-1)*5+1
          A(l)            = cell_data(i)%n_vectors(1,3)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1
          k = k + 1
          A(l)            = cell_data(i)%n_vectors(2,3)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1

	end do

!=====================================================================
!	this is a loop over all wall_4 volumes
!=====================================================================
	do i = c_wall_41+1,wall_4

	  k = (i-1)*5+1
	  A(l)  	    = (cell_data(i)%n_vectors(1,1) +	&
			       cell_data(i)%n_vectors(1,2) +	&
			       cell_data(i)%n_vectors(1,3))/2.0d0
          col(l)=k
          row(l)=i
          l=l+1
	  k = k + 1
	  A(l)  	    = (cell_data(i)%n_vectors(2,1) +	&
			       cell_data(i)%n_vectors(2,2) +	&
			       cell_data(i)%n_vectors(2,3))/2.0d0
          col(l)=k
          row(l)=i
          l=l+1

          k = (cell_data(i)%neighbours(1)-1)*5+1
          A(l)            = cell_data(i)%n_vectors(1,1)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1
          k = k + 1
          A(l)            = cell_data(i)%n_vectors(2,1)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1

          k = (cell_data(i)%neighbours(2)-1)*5+1
          A(l)            = cell_data(i)%n_vectors(1,2)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1
          k = k + 1
          A(l)            = cell_data(i)%n_vectors(2,2)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1

          k = (cell_data(i)%neighbours(3)-1)*5+1
          A(l)            = cell_data(i)%n_vectors(1,3)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1
          k = k + 1
          A(l)            = cell_data(i)%n_vectors(2,3)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1

	end do

!=====================================================================
!	this is a loop over all c_wall_34 corners
!=====================================================================

	do i = wall_4+1, c_wall_34

	  k = (i-1)*5+1
	  A(l)  	    = (cell_data(i)%n_vectors(1,1) +	&
			       cell_data(i)%n_vectors(1,2))/2.0d0
          col(l)=k
          row(l)=i
          l=l+1
	  k = k + 1
	  A(l)  	    = (cell_data(i)%n_vectors(2,1) +	&
			       cell_data(i)%n_vectors(2,2))/2.0d0
          col(l)=k
          row(l)=i
          l=l+1

          k = (cell_data(i)%neighbours(1)-1)*5+1
          A(l)            = cell_data(i)%n_vectors(1,1)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1
          k = k + 1
          A(l)            = cell_data(i)%n_vectors(2,1)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1

          k = (cell_data(i)%neighbours(2)-1)*5+1
          A(l)            = cell_data(i)%n_vectors(1,2)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1
          k = k + 1
          A(l)            = cell_data(i)%n_vectors(2,2)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1

	end do

!=====================================================================
!	this is a loop over all wall_3 cells
!=====================================================================

	do i = c_wall_34+1,wall_3

	  k = (i-1)*5+1
	  A(l)  	    = (cell_data(i)%n_vectors(1,1) +	&
			       cell_data(i)%n_vectors(1,2) +	&
			       cell_data(i)%n_vectors(1,4))/2.0d0
          col(l)=k
          row(l)=i
          l=l+1
	  k = k + 1
	  A(l)  	    = (cell_data(i)%n_vectors(2,1) +	&
			       cell_data(i)%n_vectors(2,2) +	&
			       cell_data(i)%n_vectors(2,4))/2.0d0
          col(l)=k
          row(l)=i
          l=l+1

          k = (cell_data(i)%neighbours(1)-1)*5+1
          A(l)            = cell_data(i)%n_vectors(1,1)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1
          k = k + 1
          A(l)            = cell_data(i)%n_vectors(2,1)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1

          k = (cell_data(i)%neighbours(2)-1)*5+1
          A(l)            = cell_data(i)%n_vectors(1,2)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1
          k = k + 1
          A(l)            = cell_data(i)%n_vectors(2,2)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1

          k = (cell_data(i)%neighbours(4)-1)*5+1
          A(l)            = cell_data(i)%n_vectors(1,4)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1
          k = k + 1
          A(l)            = cell_data(i)%n_vectors(2,4)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1
	end do

!=====================================================================
!	this is a loop over all c_wall_23 corner cells
!=====================================================================

	do i = wall_3+1,c_wall_23

	  k = (i-1)*5+1
	  A(l)  	    = (cell_data(i)%n_vectors(1,1) +	&
			       cell_data(i)%n_vectors(1,4))/2.0d0
          col(l)=k
          row(l)=i
          l=l+1
	  k = k + 1
	  A(l)  	    = (cell_data(i)%n_vectors(2,1) +	&
			       cell_data(i)%n_vectors(2,4))/2.0d0
          col(l)=k
          row(l)=i
          l=l+1

          k = (cell_data(i)%neighbours(1)-1)*5+1
          A(l)            = cell_data(i)%n_vectors(1,1)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1
          k = k + 1
          A(l)            = cell_data(i)%n_vectors(2,1)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1

          k = (cell_data(i)%neighbours(4)-1)*5+1
          A(l)            = cell_data(i)%n_vectors(1,4)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1
          k = k + 1
          A(l)            = cell_data(i)%n_vectors(2,4)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1
	end do
!=====================================================================
!	this is a loop over all inner volumes used.
!=====================================================================
	do i = c_wall_23+1,dim

	  k = (i-1)*5+1
	  A(l)	            = (cell_data(i)%n_vectors(1,1) +	&
			       cell_data(i)%n_vectors(1,2) +	&
			       cell_data(i)%n_vectors(1,3) +	&
			       cell_data(i)%n_vectors(1,4))/2.0d0
          col(l)=k
          row(l)=i
          l=l+1
	  k = k + 1
	  A(l)   	    = (cell_data(i)%n_vectors(2,1) +	&
			       cell_data(i)%n_vectors(2,2) +	&
			       cell_data(i)%n_vectors(2,3) +	&
			       cell_data(i)%n_vectors(2,4))/2.0d0
          col(l)=k
          row(l)=i
          l=l+1

          k = (cell_data(i)%neighbours(1)-1)*5+1
          A(l)            = cell_data(i)%n_vectors(1,1)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1
          k = k + 1
          A(l)            = cell_data(i)%n_vectors(2,1)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1

          k = (cell_data(i)%neighbours(2)-1)*5+1
          A(l)            = cell_data(i)%n_vectors(1,2)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1
          k = k + 1
          A(l)            = cell_data(i)%n_vectors(2,2)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1

          k = (cell_data(i)%neighbours(3)-1)*5+1
          A(l)            = cell_data(i)%n_vectors(1,3)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1
          k = k + 1
          A(l)            = cell_data(i)%n_vectors(2,3)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1

          k = (cell_data(i)%neighbours(4)-1)*5+1
          A(l)            = cell_data(i)%n_vectors(1,4)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1
          k = k + 1
          A(l)            = cell_data(i)%n_vectors(2,4)/2.0d0
          col(l)=k
          row(l)=i
          l=l+1

	end do

! Uwe Laun: Die zu Null gesetzten Elemente werden nun entfernt und die
!           in C entstehenden Luecken zugeschoben.



        l=l-1
        k=1
        do i=1,l
         if (A(i).ne.0) then
          A(k)=A(i)
          col(k)=col(i)
          row(k)=row(i)
          k=k+1
         endif
       enddo
       write(*,*) 'Matrix A'
       do i = 1,k-1
         write(*,*) A(i),row(i),col(i)
       end do
        do i=k,l
         A(i)=0.0
         col(i)=0
         row(i)=0
        enddo

        l=k-1
        i=10

! Uwe Laun: Diese Routine ordnet die Feldelemente in das jagged-diagonal-Format,
!           da ansonsten die Struktur vom atithmetic-Modul nicht angenommen wurde.
!           (coredump in matrix_concentrate)

        
        call order_jagged(A,row,col,l,dim,dim,i)

! Uwe Laun: Die Feldlaenge wird in matrix_def mit size bestimmt; also
!          besser nur ein Feld der richtigen Laenge uebergeben!

        allocate(ele3(l))
        do i=1,l
         ele3(i)=A(i)
        enddo
        deallocate(A)

! Uwe Laun: Endlich die Struktur der Matrix in das sparse-Format des
!           arithmetic-Moduls wandeln.


        call matrix_def(A2,"A2",ele3,row,col,dim,dim*5)

        deallocate(ele3)

	end subroutine visco_init_A

!=====================================================================
	subroutine visco_init_C()
!=====================================================================
!       This subroutine initializes matrix C.
!	This matrix will remain constant during calculation.
!	It's the gradient matrix for the pressure.
!=====================================================================
! Uwe Laun: C ist nun eine eindimensionales Feld in welches nacheinander alle
!           explizit gesetzten Feldelemante der Matrix eingetragen werden.
!           row und col halten dabei die entsprechenden Zeilen und Spalten.


	use visco_declaration

	implicit none

	integer		:: i,k,l,m
        real(double),allocatable,dimension(:) :: ele3

!=====================================================================
!	initialize C correctly
!=====================================================================
	C = 0.0
        col=0
        row=0
         

!=====================================================================
!	inflow lower corner
!	at the inflow boundary pressure is set to be unchanged
!	from inflow to cell-center. The same is assumed for
!	fixed wall.
!=====================================================================
	i = 1
        m = 1

	k = (i-1)*5+1
	l = cell_data(i)%neighbours(1)
	C(m)   	= cell_data(i)%n_vectors(1,1)/2.0d0
        row(m)=k; col(m)=l; m=m+1
	k = k+1
	C(m)    = cell_data(i)%n_vectors(2,1)/2.0d0
        row(m)=k; col(m)=l; m=m+1

	k = (i-1)*5+1
	l = cell_data(i)%neighbours(2)
	C(m)   	= cell_data(i)%n_vectors(1,2)/2.0d0
        row(m)=k; col(m)=l; m=m+1
	k = k+1
	C(m)    = cell_data(i)%n_vectors(2,2)/2.0d0
        row(m)=k; col(m)=l; m=m+1

	k = (i-1)*5+1
	C(m)    =          					&
	          cell_data(i)%n_vectors(1,1)/2.0d0 +		&
	          cell_data(i)%n_vectors(1,2)/2.0d0 +		&
	          cell_data(i)%n_vectors(1,3)       +		&
	          cell_data(i)%n_vectors(1,4)
                  row(m)=k; col(m)=i; m=m+1
	k = k+1
	C(m)    =          					&
	          cell_data(i)%n_vectors(2,1)/2.0d0 +		&
	          cell_data(i)%n_vectors(2,2)/2.0d0 +		&
	          cell_data(i)%n_vectors(2,3)       +		&
	          cell_data(i)%n_vectors(2,4)
                  row(m)=k; col(m)=i; m=m+1

!=====================================================================
!	inflow boundary
!=====================================================================
	do i = 2,inflow

	  k = (i-1)*5+1
	  l = cell_data(i)%neighbours(1)
	  C(m)   	= cell_data(i)%n_vectors(1,1)/2.0d0
          row(m)=k; col(m)=l; m=m+1
	  k = k+1
	  C(m)      = cell_data(i)%n_vectors(2,1)/2.0d0
          row(m)=k; col(m)=l; m=m+1

	  k = (i-1)*5+1
	  l = cell_data(i)%neighbours(2)
	  C(m)   	= cell_data(i)%n_vectors(1,2)/2.0d0
          row(m)=k; col(m)=l; m=m+1
	  k = k+1
	  C(m)      = cell_data(i)%n_vectors(2,2)/2.0d0
          row(m)=k; col(m)=l; m=m+1

	  k = (i-1)*5+1
	  l = cell_data(i)%neighbours(4)
	  C(m)   	= cell_data(i)%n_vectors(1,4)/2.0d0
          row(m)=k; col(m)=l; m=m+1
	  k = k+1
	  C(m)      = cell_data(i)%n_vectors(2,4)/2.0d0
          row(m)=k; col(m)=l; m=m+1

	  k = (i-1)*5+1
	  C(m)    =          					&
	            cell_data(i)%n_vectors(1,1)/2.0d0 +		&
	            cell_data(i)%n_vectors(1,2)/2.0d0 +		&
	            cell_data(i)%n_vectors(1,3)       +		&
	            cell_data(i)%n_vectors(1,4)/2.0d0
                    row(m)=k; col(m)=i; m=m+1
	  k = k+1
	  C(m)    =          					&
	            cell_data(i)%n_vectors(2,1)/2.0d0 +		&
	            cell_data(i)%n_vectors(2,2)/2.0d0 +		&
	            cell_data(i)%n_vectors(2,3)       +		&
	            cell_data(i)%n_vectors(2,4)/2.0d0
                    row(m)=k; col(m)=i; m=m+1

	end do

!=====================================================================
!	inflow upper corner
!=====================================================================

	i = c_inflow_2

        k = (i-1)*5+1
        l = cell_data(i)%neighbours(1)
        C(m)      = cell_data(i)%n_vectors(1,1)/2.0d0
        row(m)=k; col(m)=l; m=m+1
        k = k+1
        C(m)      = cell_data(i)%n_vectors(2,1)/2.0d0
        row(m)=k; col(m)=l; m=m+1

        k = (i-1)*5+1
        l = cell_data(i)%neighbours(4)
        C(m)      = cell_data(i)%n_vectors(1,4)/2.0d0
        row(m)=k; col(m)=l; m=m+1
        k = k+1
        C(m)      = cell_data(i)%n_vectors(2,4)/2.0d0
        row(m)=k; col(m)=l; m=m+1

	k = (i-1)*5+1
	C(m)    =          					&
	          cell_data(i)%n_vectors(1,1)/2.0d0 +		&
	          cell_data(i)%n_vectors(1,2)       +		&
	          cell_data(i)%n_vectors(1,3)       +		&
	          cell_data(i)%n_vectors(1,4)/2.0d0
                  row(m)=k; col(m)=i; m=m+1
	k = k+1
	C(m)    =          					&
	          cell_data(i)%n_vectors(2,1)/2.0d0 +		&
	          cell_data(i)%n_vectors(2,2)       +		&
	          cell_data(i)%n_vectors(2,3)       +		&
	          cell_data(i)%n_vectors(2,4)/2.0d0
                  row(m)=k; col(m)=i; m=m+1


!=====================================================================
!	outflow upper corner
!=====================================================================

	i = c_outflow_2 

        k = (i-1)*5+1
        l = cell_data(i)%neighbours(3)
        C(m)      = cell_data(i)%n_vectors(1,3)/2.0d0
        row(m)=k; col(m)=l; m=m+1
        k = k+1
        C(m)      = cell_data(i)%n_vectors(2,3)/2.0d0
        row(m)=k; col(m)=l; m=m+1

        k = (i-1)*5+1
        l = cell_data(i)%neighbours(4)
        C(m)      = cell_data(i)%n_vectors(1,4)/2.0d0
        row(m)=k; col(m)=l; m=m+1
        k = k+1
        C(m)      = cell_data(i)%n_vectors(2,4)/2.0d0
        row(m)=k; col(m)=l; m=m+1

        k = (i-1)*5+1
	C(m)    =                                               &
                  cell_data(i)%n_vectors(1,1)/2.0d0+           &
		  cell_data(i)%n_vectors(1,2)       +           &
                  cell_data(i)%n_vectors(1,3)/2.0d0 +		&
		  cell_data(i)%n_vectors(1,4)/2.0d0
        row(m)=k; col(m)=i; m=m+1
	k = k+1
	C(m)    =            					&
                  cell_data(i)%n_vectors(2,1)/2.0d0+           &
                  cell_data(i)%n_vectors(2,2)       +           &
		  cell_data(i)%n_vectors(2,3)/2.0d0 +		&
		  cell_data(i)%n_vectors(2,4)/2.0d0
        row(m)=k; col(m)=i; m=m+1

!=====================================================================
!	outflow volumes
!=====================================================================

	do i = c_outflow_2+1,outflow

          k = (i-1)*5+1
          l = cell_data(i)%neighbours(2)
          C(m)      = cell_data(i)%n_vectors(1,2)/2.0d0
          row(m)=k; col(m)=l; m=m+1
          k = k+1
          C(m)      = cell_data(i)%n_vectors(2,2)/2.0d0
          row(m)=k; col(m)=l; m=m+1

          k = (i-1)*5+1
          l = cell_data(i)%neighbours(3)
          C(m)      = cell_data(i)%n_vectors(1,3)/2.0d0
          row(m)=k; col(m)=l; m=m+1
          k = k+1
          C(m)      = cell_data(i)%n_vectors(2,3)/2.0d0
          row(m)=k; col(m)=l; m=m+1

          k = (i-1)*5+1
          l = cell_data(i)%neighbours(4)
          C(m)      = cell_data(i)%n_vectors(1,4)/2.0d0
          row(m)=k; col(m)=l; m=m+1
          k = k+1
          C(m)      = cell_data(i)%n_vectors(2,4)/2.0d0
          row(m)=k; col(m)=l; m=m+1

	  k = (i-1)*5+1
	  C(m)    =                                             &
                  cell_data(i)%n_vectors(1,1)/2.0d0+           &
		  cell_data(i)%n_vectors(1,2)/2.0d0 +           &
                  cell_data(i)%n_vectors(1,3)/2.0d0 +           &
		  cell_data(i)%n_vectors(1,4)/2.0d0
                  row(m)=k; col(m)=i; m=m+1
	  k = k+1
	  C(m)    =            					&
                  cell_data(i)%n_vectors(2,1)/2.0d0+           &
                  cell_data(i)%n_vectors(2,2)/2.0d0 +           &
		  cell_data(i)%n_vectors(2,3)/2.0d0 +     	&
		  cell_data(i)%n_vectors(2,4)/2.0d0
                  row(m)=k; col(m)=i; m=m+1
	end do

!=====================================================================
!	outflow lower corner
!=====================================================================

	i = c_outflow_4

        k = (i-1)*5+1
        l = cell_data(i)%neighbours(2)
        C(m)      = cell_data(i)%n_vectors(1,2)/2.0d0
        row(m)=k; col(m)=l; m=m+1
        k = k+1
        C(m)      = cell_data(i)%n_vectors(2,2)/2.0d0
        row(m)=k; col(m)=l; m=m+1

        k = (i-1)*5+1
        l = cell_data(i)%neighbours(3)
        C(m)      = cell_data(i)%n_vectors(1,3)/2.0d0
        row(m)=k; col(m)=l; m=m+1
        k = k+1
        C(m)      = cell_data(i)%n_vectors(2,3)/2.0d0
        row(m)=k; col(m)=l; m=m+1

	k = (i-1)*5+1
	C(m)    =                                             &
                  cell_data(i)%n_vectors(1,1)/2.0d0 +           &
		  cell_data(i)%n_vectors(1,2)/2.0d0 +           &
                  cell_data(i)%n_vectors(1,3)/2.0d0 +           &
		  cell_data(i)%n_vectors(1,4)
                  row(m)=k; col(m)=i; m=m+1
	k = k+1
	C(m)    =            					&
                  cell_data(i)%n_vectors(2,1)/2.0d0 +           &
                  cell_data(i)%n_vectors(2,2)/2.0d0 +           &
		  cell_data(i)%n_vectors(2,3)/2.0d0 +     	&
		  cell_data(i)%n_vectors(2,4)
                  row(m)=k; col(m)=i; m=m+1

!=====================================================================
!	wall_2 volumes
!=====================================================================
	do i = c_outflow_4+1,wall_2

          k = (i-1)*5+1
          l = cell_data(i)%neighbours(1)
          C(m)      = cell_data(i)%n_vectors(1,1)/2.0d0
          row(m)=k; col(m)=l; m=m+1
          k = k+1
          C(m)      = cell_data(i)%n_vectors(2,1)/2.0d0
          row(m)=k; col(m)=l; m=m+1

          k = (i-1)*5+1
          l = cell_data(i)%neighbours(3)
          C(m)      = cell_data(i)%n_vectors(1,3)/2.0d0
          row(m)=k; col(m)=l; m=m+1
          k = k+1
          C(m)      = cell_data(i)%n_vectors(2,3)/2.0d0
          row(m)=k; col(m)=l; m=m+1

          k = (i-1)*5+1
          l = cell_data(i)%neighbours(4)
          C(m)      = cell_data(i)%n_vectors(1,4)/2.0d0
          row(m)=k; col(m)=l; m=m+1
          k = k+1
          C(m)      = cell_data(i)%n_vectors(2,4)/2.0d0
          row(m)=k; col(m)=l; m=m+1

	  k = (i-1)*5+1
	  C(m)    =                                             &
		  cell_data(i)%n_vectors(1,1)/2.0d0 +   	&
		  cell_data(i)%n_vectors(1,2)       +           &
                  cell_data(i)%n_vectors(1,3)/2.0d0 +           &
		  cell_data(i)%n_vectors(1,4)/2.0d0
                  row(m)=k; col(m)=i; m=m+1
	  k = k+1
	  C(m)    =            					&
		  cell_data(i)%n_vectors(2,1)/2.0d0 +           &
                  cell_data(i)%n_vectors(2,2)       +           &
		  cell_data(i)%n_vectors(2,3)/2.0d0 +     	&
		  cell_data(i)%n_vectors(2,4)/2.0d0
                  row(m)=k; col(m)=i; m=m+1

	end do

!=====================================================================
!	all the c_wall_12 cells (corner cells)
!=====================================================================
	do i = wall_2+1, c_wall_12

          k = (i-1)*5+1
          l = cell_data(i)%neighbours(3)
          C(m)      = cell_data(i)%n_vectors(1,3)/2.0d0
          row(m)=k; col(m)=l; m=m+1
          k = k+1
          C(m)      = cell_data(i)%n_vectors(2,3)/2.0d0
          row(m)=k; col(m)=l; m=m+1

          k = (i-1)*5+1
          l = cell_data(i)%neighbours(4)
          C(m)      = cell_data(i)%n_vectors(1,4)/2.0d0
          row(m)=k; col(m)=l; m=m+1
          k = k+1
          C(m)      = cell_data(i)%n_vectors(2,4)/2.0d0
          row(m)=k; col(m)=l; m=m+1

	  k = (i-1)*5+1
	  C(m)    =                                             &
		  cell_data(i)%n_vectors(1,1)       +   	&
		  cell_data(i)%n_vectors(1,2)       +           &
                  cell_data(i)%n_vectors(1,3)/2.0d0 +           &
		  cell_data(i)%n_vectors(1,4)/2.0d0
                  row(m)=k; col(m)=i; m=m+1
	  k = k+1
	  C(m)    =            					&
		  cell_data(i)%n_vectors(2,1)       +           &
                  cell_data(i)%n_vectors(2,2)       +           &
		  cell_data(i)%n_vectors(2,3)/2.0d0 +     	&
		  cell_data(i)%n_vectors(2,4)/2.0d0
                  row(m)=k; col(m)=i; m=m+1

	end do

!=====================================================================
!	this is a loop over all wall_1 volumes
!=====================================================================
	do i = c_wall_12+1,wall_1

	  k = (i-1)*5+1
          l = cell_data(i)%neighbours(2)
          C(m)      = cell_data(i)%n_vectors(1,2)/2.0d0
          row(m)=k; col(m)=l; m=m+1
          k = k+1
          C(m)      = cell_data(i)%n_vectors(2,2)/2.0d0
          row(m)=k; col(m)=l; m=m+1

          k = (i-1)*5+1
          l = cell_data(i)%neighbours(3)
          C(m)      = cell_data(i)%n_vectors(1,3)/2.0d0
          row(m)=k; col(m)=l; m=m+1
          k = k+1
          C(m)      = cell_data(i)%n_vectors(2,3)/2.0d0
          row(m)=k; col(m)=l; m=m+1

          k = (i-1)*5+1
          l = cell_data(i)%neighbours(4)
          C(m)      = cell_data(i)%n_vectors(1,4)/2.0d0
          row(m)=k; col(m)=l; m=m+1
          k = k+1
          C(m)      = cell_data(i)%n_vectors(2,4)/2.0d0
          row(m)=k; col(m)=l; m=m+1

	  k = (i-1)*5+1
	  C(m)    =                                             &
		  cell_data(i)%n_vectors(1,1)       +   	&
		  cell_data(i)%n_vectors(1,2)/2.0d0 +           &
                  cell_data(i)%n_vectors(1,3)/2.0d0 +           &
		  cell_data(i)%n_vectors(1,4)/2.0d0
                  row(m)=k; col(m)=i; m=m+1
	  k = k+1
	  C(m)    =            					&
		  cell_data(i)%n_vectors(2,1)       +           &
                  cell_data(i)%n_vectors(2,2)/2.0d0 +           &
		  cell_data(i)%n_vectors(2,3)/2.0d0 +     	&
		  cell_data(i)%n_vectors(2,4)/2.0d0
                  row(m)=k; col(m)=i; m=m+1

	end do
!=====================================================================
!	this is a loop over all c_wall_41 corners
!=====================================================================

	do i = wall_1+1,c_wall_41

	  k = (i-1)*5+1
          l = cell_data(i)%neighbours(2)
          C(m)      = cell_data(i)%n_vectors(1,2)/2.0d0
          row(m)=k; col(m)=l; m=m+1
          k = k+1
          C(m)      = cell_data(i)%n_vectors(2,2)/2.0d0
          row(m)=k; col(m)=l; m=m+1

          k = (i-1)*5+1
          l = cell_data(i)%neighbours(3)
          C(m)      = cell_data(i)%n_vectors(1,3)/2.0d0
          row(m)=k; col(m)=l; m=m+1
          k = k+1
          C(m)      = cell_data(i)%n_vectors(2,3)/2.0d0
          row(m)=k; col(m)=l; m=m+1

	  k = (i-1)*5+1
	  C(m)    =                                             &
		  cell_data(i)%n_vectors(1,1)       +   	&
		  cell_data(i)%n_vectors(1,2)/2.0d0 +           &
                  cell_data(i)%n_vectors(1,3)/2.0d0 +           &
		  cell_data(i)%n_vectors(1,4)
                  row(m)=k; col(m)=i; m=m+1
	  k = k+1
	  C(m)    =            					&
		  cell_data(i)%n_vectors(2,1)       +           &
                  cell_data(i)%n_vectors(2,2)/2.0d0 +           &
		  cell_data(i)%n_vectors(2,3)/2.0d0 +     	&
		  cell_data(i)%n_vectors(2,4)
                  row(m)=k; col(m)=i; m=m+1

	end do
!=====================================================================
!	wall_4 volumes
!=====================================================================

	do i = c_wall_41+1,wall_4

	  k = (i-1)*5+1
          l = cell_data(i)%neighbours(1)
          C(m)      = cell_data(i)%n_vectors(1,1)/2.0d0
          row(m)=k; col(m)=l; m=m+1
          k = k+1
          C(m)      = cell_data(i)%n_vectors(2,1)/2.0d0
          row(m)=k; col(m)=l; m=m+1

	  k = (i-1)*5+1
          l = cell_data(i)%neighbours(2)
          C(m)      = cell_data(i)%n_vectors(1,2)/2.0d0
          row(m)=k; col(m)=l; m=m+1
          k = k+1
          C(m)      = cell_data(i)%n_vectors(2,2)/2.0d0
          row(m)=k; col(m)=l; m=m+1

          k = (i-1)*5+1
          l = cell_data(i)%neighbours(3)
          C(m)      = cell_data(i)%n_vectors(1,3)/2.0d0
          row(m)=k; col(m)=l; m=m+1
          k = k+1
          C(m)      = cell_data(i)%n_vectors(2,3)/2.0d0
          row(m)=k; col(m)=l; m=m+1

	  k = (i-1)*5+1
	  C(m)    =                                             &
		  cell_data(i)%n_vectors(1,1)/2.0d0 +   	&
		  cell_data(i)%n_vectors(1,2)/2.0d0 +           &
                  cell_data(i)%n_vectors(1,3)/2.0d0 +           &
		  cell_data(i)%n_vectors(1,4)
                  row(m)=k; col(m)=i; m=m+1
	  k = k+1
	  C(m)    =            					&
		  cell_data(i)%n_vectors(2,1)/2.0d0 +           &
                  cell_data(i)%n_vectors(2,2)/2.0d0 +           &
		  cell_data(i)%n_vectors(2,3)/2.0d0 +     	&
		  cell_data(i)%n_vectors(2,4)
                  row(m)=k; col(m)=i; m=m+1

	end do
!=====================================================================
!	this is a loop over all c_wall_34 corners
!=====================================================================

	do i = wall_4+1, c_wall_34

	  k = (i-1)*5+1
          l = cell_data(i)%neighbours(1)
          C(m)      = cell_data(i)%n_vectors(1,1)/2.0d0
          row(m)=k; col(m)=l; m=m+1
          k = k+1
          C(m)      = cell_data(i)%n_vectors(2,1)/2.0d0
          row(m)=k; col(m)=l; m=m+1

          k = (i-1)*5+1
          l = cell_data(i)%neighbours(2)
          C(m)      = cell_data(i)%n_vectors(1,2)/2.0d0
          row(m)=k; col(m)=l; m=m+1
          k = k+1
          C(m)      = cell_data(i)%n_vectors(2,2)/2.0d0
          row(m)=k; col(m)=l; m=m+1

	  k = (i-1)*5+1
	  C(m)    =                                             &
		  cell_data(i)%n_vectors(1,1)/2.0d0 +   	&
		  cell_data(i)%n_vectors(1,2)/2.0d0 +           &
                  cell_data(i)%n_vectors(1,3)       +           &
		  cell_data(i)%n_vectors(1,4)
                  row(m)=k; col(m)=i; m=m+1
	  k = k+1
	  C(m)    =            					&
		  cell_data(i)%n_vectors(2,1)/2.0d0 +           &
                  cell_data(i)%n_vectors(2,2)/2.0d0 +           &
		  cell_data(i)%n_vectors(2,3)       +     	&
		  cell_data(i)%n_vectors(2,4)
                  row(m)=k; col(m)=i; m=m+1

	end do
!=====================================================================
!	this is a loop over all wall_3 cells
!=====================================================================

	do i = c_wall_34+1,wall_3
	
          k = (i-1)*5+1
          l = cell_data(i)%neighbours(1)
          C(m)      = cell_data(i)%n_vectors(1,1)/2.0d0
          row(m)=k; col(m)=l; m=m+1
          k = k+1
          C(m)      = cell_data(i)%n_vectors(2,1)/2.0d0
          row(m)=k; col(m)=l; m=m+1

	  k = (i-1)*5+1
          l = cell_data(i)%neighbours(2)
          C(m)      = cell_data(i)%n_vectors(1,2)/2.0d0
          row(m)=k; col(m)=l; m=m+1
          k = k+1
          C(m)      = cell_data(i)%n_vectors(2,2)/2.0d0
          row(m)=k; col(m)=l; m=m+1

          k = (i-1)*5+1
          l = cell_data(i)%neighbours(4)
          C(m)      = cell_data(i)%n_vectors(1,4)/2.0d0
          row(m)=k; col(m)=l; m=m+1
          k = k+1
          C(m)      = cell_data(i)%n_vectors(2,4)/2.0d0
          row(m)=k; col(m)=l; m=m+1

	  k = (i-1)*5+1
	  C(m)    =                                             &
		  cell_data(i)%n_vectors(1,1)/2.0d0 +   	&
		  cell_data(i)%n_vectors(1,2)/2.0d0 +           &
                  cell_data(i)%n_vectors(1,3)       +           &
		  cell_data(i)%n_vectors(1,4)/2.0d0
                  row(m)=k; col(m)=i; m=m+1
	  k = k+1
	  C(m)    =            					&
		  cell_data(i)%n_vectors(2,1)/2.0d0 +           &
                  cell_data(i)%n_vectors(2,2)/2.0d0 +           &
		  cell_data(i)%n_vectors(2,3)       +     	&
		  cell_data(i)%n_vectors(2,4)/2.0d0
                  row(m)=k; col(m)=i; m=m+1

	end do
!=====================================================================
!	this is a loop over all c_wall_23 corner cells
!=====================================================================

	do i = wall_3+1,c_wall_23
	
	  k = (i-1)*5+1
          l = cell_data(i)%neighbours(1)
          C(m)      = cell_data(i)%n_vectors(1,1)/2.0d0
          row(m)=k; col(m)=l; m=m+1
          k = k+1
          C(m)      = cell_data(i)%n_vectors(2,1)/2.0d0
          row(m)=k; col(m)=l; m=m+1

          k = (i-1)*5+1
          l = cell_data(i)%neighbours(4)
          C(m)      = cell_data(i)%n_vectors(1,4)/2.0d0
          row(m)=k; col(m)=l; m=m+1
          k = k+1
          C(m)      = cell_data(i)%n_vectors(2,4)/2.0d0
          row(m)=k; col(m)=l; m=m+1

	  k = (i-1)*5+1
	  C(m)    =                                             &
		  cell_data(i)%n_vectors(1,1)/2.0d0 +   	&
		  cell_data(i)%n_vectors(1,2)       +           &
                  cell_data(i)%n_vectors(1,3)       +           &
		  cell_data(i)%n_vectors(1,4)/2.0d0
                  row(m)=k; col(m)=i; m=m+1
	  k = k+1
	  C(m)    =            					&
		  cell_data(i)%n_vectors(2,1)/2.0d0 +           &
                  cell_data(i)%n_vectors(2,2)       +           &
		  cell_data(i)%n_vectors(2,3)       +     	&
		  cell_data(i)%n_vectors(2,4)/2.0d0
                  row(m)=k; col(m)=i; m=m+1

	end do
!=====================================================================
!       this is a loop over all inner volumes used.
!=====================================================================
	do i = c_wall_23+1,dim

	  k = (i-1)*5+1
	  l = cell_data(i)%neighbours(1)
	  C(m)   	= cell_data(i)%n_vectors(1,1)/2.0d0
          row(m)=k; col(m)=l; m=m+1
	  k = k+1
	  C(m)      = cell_data(i)%n_vectors(2,1)/2.0d0
          row(m)=k; col(m)=l; m=m+1

	  k = (i-1)*5+1
	  l = cell_data(i)%neighbours(2)
	  C(m)   	= cell_data(i)%n_vectors(1,2)/2.0d0
          row(m)=k; col(m)=l; m=m+1
	  k = k+1
	  C(m)      = cell_data(i)%n_vectors(2,2)/2.0d0
          row(m)=k; col(m)=l; m=m+1

	  k = (i-1)*5+1
	  l = cell_data(i)%neighbours(3)
	  C(m)   	= cell_data(i)%n_vectors(1,3)/2.0d0
          row(m)=k; col(m)=l; m=m+1
	  k = k+1
	  C(m)      = cell_data(i)%n_vectors(2,3)/2.0d0
          row(m)=k; col(m)=l; m=m+1

	  k = (i-1)*5+1
	  l = cell_data(i)%neighbours(4)
	  C(m)   	= cell_data(i)%n_vectors(1,4)/2.0d0
          row(m)=k; col(m)=l; m=m+1
	  k = k+1
	  C(m)      = cell_data(i)%n_vectors(2,4)/2.0d0
          row(m)=k; col(m)=l; m=m+1

	end do

! Uwe Laun: Die zu Null gesetzten Elemente werden nun entfernt und die 
!           in C entstehenden Luecken zugeschoben.

        m=m-1
        k=1
        do i=1,m
         if (C(i).ne.0) then
          C(k)=C(i)
          col(k)=col(i)
          row(k)=row(i)
          k=k+1
         endif
        enddo
       write(*,*) 'Matrix C'
       do i = 1,k-1
         write(*,*) C(i),row(i),col(i)
       end do
        do i=k,m
         C(i)=0.0
         row(i)=0
         col(i)=0
        enddo
        m=k-1
        i=5

! Uwe Laun: Diese Routine ordnet die Feldelemente in das jagged-diagonal-Format,
!           da ansonsten die Struktur vom atithmetic-Modul nicht angenommen wurde.
!           (coredump in matrix_concentrate)
        
        call order_jagged(C,row,col,m,dim,dim*5,i)

! Uwe Laun: Die Feldlaenge wird in matrix_def mit size bestimmt; also 
!          besser nur ein Feld der richtigen Laenge uebergeben!

        allocate(ele3(m))
        do i=1,m
         ele3(i)=C(i)
        enddo
        deallocate(C)

! Uwe Laun: Endlich die Struktur der Matrix in das sparse-Format des
!           arithmetic-Moduls wandeln.

        call matrix_def(C2,"C2",ele3,row,col,dim*5,dim)

! Uwe Laun: Hilfsfelder allokieren

        deallocate(row)
        deallocate(col)
        deallocate(ele3)

	end subroutine visco_init_C

!=====================================================================
	subroutine visco_allocate_vectors()
!=====================================================================
!	this subroutine initializes vectors of unknowns
!=====================================================================

	use visco_declaration
        use arithmetic

	implicit none

!=====================================================================
!	allocate memory for the vectors needed
!=====================================================================
! Uwe Laun: Hier werden die Vektorfelder allokiert und dem vector-Typ
!           des arithmetic-Moduls zugaenglich gemacht.

        allocate(q(5*dim),STAT=info)
        q = 0.0d0
        call vector_def2(q2,"q2",q)
        allocate(q_old(5*dim),STAT=info)
        q_old = 0.0d0
        call vector_def2(q_old2,"q_old2",q_old)
        allocate(Bq(5*dim),STAT=info)
        Bq = 0.0d0
        call vector_def2(Bq2,"Bq2",Bq)
        allocate(r(5*dim),STAT=info)
        r = 0.0d0
        call vector_def2(r2,"r2",r)
        allocate(S(5*dim),STAT=info)
        S = 0.0d0
        call vector_def2(S2,"S2",S)
        allocate(T(dim),STAT=info)
        T = 0.0d0
        call vector_def2(T2,"T2",T)
        allocate(U(dim),STAT=info)
        U = 0.0d0
        call vector_def2(U2,"U2",U)
        allocate(p(dim),STAT=info)
        p = 0.0d0
        call vector_def2(p2,"p2",p)
        allocate(pivot(dim),STAT=info)
        pivot = 0

!	p	= 0.0
!	q 	= 0.0
!	Bq 	= 0.0
!	r 	= 0.0
!	S	= 0.0
!	T	= 0.0
!	U	= 0.0
!	pivot 	= 0
	
	end subroutine visco_allocate_vectors

	end module visco_init
