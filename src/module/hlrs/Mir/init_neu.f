      module visco_init_1
      contains
      
!=====================================================================
      subroutine visco_init_A_1()
!=====================================================================
!       This subroutine initializes matrix A.
!	This matrix will remain constant during calculation.
!	It's the divergence matrix for velocity.
!=====================================================================
! Uwe Laun: A ist nun eine eindimensionales Feld in welches nacheinander alle
!           explizit gesetzten Feldelemente der Matrix eingetragen werden. 
!           row und col halten dabei die entsprechenden Zeilen und Spalten.
!=====================================================================

	use visco_declaration
        use arithmetic

	implicit none

	integer		:: i,j,k,num_sides
        integer         :: l
        real(double),dimension(:),allocatable  :: ele3

!=====================================================================
!	initialize A correctly
!=====================================================================
        if (.not.allocated(A)) allocate(A(10*dim),STAT=info)
	A = 0.0
        col=0
        row=0
!=====================================================================
!	inflow lower corner
!=====================================================================
	i = 1
        l = 1
!=========================================================================
!	get number of sides
!=========================================================================
        num_sides = cell_data(i)%num_sides
        call visco_get_alpha(i)
!=========================================================================
!	Calculate the neighbouring entries from the factors.
!=========================================================================
        do j = 1,2
          k = (cell_data(i)%neighbours(j)-1)*5+1
          A(l)      = cell_data(i)%n_vectors(1,j)*(1-cell_data(i)%alphas(j))/2.0d0
          A(l) = A(l)/cell_data(i)%volume
          col(l)=k
          row(l)=i
          l=l+1
          k = k + 1
          A(l)      = cell_data(i)%n_vectors(2,j)*(1-cell_data(i)%alphas(j))/2.0d0
          A(l) = A(l)/cell_data(i)%volume
          col(l)=k
          row(l)=i
          l=l+1
        end do
!=========================================================================
!	Sum up over all the sides for calculating the values
!       for the center of the cell.
!=========================================================================
        k = (i-1)*5+1
        do j = 1,2
          A(l) =  A(l) + cell_data(i)%n_vectors(1,j)*(1+cell_data(i)%alphas(j))/2.0d0
        end do
        A(l) = A(l)/cell_data(i)%volume
        col(l)=k
        row(l)=i
        l=l+1
	k = k + 1
        do j = 1,2
          A(l) =  A(l) + cell_data(i)%n_vectors(2,j)*(1+cell_data(i)%alphas(j))/2.0d0
        end do
        A(l) = A(l)/cell_data(i)%volume
        col(l)=k
        row(l)=i
        l=l+1
!=====================================================================
!	this is a loop over inflow boundaries
!=====================================================================
	do i = 2,inflow
!=========================================================================
!	get number of sides
!=========================================================================
          num_sides = cell_data(i)%num_sides
          call visco_get_alpha(i)
!=========================================================================
!	Calculate the neighbouring entries from the factors.
!       Only neighbours 1,2,4 may contribute.
!=========================================================================
          do j = 1,2
            k = (cell_data(i)%neighbours(j)-1)*5+1
            A(l) = cell_data(i)%n_vectors(1,j)*(1-cell_data(i)%alphas(j))/2.0
            A(l) = A(l)/cell_data(i)%volume
            col(l)=k
            row(l)=i
            l = l+1
            k = k+1
            A(l) = cell_data(i)%n_vectors(2,j)*(1-cell_data(i)%alphas(j))/2.0
            A(l) = A(l)/cell_data(i)%volume
            col(l)=k
            row(l)=i
            l = l+1
          end do
          j = 4
          k = (cell_data(i)%neighbours(j)-1)*5+1
          A(l) = cell_data(i)%n_vectors(1,j)*(1-cell_data(i)%alphas(j))/2.0
          A(l) = A(l)/cell_data(i)%volume
          col(l)=k
          row(l)=i
          l = l+1
          k = k+1
          A(l) = cell_data(i)%n_vectors(2,j)*(1-cell_data(i)%alphas(j))/2.0
          A(l) = A(l)/cell_data(i)%volume
          col(l)=k
          row(l)=i
          l = l+1
!=========================================================================
!	Sum up over all the sides for calculating the values
!       for the center of the cell.
!=========================================================================
	  k = (i-1)*5+1
          do j = 1,2
            A(l) = A(l) + (cell_data(i)%alphas(j)+1)/2.0*cell_data(i)%n_vectors(1,j)
          end do
          j = 4
          A(l) = A(l) + (cell_data(i)%alphas(j)+1)/2.0*cell_data(i)%n_vectors(1,j)
          A(l) = A(l)/cell_data(i)%volume
          col(l)=k
          row(l)=i
          l=l+1
          k = k + 1
          do j = 1,2
            A(l) = A(l) + (cell_data(i)%alphas(j)+1)/2.0*cell_data(i)%n_vectors(2,j)
          end do
          j = 4
          A(l) = A(l) + (cell_data(i)%alphas(j)+1)/2.0*cell_data(i)%n_vectors(2,j)
          A(l) = A(l)/cell_data(i)%volume
          col(l)=k
          row(l)=i
          l=l+1
	end do
!=====================================================================
!	inflow upper corner
!=====================================================================
	i = c_inflow_2
!=========================================================================
!	get number of sides
!=========================================================================
        num_sides = cell_data(i)%num_sides
        call visco_get_alpha(i)
!=========================================================================
!	Calculate the neighbouring entries from the factors.
!=========================================================================
        j = 1
        k = (cell_data(i)%neighbours(j)-1)*5+1
        A(l)            = cell_data(i)%n_vectors(1,j)*(1-cell_data(i)%alphas(j))/2.0d0
        A(l) = A(l)/cell_data(i)%volume
        col(l)=k
        row(l)=i
        l=l+1
        k = k + 1
        A(l)            = cell_data(i)%n_vectors(2,j)*(1-cell_data(i)%alphas(j))/2.0d0
        A(l) = A(l)/cell_data(i)%volume
        col(l)=k
        row(l)=i
        l=l+1
        j = 4
        k = (cell_data(i)%neighbours(j)-1)*5+1
        A(l)            = cell_data(i)%n_vectors(1,j)*(1-cell_data(i)%alphas(j))/2.0d0
        A(l) = A(l)/cell_data(i)%volume
        col(l)=k
        row(l)=i
        l=l+1
        k = k + 1
        A(l)            = cell_data(i)%n_vectors(2,j)*(1-cell_data(i)%alphas(j))/2.0d0
        A(l) = A(l)/cell_data(i)%volume
        col(l)=k
        row(l)=i
        l=l+1
        k = (i-1)*5+1
        j = 1
        A(l) =  A(l) + cell_data(i)%n_vectors(1,j)*(1+cell_data(i)%alphas(j))/2.0d0
        j = 4
        A(l) =  A(l) + cell_data(i)%n_vectors(1,j)*(1+cell_data(i)%alphas(j))/2.0d0
        A(l) = A(l)/cell_data(i)%volume
        col(l)=k
        row(l)=i
        l=l+1
	k = k + 1
        j = 1
        A(l) =  A(l) + cell_data(i)%n_vectors(2,j)*(1+cell_data(i)%alphas(j))/2.0d0
        j = 4
        A(l) =  A(l) + cell_data(i)%n_vectors(2,j)*(1+cell_data(i)%alphas(j))/2.0d0        
        A(l) = A(l)/cell_data(i)%volume
        col(l)=k
        row(l)=i
        l=l+1
!=====================================================================
!	outflow upper corner
!=====================================================================
	i = c_outflow_2
!=========================================================================
!	get number of sides
!=========================================================================
        num_sides = cell_data(i)%num_sides
        call visco_get_alpha(i)
!=====================================================================
!	to ensure that velocity at the ouflow boundary
!       is equal to that at cell center (unchanged over the outflow)
!       alpha(1) is set to one.
!=====================================================================
        cell_data(i)%alphas(1) = 1
!=========================================================================
!	Calculate the neighbouring entries from the factors.
!=========================================================================
        do j = 3,4
          k = (cell_data(i)%neighbours(j)-1)*5+1
          A(l) = cell_data(i)%n_vectors(1,j)*(1-cell_data(i)%alphas(j))/2.0
          A(l) = A(l)/cell_data(i)%volume
          col(l)=k
          row(l)=i
          l = l+1
          k = k+1
          A(l) = cell_data(i)%n_vectors(2,j)*(1-cell_data(i)%alphas(j))/2.0
          A(l) = A(l)/cell_data(i)%volume
          col(l)=k
          row(l)=i
          l = l+1
	end do
!=========================================================================
!	Sum up over all the sides for calculating the values
!       for the center of the cell.
!=========================================================================
        k = (i-1)*5+1
        j = 1
        A(l) = A(l) + (cell_data(i)%alphas(j)+1)/2.0*cell_data(i)%n_vectors(1,j)
        do j = 3,4
          A(l) = A(l) + (cell_data(i)%alphas(j)+1)/2.0*cell_data(i)%n_vectors(1,j)
        end do
        A(l) = A(l)/cell_data(i)%volume
        col(l)=k
        row(l)=i
        l=l+1
        k = k + 1
        j = 1
        A(l) = A(l) + (cell_data(i)%alphas(j)+1)/2.0*cell_data(i)%n_vectors(2,j)
        do j = 3,4
          A(l) = A(l) + (cell_data(i)%alphas(j)+1)/2.0*cell_data(i)%n_vectors(2,j)
        end do
        A(l) = A(l)/cell_data(i)%volume
        col(l)=k
        row(l)=i
        l=l+1
!=====================================================================
!	this is a loop over all outflow volumes
!=====================================================================
	do i = c_outflow_2+1,outflow
!=========================================================================
!	get number of sides
!=========================================================================
          num_sides = cell_data(i)%num_sides
          call visco_get_alpha(i)
!=====================================================================
!	to ensure that velocity at the ouflow boundary
!       is equal to that at cell center (unchanged over the outflow)
!       alpha(1) is set to one.
!=====================================================================
          cell_data(i)%alphas(1) = 1
!=========================================================================
!	Calculate the neighbouring entries from the factors.
!=========================================================================
          do j = 2,4
            k = (cell_data(i)%neighbours(j)-1)*5+1
            A(l) = cell_data(i)%n_vectors(1,j)*(1-cell_data(i)%alphas(j))/2.0
            A(l) = A(l)/cell_data(i)%volume
            col(l)=k
            row(l)=i
            l = l+1
            k = k+1
            A(l) = cell_data(i)%n_vectors(2,j)*(1-cell_data(i)%alphas(j))/2.0
            A(l) = A(l)/cell_data(i)%volume
            col(l)=k
            row(l)=i
            l = l+1
	  end do
!=========================================================================
!	Sum up over all the sides for calculating the values
!       for the center of the cell.
!=========================================================================
          k = (i-1)*5+1
          do j = 1,num_sides
            A(l) = A(l) + (cell_data(i)%alphas(j)+1)/2.0*cell_data(i)%n_vectors(1,j)
          end do
          A(l) = A(l)/cell_data(i)%volume
          col(l)=k
          row(l)=i
          l=l+1
          k = k + 1
          do j = 1,num_sides
            A(l) = A(l) + (cell_data(i)%alphas(j)+1)/2.0*cell_data(i)%n_vectors(2,j)
          end do
          A(l) = A(l)/cell_data(i)%volume
          col(l)=k
          row(l)=i
          l=l+1
	end do
!=====================================================================
!	this is the lower corner outflow
!=====================================================================
	i = c_outflow_4
!=========================================================================
!	get number of sides
!=========================================================================
        num_sides = cell_data(i)%num_sides
        call visco_get_alpha(i)
!=====================================================================
!	to ensure that velocity at the ouflow boundary
!       is equal to that at cell center (unchanged over the outflow)
!       alpha(1) is set to one.
!=====================================================================
        cell_data(i)%alphas(1) = 1
!=========================================================================
!	Calculate the neighbouring entries from the factors.
!=========================================================================
        do j = 2,3
          k = (cell_data(i)%neighbours(j)-1)*5+1
          A(l) = cell_data(i)%n_vectors(1,j)*(1-cell_data(i)%alphas(j))/2.0
          A(l) = A(l)/cell_data(i)%volume
          col(l)=k
          row(l)=i
          l = l+1
          k = k+1
          A(l) = cell_data(i)%n_vectors(2,j)*(1-cell_data(i)%alphas(j))/2.0
          A(l) = A(l)/cell_data(i)%volume
          col(l)=k
          row(l)=i
          l = l+1
	end do
!=========================================================================
!	Sum up over all the sides for calculating the values
!       for the center of the cell.
!=========================================================================
        k = (i-1)*5+1
        j = 1
        A(l) = A(l) + (cell_data(i)%alphas(j)+1)/2.0*cell_data(i)%n_vectors(1,j)
        do j = 2,3
          A(l) = A(l) + (cell_data(i)%alphas(j)+1)/2.0*cell_data(i)%n_vectors(1,j)
        end do
        A(l) = A(l)/cell_data(i)%volume
        col(l)=k
        row(l)=i
        l=l+1
        k = k + 1
        j = 1
        A(l) = A(l) + (cell_data(i)%alphas(j)+1)/2.0*cell_data(i)%n_vectors(2,j)
        do j = 2,3
          A(l) = A(l) + (cell_data(i)%alphas(j)+1)/2.0*cell_data(i)%n_vectors(2,j)
        end do
        A(l) = A(l)/cell_data(i)%volume
        col(l)=k
        row(l)=i
        l=l+1
!=====================================================================
!	this is a loop over all wall_2 volumes
!=====================================================================
	A_WALL_2: do i = c_outflow_4+1,wall_2
!=========================================================================
!	get number of sides
!=========================================================================
          num_sides = cell_data(i)%num_sides
          call visco_get_alpha(i)
!=========================================================================
!	Calculate the neighbouring entries from the factors.
!=========================================================================
          j = 1
          k = (cell_data(i)%neighbours(j)-1)*5+1
          A(l) = cell_data(i)%n_vectors(1,j)*(1-cell_data(i)%alphas(j))/2.0
          A(l) = A(l)/cell_data(i)%volume
          col(l)=k
          row(l)=i
          l = l+1
          k = k+1
          A(l) = cell_data(i)%n_vectors(2,j)*(1-cell_data(i)%alphas(j))/2.0
          A(l) = A(l)/cell_data(i)%volume
          col(l)=k
          row(l)=i
          l = l+1
          do j = 3,num_sides
            k = (cell_data(i)%neighbours(j)-1)*5+1
            A(l) = cell_data(i)%n_vectors(1,j)*(1-cell_data(i)%alphas(j))/2.0
            A(l) = A(l)/cell_data(i)%volume
            col(l)=k
            row(l)=i
            l = l+1
            k = k+1
            A(l) = cell_data(i)%n_vectors(2,j)*(1-cell_data(i)%alphas(j))/2.0
            A(l) = A(l)/cell_data(i)%volume
            col(l)=k
            row(l)=i
            l = l+1
	  end do
!=========================================================================
!	Sum up over all the sides for calculating the values
!       for the center of the cell. wall 2 does not contributue
!=========================================================================
          k = (i-1)*5+1
          j = 1
          A(l) = A(l) + (cell_data(i)%alphas(j)+1)/2.0*cell_data(i)%n_vectors(1,j)
          do j = 3,num_sides
            A(l) = A(l) + (cell_data(i)%alphas(j)+1)/2.0*cell_data(i)%n_vectors(1,j)
          end do
          A(l) = A(l)/cell_data(i)%volume
          col(l)=k
          row(l)=i
          l=l+1
          k = k + 1
          j = 1
          A(l) = A(l) + (cell_data(i)%alphas(j)+1)/2.0*cell_data(i)%n_vectors(2,j)
          do j = 3,num_sides
            A(l) = A(l) + (cell_data(i)%alphas(j)+1)/2.0*cell_data(i)%n_vectors(2,j)
          end do
          A(l) = A(l)/cell_data(i)%volume
          col(l)=k
          row(l)=i
          l=l+1
	end do A_WALL_2
!=====================================================================
!	all the c_wall_12 cells (corner cells)
!=====================================================================
	do i = wall_2+1, c_wall_12
          num_sides = cell_data(i)%num_sides
          call visco_get_alpha(i)
!=========================================================================
!	Sum up over all the sides for calculating the values
!       for the center of the cell.
!=========================================================================
	  k = (i-1)*5+1
          do j = 3,num_sides
            A(l) = A(l) + (cell_data(i)%alphas(j)+1)/2.0*cell_data(i)%n_vectors(1,j)
          end do
          A(l) = A(l)/cell_data(i)%volume
          col(l)=k
          row(l)=i
          l=l+1
          k = k + 1
          do j = 3,num_sides
            A(l) = A(l) + (cell_data(i)%alphas(j)+1)/2.0*cell_data(i)%n_vectors(2,j)
          end do
          A(l) = A(l)/cell_data(i)%volume
          col(l)=k
          row(l)=i
          l=l+1
!=========================================================================
!	Calculate the neighbouring entries from the factors.
!=========================================================================
          do j = 3,num_sides
            k = (cell_data(i)%neighbours(j)-1)*5+1
            A(l) = cell_data(i)%n_vectors(1,j)*(1-cell_data(i)%alphas(j))/2.0
            A(l) = A(l)/cell_data(i)%volume
            col(l)=k
            row(l)=i
            l = l+1
            k = k+1
            A(l) = cell_data(i)%n_vectors(2,j)*(1-cell_data(i)%alphas(j))/2.0
            A(l) = A(l)/cell_data(i)%volume
            col(l)=k
            row(l)=i
            l = l+1
	  end do
	end do
!=====================================================================
!	this is a loop over all wall_1 volumes
!=====================================================================
	do i = c_wall_12+1,wall_1
          num_sides = cell_data(i)%num_sides
          call visco_get_alpha(i)
!=========================================================================
!	Sum up over all the sides for calculating the values
!       for the center of the cell.
!=========================================================================
	  k = (i-1)*5+1
          do j = 2,num_sides
            A(l) = A(l) + (cell_data(i)%alphas(j)+1)/2.0*cell_data(i)%n_vectors(1,j)
          end do
          A(l) = A(l)/cell_data(i)%volume
          col(l)=k
          row(l)=i
          l=l+1
          k = k + 1
          do j = 1,num_sides
            A(l) = A(l) + (cell_data(i)%alphas(j)+1)/2.0*cell_data(i)%n_vectors(2,j)
          end do
          A(l) = A(l)/cell_data(i)%volume
          col(l)=k
          row(l)=i
          l=l+1
!=========================================================================
!	Calculate the neighbouring entries from the factors.
!=========================================================================
          do j = 2,num_sides
            k = (cell_data(i)%neighbours(j)-1)*5+1
            A(l) = cell_data(i)%n_vectors(1,j)*(1-cell_data(i)%alphas(j))/2.0
            A(l) = A(l)/cell_data(i)%volume
            col(l)=k
            row(l)=i
            l = l+1
            k = k+1
            A(l) = cell_data(i)%n_vectors(2,j)*(1-cell_data(i)%alphas(j))/2.0
            A(l) = A(l)/cell_data(i)%volume
            col(l)=k
            row(l)=i
            l = l+1
	  end do
	end do
!=====================================================================
!	this is a loop over all c_wall_41 corners
!=====================================================================
	do i = wall_1+1,c_wall_41
          num_sides = cell_data(i)%num_sides
          call visco_get_alpha(i)
!=========================================================================
!	Sum up over all the sides for calculating the values
!       for the center of the cell.
!=========================================================================
	  k = (i-1)*5+1
          do j = 2,3
            A(l) = A(l) + (cell_data(i)%alphas(j)+1)/2.0*cell_data(i)%n_vectors(1,j)
          end do
          A(l) = A(l)/cell_data(i)%volume
          col(l)=k
          row(l)=i
          l=l+1
          k = k + 1
          do j = 2,3
            A(l) = A(l) + (cell_data(i)%alphas(j)+1)/2.0*cell_data(i)%n_vectors(2,j)
          end do
          A(l) = A(l)/cell_data(i)%volume
          col(l)=k
          row(l)=i
          l=l+1
!=========================================================================
!	Calculate the neighbouring entries from the factors.
!=========================================================================
          do j = 2,3
            k = (cell_data(i)%neighbours(j)-1)*5+1
            A(l) = cell_data(i)%n_vectors(1,j)*(1-cell_data(i)%alphas(j))/2.0
            A(l) = A(l)/cell_data(i)%volume
            col(l)=k
            row(l)=i
            l = l+1
            k = k+1
            A(l) = cell_data(i)%n_vectors(2,j)*(1-cell_data(i)%alphas(j))/2.0
            A(l) = A(l)/cell_data(i)%volume
            col(l)=k
            row(l)=i
            l = l+1
	  end do
	end do
!=====================================================================
!	this is a loop over all wall_4 volumes
!=====================================================================
	do i = c_wall_41+1,wall_4
          num_sides = cell_data(i)%num_sides
          call visco_get_alpha(i)
!=========================================================================
!	Calculate the neighbouring entries from the factors.
!=========================================================================
          do j = 1,num_sides-1
            k = (cell_data(i)%neighbours(j)-1)*5+1
            A(l) = cell_data(i)%n_vectors(1,j)*(1-cell_data(i)%alphas(j))/2.0
            A(l) = A(l)/cell_data(i)%volume
            col(l)=k
            row(l)=i
            l = l+1
            k = k+1
            A(l) = cell_data(i)%n_vectors(2,j)*(1-cell_data(i)%alphas(j))/2.0
            A(l) = A(l)/cell_data(i)%volume
            col(l)=k
            row(l)=i
            l = l+1
	  end do
!=========================================================================
!	Sum up over all the sides for calculating the values
!       for the center of the cell.
!=========================================================================
	  k = (i-1)*5+1
          do j = 1,num_sides-1
            A(l) = A(l) + (cell_data(i)%alphas(j)+1)/2.0*cell_data(i)%n_vectors(1,j)
          end do
          A(l) = A(l)/cell_data(i)%volume
          col(l)=k
          row(l)=i
          l=l+1
          k = k + 1
          do j = 1,num_sides-1
            A(l) = A(l) + (cell_data(i)%alphas(j)+1)/2.0*cell_data(i)%n_vectors(2,j)
          end do
          A(l) = A(l)/cell_data(i)%volume
          col(l)=k
          row(l)=i
          l=l+1
	end do
!=====================================================================
!	this is a loop over all c_wall_34 corners
!=====================================================================
	do i = wall_4+1, c_wall_34
          num_sides = cell_data(i)%num_sides
          call visco_get_alpha(i)
!=========================================================================
!	Sum up over all the sides for calculating the values
!       for the center of the cell.
!=========================================================================
	  k = (i-1)*5+1
          do j = 1,num_sides-2
            A(l) = A(l) + (cell_data(i)%alphas(j)+1)/2.0*cell_data(i)%n_vectors(1,j)
          end do
          A(l) = A(l)/cell_data(i)%volume
          col(l)=k
          row(l)=i
          l=l+1
          k = k + 1
          do j = 1,num_sides-2
            A(l) = A(l) + (cell_data(i)%alphas(j)+1)/2.0*cell_data(i)%n_vectors(2,j)
          end do
          A(l) = A(l)/cell_data(i)%volume
          col(l)=k
          row(l)=i
          l=l+1
!=========================================================================
!	Calculate the neighbouring entries from the factors.
!=========================================================================
          do j = 1,num_sides-2
            k = (cell_data(i)%neighbours(j)-1)*5+1
            A(l) = cell_data(i)%n_vectors(1,j)*(1-cell_data(i)%alphas(j))/2.0
            A(l) = A(l)/cell_data(i)%volume
            col(l)=k
            row(l)=i
            l = l+1
            k = k+1
            A(l) = cell_data(i)%n_vectors(2,j)*(1-cell_data(i)%alphas(j))/2.0
            A(l) = A(l)/cell_data(i)%volume
            col(l)=k
            row(l)=i
            l = l+1
	  end do
	end do
!=====================================================================
!	this is a loop over all wall_3 cells
!=====================================================================
	do i = c_wall_34+1,wall_3
          num_sides = cell_data(i)%num_sides
          call visco_get_alpha(i)
!=========================================================================
!	Sum up over all the sides for calculating the values
!       for the center of the cell.
!=========================================================================
	  k = (i-1)*5+1
          do j = 1,2
            A(l) = A(l) + (cell_data(i)%alphas(j)+1)/2.0*cell_data(i)%n_vectors(1,j)
          end do
          j = 4
          A(l) = A(l) + (cell_data(i)%alphas(j)+1)/2.0*cell_data(i)%n_vectors(1,j)
          A(l) = A(l)/cell_data(i)%volume
          col(l)=k
          row(l)=i
          l=l+1
          k = k + 1
          do j = 1,2
            A(l) = A(l) + (cell_data(i)%alphas(j)+1)/2.0*cell_data(i)%n_vectors(2,j)
          end do
          j = 4
          A(l) = A(l) + (cell_data(i)%alphas(j)+1)/2.0*cell_data(i)%n_vectors(2,j)
          A(l) = A(l)/cell_data(i)%volume
          col(l)=k
          row(l)=i
          l=l+1
!=========================================================================
!	Calculate the neighbouring entries from the factors.
!=========================================================================
          do j = 1,2
            k = (cell_data(i)%neighbours(j)-1)*5+1
            A(l) = cell_data(i)%n_vectors(1,j)*(1-cell_data(i)%alphas(j))/2.0
            A(l) = A(l)/cell_data(i)%volume
            col(l)=k
            row(l)=i
            l = l+1
            k = k+1
            A(l) = cell_data(i)%n_vectors(2,j)*(1-cell_data(i)%alphas(j))/2.0
            A(l) = A(l)/cell_data(i)%volume
            col(l)=k
            row(l)=i
            l = l+1
          end do
          j = 4
          k = (cell_data(i)%neighbours(j)-1)*5+1
          A(l) = cell_data(i)%n_vectors(1,j)*(1-cell_data(i)%alphas(j))/2.0
          A(l) = A(l)/cell_data(i)%volume
          col(l)=k
          row(l)=i
          l = l+1
          k = k+1
          A(l) = cell_data(i)%n_vectors(2,j)*(1-cell_data(i)%alphas(j))/2.0
          A(l) = A(l)/cell_data(i)%volume
          col(l)=k
          row(l)=i
          l = l+1
	end do
!=====================================================================
!	this is a loop over all c_wall_23 corner cells
!=====================================================================
	do i = wall_3+1,c_wall_23
          num_sides = cell_data(i)%num_sides
          call visco_get_alpha(i)
!=========================================================================
!	Sum up over all the sides for calculating the values
!       for the center of the cell.
!=========================================================================
          k = (i-1)*5+1
          j = 1
          A(l) = A(l) + (cell_data(i)%alphas(j)+1)/2.0*cell_data(i)%n_vectors(1,j)
          j = 4
          A(l) = A(l) + (cell_data(i)%alphas(j)+1)/2.0*cell_data(i)%n_vectors(1,j)
          A(l) = A(l)/cell_data(i)%volume
          col(l)=k
          row(l)=i
          l=l+1
          k = k + 1
          j = 1
          A(l) = A(l) + (cell_data(i)%alphas(j)+1)/2.0*cell_data(i)%n_vectors(2,j)
          j = 4
          A(l) = A(l) + (cell_data(i)%alphas(j)+1)/2.0*cell_data(i)%n_vectors(2,j)
          A(l) = A(l)/cell_data(i)%volume
          col(l)=k
          row(l)=i
          l=l+1
!=========================================================================
!	Calculate the neighbouring entries from the factors.
!=========================================================================
          j = 1
          k = (cell_data(i)%neighbours(j)-1)*5+1
          A(l) = cell_data(i)%n_vectors(1,j)*(1-cell_data(i)%alphas(j))/2.0
          A(l) = A(l)/cell_data(i)%volume
          col(l)=k
          row(l)=i
          l = l+1
          k = k+1
          A(l) = cell_data(i)%n_vectors(2,j)*(1-cell_data(i)%alphas(j))/2.0
          A(l) = A(l)/cell_data(i)%volume
          col(l)=k
          row(l)=i
          l = l+1
          j = 4
          k = (cell_data(i)%neighbours(j)-1)*5+1
          A(l) = cell_data(i)%n_vectors(1,j)*(1-cell_data(i)%alphas(j))/2.0
          A(l) = A(l)/cell_data(i)%volume
          col(l)=k
          row(l)=i
          l = l+1
          k = k+1
          A(l) = cell_data(i)%n_vectors(2,j)*(1-cell_data(i)%alphas(j))/2.0
          A(l) = A(l)/cell_data(i)%volume
          col(l)=k
          row(l)=i
          l = l+1
	end do
!=====================================================================
!	this is a loop over all inner volumes used.
!=====================================================================
        do i = c_wall_23+1,dim
!=========================================================================
!	only inner cells!!!
!=========================================================================
          num_sides = cell_data(i)%num_sides
          call visco_get_alpha(i)
!=========================================================================
!	Sum up over all the sides for calculating the values
!       for the center of the cell.
!=========================================================================
	  k = (i-1)*5+1
          do j = 1,num_sides
            A(l) = A(l) + (cell_data(i)%alphas(j)+1)/2.0*cell_data(i)%n_vectors(1,j)
          end do
          A(l) = A(l)/cell_data(i)%volume
          col(l)=k
          row(l)=i
          l=l+1
          k = k + 1
          do j = 1,num_sides
            A(l) = A(l) + (cell_data(i)%alphas(j)+1)/2.0*cell_data(i)%n_vectors(2,j)
          end do
          A(l) = A(l)/cell_data(i)%volume
          col(l)=k
          row(l)=i
          l=l+1
!=========================================================================
!	Calculate the neighbouring entries from the factors.
!=========================================================================
          do j = 1,num_sides
            k = (cell_data(i)%neighbours(j)-1)*5+1
            A(l) = cell_data(i)%n_vectors(1,j)*(1-cell_data(i)%alphas(j))/2.0
            A(l) = A(l)/cell_data(i)%volume
            col(l)=k
            row(l)=i
            l = l+1
            k = k+1
            A(l) = cell_data(i)%n_vectors(2,j)*(1-cell_data(i)%alphas(j))/2.0
            A(l) = A(l)/cell_data(i)%volume
            col(l)=k
            row(l)=i
            l = l+1
	  end do
        end do
!=========================================================================
! Uwe Laun: Die zu Null gesetzten Elemente werden nun entfernt und die
!           in C entstehenden Luecken zugeschoben.
!=========================================================================
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

        do i=k,l
         A(i)=0.0
         col(i)=0
         row(i)=0
        enddo

        l=k-1
        i=10

!=========================================================================
! Uwe Laun: Diese Routine ordnet die Feldelemente in das jagged-diagonal-Format,
!           da ansonsten die Struktur vom arithmetic-Modul nicht angenommen wurde.
!           (coredump in matrix_concentrate)
!=========================================================================
        call order_jagged(A,row,col,l,dim,dim,i)

!=========================================================================
! Uwe Laun: Die Feldlaenge wird in matrix_def mit size bestimmt; also
!          besser nur ein Feld der richtigen Laenge uebergeben!
!=========================================================================

        allocate(ele3(l))
        do i=1,l
         ele3(i)=A(i)
        enddo
        deallocate(A)

!=========================================================================
! Uwe Laun: Endlich die Struktur der Matrix in das sparse-Format des
!           arithmetic-Moduls wandeln.
!=========================================================================

        call matrix_def(A2,"A2",ele3,row,col,dim,dim*5)

        deallocate(ele3)

!        deallocate(alpha)
        
      end subroutine visco_init_A_1

!=====================================================================
      subroutine visco_init_C_1()
!=====================================================================
!       This subroutine initializes matrix C.
!	This matrix will remain constant during calculation.
!	It's the gradient matrix for the pressure.
!=====================================================================
! Uwe Laun: C ist nun eine eindimensionales Feld in welches nacheinander alle
!           explizit gesetzten Feldelemante der Matrix eingetragen werden.
!           row und col halten dabei die entsprechenden Zeilen und Spalten.
!=====================================================================

	use visco_declaration

	implicit none

	integer		:: i,k,l,m,j,num_sides
        real(double),allocatable,dimension(:) :: ele3

!=====================================================================
!	initialize C correctly
!=====================================================================
        if (.not.allocated(C)) allocate(C(10*dim),STAT=info)
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
!=========================================================================
!	get number of sides
!=========================================================================
        num_sides = cell_data(i)%num_sides
!        allocate(alpha(num_sides),STAT=info)
        call visco_get_alpha(i)
!=====================================================================
!	to ensure that pressure at the inflow boundary
!       is equal to that at cell center (unchanged over the inflow)
!       alpha(3) is set to one as well as alpha(4).
!=====================================================================
        cell_data(i)%alphas(3) = -1
        cell_data(i)%alphas(4) = -1
!=========================================================================
!	Calculate the neighbouring entries from the factors.
!=========================================================================
        k = (i-1)*5+1
        j = 1
	l = cell_data(i)%neighbours(j)
        C(m)   	= cell_data(i)%n_vectors(1,j)*(1+cell_data(i)%alphas(1))/2.0d0
        C(m) = C(m)/cell_data(i)%volume
        row(m)=k; col(m)=l; m=m+1
	k = k+1
        C(m)    = cell_data(i)%n_vectors(2,j)*(1+cell_data(i)%alphas(1))/2.0d0
        C(m) = C(m)/cell_data(i)%volume
        row(m)=k; col(m)=l; m=m+1

        k = (i-1)*5+1
        j = 2
	l = cell_data(i)%neighbours(j)
        C(m)   	= cell_data(i)%n_vectors(1,j)*(1+cell_data(i)%alphas(2))/2.0d0
        C(m) = C(m)/cell_data(i)%volume
        row(m)=k; col(m)=l; m=m+1
	k = k+1
        C(m)    = cell_data(i)%n_vectors(2,j)*(1+cell_data(i)%alphas(2))/2.0d0
        C(m) = C(m)/cell_data(i)%volume
        row(m)=k; col(m)=l; m=m+1
!=========================================================================
!	Sum up over all the sides for calculating the values
!       for the center of the cell.
!=========================================================================
        k = (i-1)*5+1
        do j = 1,num_sides
          C(m)    =  C(m) + cell_data(i)%n_vectors(1,j)*(1-cell_data(i)%alphas(j))/2.0d0
        end do
        C(m) = C(m)/cell_data(i)%volume
        row(m)=k; col(m)=i; m=m+1
	k = k+1
        do j = 1,num_sides
          C(m)    =  C(m) + cell_data(i)%n_vectors(2,j)*(1-cell_data(i)%alphas(j))/2.0d0
        end do
        C(m) = C(m)/cell_data(i)%volume
        row(m)=k; col(m)=i; m=m+1
!=====================================================================
!	inflow boundary
!=====================================================================
	do i = 2,inflow
!=========================================================================
!	get number of sides
!=========================================================================
          num_sides = cell_data(i)%num_sides
          call visco_get_alpha(i)
!=====================================================================
!	to ensure that pressure at the inflow boundary
!       is equal to that at cell center (unchanged over the inflow)
!       alpha(3) is set to one as well as alpha(4).
!=====================================================================
          cell_data(i)%alphas(3) = -1
!=========================================================================
!	Calculate the neighbouring entries from the factors.
!=========================================================================
          k = (i-1)*5+1
          j = 1
	  l = cell_data(i)%neighbours(j)
          C(m)   	= cell_data(i)%n_vectors(1,j)*(1+cell_data(i)%alphas(1))/2.0d0
          C(m) = C(m)/cell_data(i)%volume
          row(m)=k; col(m)=l; m=m+1
	  k = k+1
          C(m)    = cell_data(i)%n_vectors(2,j)*(1+cell_data(i)%alphas(1))/2.0d0
          C(m) = C(m)/cell_data(i)%volume
          row(m)=k; col(m)=l; m=m+1

          k = (i-1)*5+1
          j = 2
	  l = cell_data(i)%neighbours(j)
          C(m)    = cell_data(i)%n_vectors(1,j)*(1+cell_data(i)%alphas(2))/2.0d0
          C(m) = C(m)/cell_data(i)%volume
          row(m)=k; col(m)=l; m=m+1
	  k = k+1
          C(m)    = cell_data(i)%n_vectors(2,j)*(1+cell_data(i)%alphas(2))/2.0d0
          C(m) = C(m)/cell_data(i)%volume
          row(m)=k; col(m)=l; m=m+1
        
          k = (i-1)*5+1
          j = 4
	  l = cell_data(i)%neighbours(j)
          C(m)   	= cell_data(i)%n_vectors(1,j)*(1+cell_data(i)%alphas(4))/2.0d0
          C(m) = C(m)/cell_data(i)%volume
          row(m)=k; col(m)=l; m=m+1
	  k = k+1
          C(m)      = cell_data(i)%n_vectors(2,j)*(1+cell_data(i)%alphas(4))/2.0d0
          C(m) = C(m)/cell_data(i)%volume
          row(m)=k; col(m)=l; m=m+1
!=========================================================================
!	Sum up over all the sides for calculating the values
!       for the center of the cell.
!=========================================================================
          k = (i-1)*5+1
          do j = 1,num_sides
            C(m)    =  C(m) + cell_data(i)%n_vectors(1,j)*(1-cell_data(i)%alphas(j))/2.0d0
          end do
          C(m) = C(m)/cell_data(i)%volume
          row(m)=k; col(m)=i; m=m+1
	  k = k+1
          do j = 1,num_sides
            C(m)    =  C(m) + cell_data(i)%n_vectors(2,j)*(1-cell_data(i)%alphas(j))/2.0d0
          end do
          C(m) = C(m)/cell_data(i)%volume
          row(m)=k; col(m)=i; m=m+1
	end do
!=====================================================================
!	inflow upper corner
!=====================================================================
	i = c_inflow_2
!=========================================================================
!	get number of sides
!=========================================================================
        num_sides = cell_data(i)%num_sides
        call visco_get_alpha(i)
!=====================================================================
!	to ensure that pressure at the inflow boundary
!       is equal to that at cell center (unchanged over the inflow)
!       alpha(3) is set to one as well as alpha(2)(fixed wall).
!=====================================================================
        cell_data(i)%alphas(3) = -1
        cell_data(i)%alphas(2) = -1
!=========================================================================
!	Calculate the neighbouring entries from the factors.
!=========================================================================
        k = (i-1)*5+1
        j = 1
	l = cell_data(i)%neighbours(j)
        C(m)   	= cell_data(i)%n_vectors(1,j)*(1+cell_data(i)%alphas(1))/2.0d0
        C(m) = C(m)/cell_data(i)%volume
        row(m)=k; col(m)=l; m=m+1
	k = k+1
        C(m)    = cell_data(i)%n_vectors(2,j)*(1+cell_data(i)%alphas(1))/2.0d0
        C(m) = C(m)/cell_data(i)%volume
        row(m)=k; col(m)=l; m=m+1

        k = (i-1)*5+1
        j = 4
        l = cell_data(i)%neighbours(j)
        C(m)   	= cell_data(i)%n_vectors(1,j)*(1+cell_data(i)%alphas(4))/2.0d0
        C(m) = C(m)/cell_data(i)%volume
        row(m)=k; col(m)=l; m=m+1
	k = k+1
        C(m)    = cell_data(i)%n_vectors(2,j)*(1+cell_data(i)%alphas(4))/2.0d0
        C(m) = C(m)/cell_data(i)%volume
        row(m)=k; col(m)=l; m=m+1
!=========================================================================
!	Sum up over all the sides for calculating the values
!       for the center of the cell.
!=========================================================================
        k = (i-1)*5+1
        do  j = 1,num_sides
          C(m)    =  C(m) + cell_data(i)%n_vectors(1,j)*(1-cell_data(i)%alphas(j))/2.0d0
        end do
        C(m) = C(m)/cell_data(i)%volume
        row(m)=k; col(m)=i; m=m+1
        k = k+1
        do j = 1,num_sides
          C(m)    =  C(m) + cell_data(i)%n_vectors(2,j)*(1-cell_data(i)%alphas(j))/2.0d0
        end do
        C(m) = C(m)/cell_data(i)%volume
        row(m)=k; col(m)=i; m=m+1
!=====================================================================
!	outflow upper corner
!=====================================================================
	i = c_outflow_2 
!=========================================================================
!	get number of sides
!=========================================================================
        num_sides = cell_data(i)%num_sides
        call visco_get_alpha(i)
!=====================================================================
!	to ensure that pressure at the outflow boundary
!       is equal to that at cell center (unchanged over the outflow)
!       alpha(1) is set to one as well as alpha(2)(fixed wall).
!=====================================================================
        cell_data(i)%alphas(1) = 1
        cell_data(i)%alphas(2) = -1
!=========================================================================
!	Calculate the neighbouring entries from the factors.
!=========================================================================
        k = (i-1)*5+1
        j = 3
        l = cell_data(i)%neighbours(j)
        C(m)   	= cell_data(i)%n_vectors(1,j)*(1+cell_data(i)%alphas(3))/2.0d0
        C(m) = C(m)/cell_data(i)%volume
        row(m)=k; col(m)=l; m=m+1
	k = k+1
        C(m)    = cell_data(i)%n_vectors(2,j)*(1+cell_data(i)%alphas(3))/2.0d0
        C(m) = C(m)/cell_data(i)%volume
        row(m)=k; col(m)=l; m=m+1

        k = (i-1)*5+1
        j = 4
        l = cell_data(i)%neighbours(j)
        C(m)   	= cell_data(i)%n_vectors(1,j)*(1+cell_data(i)%alphas(4))/2.0d0
        C(m) = C(m)/cell_data(i)%volume
        row(m)=k; col(m)=l; m=m+1
	k = k+1
        C(m)    = cell_data(i)%n_vectors(2,j)*(1+cell_data(i)%alphas(4))/2.0d0
        C(m) = C(m)/cell_data(i)%volume
        row(m)=k; col(m)=l; m=m+1
!=========================================================================
!	Sum up over all the sides for calculating the values
!       for the center of the cell.
!=========================================================================
        k = (i-1)*5+1
        do j = 1,num_sides
          C(m)    =  C(m) + cell_data(i)%n_vectors(1,j)*(1-cell_data(i)%alphas(j))/2.0d0
        end do
        C(m) = C(m)/cell_data(i)%volume
        row(m)=k; col(m)=i; m=m+1
        k = k+1
        do j = 1,num_sides
          C(m)    =  C(m) + cell_data(i)%n_vectors(2,j)*(1-cell_data(i)%alphas(j))/2.0d0
        end do
        C(m) = C(m)/cell_data(i)%volume
        row(m)=k; col(m)=i; m=m+1
!=====================================================================
!	outflow volumes
!=====================================================================
	do i = c_outflow_2+1,outflow
!=========================================================================
!	get number of sides
!=========================================================================
          num_sides = cell_data(i)%num_sides
          call visco_get_alpha(i)
!=====================================================================
!	to ensure that pressure at the outflow boundary
!       is equal to that at cell center (unchanged over the outflow)
!       alpha(1) is set to.
!=====================================================================
        cell_data(i)%alphas(1) = 1
!=========================================================================
!	Calculate the neighbouring entries from the factors.
!=========================================================================
          k = (i-1)*5+1
          j = 2
          l = cell_data(i)%neighbours(j)
          C(m)   = cell_data(i)%n_vectors(1,j)*(1+cell_data(i)%alphas(2))/2.0d0
          C(m) = C(m)/cell_data(i)%volume
          row(m)=k; col(m)=l; m=m+1
	  k = k+1
          C(m)    = cell_data(i)%n_vectors(2,j)*(1+cell_data(i)%alphas(2))/2.0d0
          C(m) = C(m)/cell_data(i)%volume
          row(m)=k; col(m)=l; m=m+1

          k = (i-1)*5+1
          j = 3
          l = cell_data(i)%neighbours(j)
          C(m)   = cell_data(i)%n_vectors(1,j)*(1+cell_data(i)%alphas(3))/2.0d0
          C(m) = C(m)/cell_data(i)%volume
          row(m)=k; col(m)=l; m=m+1
	  k = k+1
          C(m)    = cell_data(i)%n_vectors(2,j)*(1+cell_data(i)%alphas(3))/2.0d0
          C(m) = C(m)/cell_data(i)%volume
          row(m)=k; col(m)=l; m=m+1
        
          k = (i-1)*5+1
          j = 4
	  l = cell_data(i)%neighbours(j)
          C(m)   = cell_data(i)%n_vectors(1,j)*(1+cell_data(i)%alphas(4))/2.0d0
          C(m) = C(m)/cell_data(i)%volume
          row(m)=k; col(m)=l; m=m+1
	  k = k+1
          C(m)   = cell_data(i)%n_vectors(2,j)*(1+cell_data(i)%alphas(4))/2.0d0
          C(m) = C(m)/cell_data(i)%volume
          row(m)=k; col(m)=l; m=m+1
!=========================================================================
!	Sum up over all the sides for calculating the values
!       for the center of the cell.
!=========================================================================
          k = (i-1)*5+1
          do j = 1,num_sides
            C(m)    =  C(m) + cell_data(i)%n_vectors(1,j)*(1-cell_data(i)%alphas(j))/2.0d0
          end do
          C(m) = C(m)/cell_data(i)%volume
          row(m)=k; col(m)=i; m=m+1
	  k = k+1
          do j = 1,num_sides
            C(m)    =  C(m) + cell_data(i)%n_vectors(2,j)*(1-cell_data(i)%alphas(j))/2.0d0
          end do
          C(m) = C(m)/cell_data(i)%volume
          row(m)=k; col(m)=i; m=m+1
	end do
!=====================================================================
!	outflow lower corner
!=====================================================================
	i = c_outflow_4
!=========================================================================
!	get number of sides
!=========================================================================
        num_sides = cell_data(i)%num_sides
        call visco_get_alpha(i)
!=====================================================================
!	to ensure that pressure at the outflow boundary
!       is equal to that at cell center (unchanged over the outflow)
!       alpha(1) is set to one as well as alpha(2)(fixed wall).
!=====================================================================
        cell_data(i)%alphas(1) = 1
        cell_data(i)%alphas(4) = -1
!=========================================================================
!	Calculate the neighbouring entries from the factors.
!=========================================================================
        k = (i-1)*5+1
        j = 2
        l = cell_data(i)%neighbours(j)
        C(m)   	= cell_data(i)%n_vectors(1,j)*(1+cell_data(i)%alphas(2))/2.0d0
        C(m) = C(m)/cell_data(i)%volume
        row(m)=k; col(m)=l; m=m+1
	k = k+1
        C(m)    = cell_data(i)%n_vectors(2,j)*(1+cell_data(i)%alphas(2))/2.0d0
        C(m) = C(m)/cell_data(i)%volume
        row(m)=k; col(m)=l; m=m+1

        k = (i-1)*5+1
        j = 3
        l = cell_data(i)%neighbours(j)
        C(m)   	= cell_data(i)%n_vectors(1,j)*(1+cell_data(i)%alphas(3))/2.0d0
        C(m) = C(m)/cell_data(i)%volume
        row(m)=k; col(m)=l; m=m+1
	k = k+1
        C(m)    = cell_data(i)%n_vectors(2,j)*(1+cell_data(i)%alphas(3))/2.0d0
        C(m) = C(m)/cell_data(i)%volume
        row(m)=k; col(m)=l; m=m+1
!=========================================================================
!	Sum up over all the sides for calculating the values
!       for the center of the cell.
!=========================================================================
        k = (i-1)*5+1
        do j = 1,num_sides
          C(m)    =  C(m) + cell_data(i)%n_vectors(1,j)*(1-cell_data(i)%alphas(j))/2.0d0
        end do
        C(m) = C(m)/cell_data(i)%volume
        row(m)=k; col(m)=i; m=m+1
        k = k+1
        do j = 1,num_sides
          C(m)    =  C(m) + cell_data(i)%n_vectors(2,j)*(1-cell_data(i)%alphas(j))/2.0d0
        end do
        C(m) = C(m)/cell_data(i)%volume
        row(m)=k; col(m)=i; m=m+1
!=====================================================================
!	wall_2 volumes
!=====================================================================
	do i = c_outflow_4+1,wall_2
!=========================================================================
!	get number of sides
!=========================================================================
          num_sides = cell_data(i)%num_sides
          call visco_get_alpha(i)
!=====================================================================
!	to ensure that pressure is unchanged across
!       the fixed boundary, boundary-alpha is set to 1
!=====================================================================
          cell_data(i)%alphas(2) = -1
!=========================================================================
!	Calculate the neighbouring entries from the factors.
!=========================================================================
          k = (i-1)*5+1
          j = 1
          l = cell_data(i)%neighbours(j)
          C(m)   	= cell_data(i)%n_vectors(1,j)*(1+cell_data(i)%alphas(1))/2.0d0
          C(m) = C(m)/cell_data(i)%volume
          row(m)=k; col(m)=l; m=m+1
	  k = k+1
          C(m)    = cell_data(i)%n_vectors(2,j)*(1+cell_data(i)%alphas(1))/2.0d0
          C(m) = C(m)/cell_data(i)%volume
          row(m)=k; col(m)=l; m=m+1

          do j = 3,4
	    k = (i-1)*5+1
            l = cell_data(i)%neighbours(j)
            C(m)   	= cell_data(i)%n_vectors(1,j)*(1+cell_data(i)%alphas(j))/2.0d0
            C(m) = C(m)/cell_data(i)%volume
            row(m)=k; col(m)=l; m=m+1
	    k = k+1
            C(m)    = cell_data(i)%n_vectors(2,j)*(1+cell_data(i)%alphas(j))/2.0d0
            C(m) = C(m)/cell_data(i)%volume
            row(m)=k; col(m)=l; m=m+1
          end do
!=========================================================================
!	Sum up over all the sides for calculating the values
!       for the center of the cell.
!=========================================================================
          k = (i-1)*5+1
          do j = 1,num_sides
            C(m)    =  C(m) + cell_data(i)%n_vectors(1,j)*(1-cell_data(i)%alphas(j))/2.0d0
          end do
          C(m) = C(m)/cell_data(i)%volume
          row(m)=k; col(m)=i; m=m+1
          k = k+1
          do j = 1,num_sides
            C(m)    =  C(m) + cell_data(i)%n_vectors(2,j)*(1-cell_data(i)%alphas(j))/2.0d0
          end do
          C(m) = C(m)/cell_data(i)%volume
          row(m)=k; col(m)=i; m=m+1
	end do
!=====================================================================
!	all the c_wall_12 cells (corner cells)
!=====================================================================
	do i = wall_2+1, c_wall_12
!=========================================================================
!	get number of sides
!=========================================================================
          num_sides = cell_data(i)%num_sides
          call visco_get_alpha(i)
!=====================================================================
!	to ensure that pressure is unchanged across
!       the fixed boundary, boundary-alpha is set to 1
!=====================================================================
          cell_data(i)%alphas(2) = -1
          cell_data(i)%alphas(1) = -1
!=========================================================================
!	Calculate the neighbouring entries from the factors.
!=========================================================================
          do j = 3,4
	    k = (i-1)*5+1
            l = cell_data(i)%neighbours(j)
            C(m)   	= cell_data(i)%n_vectors(1,j)*(1+cell_data(i)%alphas(j))/2.0d0
            C(m) = C(m)/cell_data(i)%volume
            row(m)=k; col(m)=l; m=m+1
	    k = k+1
            C(m)    = cell_data(i)%n_vectors(2,j)*(1+cell_data(i)%alphas(j))/2.0d0
            C(m) = C(m)/cell_data(i)%volume
            row(m)=k; col(m)=l; m=m+1
          end do
!=========================================================================
!	Sum up over all the sides for calculating the values
!       for the center of the cell.
!=========================================================================
          k = (i-1)*5+1
          do j = 1,num_sides
            C(m)    =  C(m) + cell_data(i)%n_vectors(1,j)*(1-cell_data(i)%alphas(j))/2.0d0
          end do
          C(m) = C(m)/cell_data(i)%volume
          row(m)=k; col(m)=i; m=m+1
          k = k+1
          do j = 1,num_sides
            C(m)    =  C(m) + cell_data(i)%n_vectors(2,j)*(1-cell_data(i)%alphas(j))/2.0d0
          end do
          C(m) = C(m)/cell_data(i)%volume
          row(m)=k; col(m)=i; m=m+1
	end do
!=====================================================================
!	this is a loop over all wall_1 volumes
!=====================================================================
	do i = c_wall_12+1,wall_1
!=========================================================================
!	get number of sides
!=========================================================================
          num_sides = cell_data(i)%num_sides
          call visco_get_alpha(i)
!=====================================================================
!	to ensure that pressure is unchanged across
!       the fixed boundary, boundary-alpha is set to 1
!=====================================================================
          cell_data(i)%alphas(1) = -1
!=========================================================================
!	Calculate the neighbouring entries from the factors.
!=========================================================================
          do j = 2,4
	    k = (i-1)*5+1
            l = cell_data(i)%neighbours(j)
            C(m)   	= cell_data(i)%n_vectors(1,j)*(1+cell_data(i)%alphas(j))/2.0d0
            C(m) = C(m)/cell_data(i)%volume
            row(m)=k; col(m)=l; m=m+1
	    k = k+1
            C(m)    = cell_data(i)%n_vectors(2,j)*(1+cell_data(i)%alphas(j))/2.0d0
            C(m) = C(m)/cell_data(i)%volume
            row(m)=k; col(m)=l; m=m+1
          end do
!=========================================================================
!	Sum up over all the sides for calculating the values
!       for the center of the cell.
!=========================================================================
          k = (i-1)*5+1
          do j = 1,num_sides
            C(m)    =  C(m) + cell_data(i)%n_vectors(1,j)*(1-cell_data(i)%alphas(j))/2.0d0
          end do
          C(m) = C(m)/cell_data(i)%volume
          row(m)=k; col(m)=i; m=m+1
          k = k+1
          do j = 1,num_sides
            C(m)    =  C(m) + cell_data(i)%n_vectors(2,j)*(1-cell_data(i)%alphas(j))/2.0d0
          end do
          C(m) = C(m)/cell_data(i)%volume
          row(m)=k; col(m)=i; m=m+1
	end do
!=====================================================================
!	this is a loop over all c_wall_41 corners
!=====================================================================
	do i = wall_1+1,c_wall_41
!=========================================================================
!	get number of sides
!=========================================================================
          num_sides = cell_data(i)%num_sides
          call visco_get_alpha(i)
!=====================================================================
!	to ensure that pressure is unchanged across
!       the fixed boundary, boundary-alpha is set to 1
!=====================================================================
          cell_data(i)%alphas(1) = -1
          cell_data(i)%alphas(4) = -1
!=========================================================================
!	Calculate the neighbouring entries from the factors.
!=========================================================================
          do j = 2,3
	    k = (i-1)*5+1
            l = cell_data(i)%neighbours(j)
            C(m)   	= cell_data(i)%n_vectors(1,j)*(1+cell_data(i)%alphas(j))/2.0d0
            C(m) = C(m)/cell_data(i)%volume
            row(m)=k; col(m)=l; m=m+1
	    k = k+1
            C(m)    = cell_data(i)%n_vectors(2,j)*(1+cell_data(i)%alphas(j))/2.0d0
            C(m) = C(m)/cell_data(i)%volume
            row(m)=k; col(m)=l; m=m+1
          end do
!=========================================================================
!	Sum up over all the sides for calculating the values
!       for the center of the cell.
!=========================================================================
          k = (i-1)*5+1
          do j = 1,num_sides
            C(m)    =  C(m) + cell_data(i)%n_vectors(1,j)*(1-cell_data(i)%alphas(j))/2.0d0
          end do
          C(m) = C(m)/cell_data(i)%volume
          row(m)=k; col(m)=i; m=m+1
          k = k+1
          do j = 1,num_sides
            C(m)    =  C(m) + cell_data(i)%n_vectors(2,j)*(1-cell_data(i)%alphas(j))/2.0d0
          end do
          C(m) = C(m)/cell_data(i)%volume
          row(m)=k; col(m)=i; m=m+1
	end do
!=====================================================================
!	wall_4 volumes
!=====================================================================
	do i = c_wall_41+1,wall_4
!=========================================================================
!	get number of sides
!=========================================================================
          num_sides = cell_data(i)%num_sides
          call visco_get_alpha(i)
!=====================================================================
!	to ensure that pressure is unchanged across
!       the fixed boundary, boundary-alpha is set to 1
!=====================================================================
          cell_data(i)%alphas(4) = -1
!=========================================================================
!	Calculate the neighbouring entries from the factors.
!=========================================================================
          do j = 1,3
	    k = (i-1)*5+1
            l = cell_data(i)%neighbours(j)
            C(m)   	= cell_data(i)%n_vectors(1,j)*(1+cell_data(i)%alphas(j))/2.0d0
            C(m) = C(m)/cell_data(i)%volume
            row(m)=k; col(m)=l; m=m+1
	    k = k+1
            C(m)    = cell_data(i)%n_vectors(2,j)*(1+cell_data(i)%alphas(j))/2.0d0
            C(m) = C(m)/cell_data(i)%volume
            row(m)=k; col(m)=l; m=m+1
          end do
!=========================================================================
!	Sum up over all the sides for calculating the values
!       for the center of the cell.
!=========================================================================
          k = (i-1)*5+1
          do j = 1,num_sides
            C(m)    =  C(m) + cell_data(i)%n_vectors(1,j)*(1-cell_data(i)%alphas(j))/2.0d0
          end do
          C(m) = C(m)/cell_data(i)%volume
          row(m)=k; col(m)=i; m=m+1
          k = k+1
          do j = 1,num_sides
            C(m)    =  C(m) + cell_data(i)%n_vectors(2,j)*(1-cell_data(i)%alphas(j))/2.0d0
          end do
          C(m) = C(m)/cell_data(i)%volume
          row(m)=k; col(m)=i; m=m+1
	end do
!=====================================================================
!	this is a loop over all c_wall_34 corners
!=====================================================================
	do i = wall_4+1, c_wall_34
!=========================================================================
!	get number of sides
!=========================================================================
          num_sides = cell_data(i)%num_sides
          call visco_get_alpha(i)
!=====================================================================
!	to ensure that pressure is unchanged across
!       the fixed boundary, boundary-alpha is set to 1
!=====================================================================
          cell_data(i)%alphas(3) = -1
          cell_data(i)%alphas(4) = -1
!=========================================================================
!	Calculate the neighbouring entries from the factors.
!=========================================================================
          do j = 1,2
	    k = (i-1)*5+1
            l = cell_data(i)%neighbours(j)
            C(m)   	= cell_data(i)%n_vectors(1,j)*(1+cell_data(i)%alphas(j))/2.0d0
            C(m) = C(m)/cell_data(i)%volume
            row(m)=k; col(m)=l; m=m+1
	    k = k+1
            C(m)    = cell_data(i)%n_vectors(2,j)*(1+cell_data(i)%alphas(j))/2.0d0
            C(m) = C(m)/cell_data(i)%volume
            row(m)=k; col(m)=l; m=m+1
          end do
!=========================================================================
!	Sum up over all the sides for calculating the values
!       for the center of the cell.
!=========================================================================
          k = (i-1)*5+1
          do j = 1,num_sides
            C(m)    =  C(m) + cell_data(i)%n_vectors(1,j)*(1-cell_data(i)%alphas(j))/2.0d0
          end do
          C(m) = C(m)/cell_data(i)%volume
          row(m)=k; col(m)=i; m=m+1
          k = k+1
          do j = 1,num_sides
            C(m)    =  C(m) + cell_data(i)%n_vectors(2,j)*(1-cell_data(i)%alphas(j))/2.0d0
          end do
          C(m) = C(m)/cell_data(i)%volume
          row(m)=k; col(m)=i; m=m+1
	end do
!=====================================================================
!	this is a loop over all wall_3 cells
!=====================================================================
	do i = c_wall_34+1,wall_3
!=========================================================================
!	get number of sides
!=========================================================================
          num_sides = cell_data(i)%num_sides
          call visco_get_alpha(i)
!=====================================================================
!	to ensure that pressure is unchanged across
!       the fixed boundary, boundary-alpha is set to 1
!=====================================================================
          cell_data(i)%alphas(3) = -1
!=========================================================================
!	Calculate the neighbouring entries from the factors.
!=========================================================================
          do j = 1,2
	    k = (i-1)*5+1
            l = cell_data(i)%neighbours(j)
            C(m)   	= cell_data(i)%n_vectors(1,j)*(1+cell_data(i)%alphas(j))/2.0d0
            C(m) = C(m)/cell_data(i)%volume
            row(m)=k; col(m)=l; m=m+1
	    k = k+1
            C(m)    = cell_data(i)%n_vectors(2,j)*(1+cell_data(i)%alphas(j))/2.0d0
            C(m) = C(m)/cell_data(i)%volume
            row(m)=k; col(m)=l; m=m+1
          end do
          j = 4
          k = (i-1)*5+1
          l = cell_data(i)%neighbours(j)
          C(m)        = cell_data(i)%n_vectors(1,j)*(1+cell_data(i)%alphas(j))/2.0d0
          C(m) = C(m)/cell_data(i)%volume
          row(m)=k; col(m)=l; m=m+1
          k = k+1
          C(m)    = cell_data(i)%n_vectors(2,j)*(1+cell_data(i)%alphas(j))/2.0d0
          C(m) = C(m)/cell_data(i)%volume
          row(m)=k; col(m)=l; m=m+1
!=========================================================================
!	Sum up over all the sides for calculating the values
!       for the center of the cell.
!=========================================================================
          k = (i-1)*5+1
          do j = 1,num_sides
            C(m)    =  C(m) + cell_data(i)%n_vectors(1,j)*(1-cell_data(i)%alphas(j))/2.0d0
          end do
          C(m) = C(m)/cell_data(i)%volume
          row(m)=k; col(m)=i; m=m+1
          k = k+1
          do j = 1,num_sides
            C(m)    =  C(m) + cell_data(i)%n_vectors(2,j)*(1-cell_data(i)%alphas(j))/2.0d0
          end do
          C(m) = C(m)/cell_data(i)%volume
          row(m)=k; col(m)=i; m=m+1
	end do
!=====================================================================
!	this is a loop over all c_wall_23 corner cells
!=====================================================================
	do i = wall_3+1,c_wall_23
!=========================================================================
!	get number of sides
!=========================================================================
          num_sides = cell_data(i)%num_sides
          call visco_get_alpha(i)
!=====================================================================
!	to ensure that pressure is unchanged across
!       the fixed boundary, boundary-alpha is set to 1
!=====================================================================
          cell_data(i)%alphas(2) = -1
          cell_data(i)%alphas(3) = -1
!=========================================================================
!	Calculate the neighbouring entries from the factors.
!=========================================================================
          j = 1
	  k = (i-1)*5+1
          l = cell_data(i)%neighbours(j)
          C(m)   	= cell_data(i)%n_vectors(1,j)*(1+cell_data(i)%alphas(j))/2.0d0
          C(m) = C(m)/cell_data(i)%volume
          row(m)=k; col(m)=l; m=m+1
	  k = k+1
          C(m)    = cell_data(i)%n_vectors(2,j)*(1+cell_data(i)%alphas(j))/2.0d0
          C(m) = C(m)/cell_data(i)%volume
          row(m)=k; col(m)=l; m=m+1
          j = 4
          k = (i-1)*5+1
          l = cell_data(i)%neighbours(j)
          C(m)        = cell_data(i)%n_vectors(1,j)*(1+cell_data(i)%alphas(j))/2.0d0
          C(m) = C(m)/cell_data(i)%volume
          row(m)=k; col(m)=l; m=m+1
          k = k+1
          C(m)    = cell_data(i)%n_vectors(2,j)*(1+cell_data(i)%alphas(j))/2.0d0
          C(m) = C(m)/cell_data(i)%volume
          row(m)=k; col(m)=l; m=m+1
!=========================================================================
!	Sum up over all the sides for calculating the values
!       for the center of the cell.
!=========================================================================
          k = (i-1)*5+1
          do j = 1,num_sides
            C(m)    =  C(m) + cell_data(i)%n_vectors(1,j)*(1-cell_data(i)%alphas(j))/2.0d0
          end do
          C(m) = C(m)/cell_data(i)%volume
          row(m)=k; col(m)=i; m=m+1
          k = k+1
          do j = 1,num_sides
            C(m)    =  C(m) + cell_data(i)%n_vectors(2,j)*(1-cell_data(i)%alphas(j))/2.0d0
          end do
          C(m) = C(m)/cell_data(i)%volume
          row(m)=k; col(m)=i; m=m+1
	end do
!=====================================================================
!       this is a loop over all inner volumes used.
!=====================================================================
	do i = c_wall_23+1,dim
!=========================================================================
!	get number of sides
!=========================================================================
          num_sides = cell_data(i)%num_sides
          call visco_get_alpha(i)
!=========================================================================
!	Calculate the neighbouring entries from the factors.
!=========================================================================
          do j = 1,num_sides
	    k = (i-1)*5+1
            l = cell_data(i)%neighbours(j)
            C(m)   	= cell_data(i)%n_vectors(1,j)*(1+cell_data(i)%alphas(j))/2.0d0
            C(m) = C(m)/cell_data(i)%volume
            row(m)=k; col(m)=l; m=m+1
	    k = k+1
            C(m)    = cell_data(i)%n_vectors(2,j)*(1+cell_data(i)%alphas(j))/2.0d0
            C(m) = C(m)/cell_data(i)%volume
            row(m)=k; col(m)=l; m=m+1
          end do
!=========================================================================
!	Sum up over all the sides for calculating the values
!       for the center of the cell.
!=========================================================================
          k = (i-1)*5+1
          do j = 1,num_sides
            C(m)    =  C(m) + cell_data(i)%n_vectors(1,j)*(1-cell_data(i)%alphas(j))/2.0d0
          end do
          C(m) = C(m)/cell_data(i)%volume
          row(m)=k; col(m)=i; m=m+1
          k = k+1
          do j = 1,num_sides
            C(m)    =  C(m) + cell_data(i)%n_vectors(2,j)*(1-cell_data(i)%alphas(j))/2.0d0
          end do
          C(m) = C(m)/cell_data(i)%volume
          row(m)=k; col(m)=i; m=m+1
	end do
!=====================================================================
! Uwe Laun: Die zu Null gesetzten Elemente werden nun entfernt und die 
!           in C entstehenden Luecken zugeschoben.
!=====================================================================

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
        do i=k,m
         C(i)=0.0
         row(i)=0
         col(i)=0
        enddo
        m=k-1
        i=5

!=====================================================================
! Uwe Laun: Diese Routine ordnet die Feldelemente in das jagged-diagonal-Format,
!           da ansonsten die Struktur vom atithmetic-Modul nicht angenommen wurde.
!           (coredump in matrix_concentrate)
!=====================================================================
        
        call order_jagged(C,row,col,m,dim,dim*5,i)

!=====================================================================
! Uwe Laun: Die Feldlaenge wird in matrix_def mit size bestimmt; also 
!          besser nur ein Feld der richtigen Laenge uebergeben!
!=====================================================================

        allocate(ele3(m))
        do i=1,m
         ele3(i)=C(i)
       enddo
       deallocate(C)
!       deallocate(alpha)

!=====================================================================
! Uwe Laun: Endlich die Struktur der Matrix in das sparse-Format des
!           arithmetic-Moduls wandeln.
!=====================================================================

        call matrix_def(C2,"C2",ele3,row,col,dim*5,dim)

!=====================================================================
! Uwe Laun: Hilfsfelder allokieren
!=====================================================================

!       deallocate(row)
!       deallocate(col)

       deallocate(ele3)

      end subroutine visco_init_C_1

!=====================================================================
      subroutine visco_get_alpha(i)
!=====================================================================
!	this subroutine calculates alpha values
!	that indicate whether up- or downwinding
!	has to be done.
!=====================================================================

      use visco_declaration

      implicit none

      integer :: i
	
      integer :: num_sides,j
      real(double) :: factor

!=========================================================================
!	get number of sides
!=========================================================================
        num_sides = cell_data(i)%num_sides
!=========================================================================
!       calculate the factor that decides about up- or downwinding.
!=========================================================================
      do j = 1,num_sides
        factor    = cell_data(i)%n_vectors(1,j)*                     &
                    cell_data(i)%unknowns%velocities(1) +            &
                    cell_data(i)%n_vectors(2,j)*                     &
                    cell_data(i)%unknowns%velocities(2)
        if (abs(factor) > 0.0) then
          cell_data(i)%alphas(j)  = int(sign(1.0d0,factor))
        else
	  if (j .eq. 2) then
	    cell_data(i)%alphas(j) =  1.0
	  else if (j .eq. 4) then
	    cell_data(i)%alphas(j) = -1.0
	  else if (j .eq. 1) then
	    cell_data(i)%alphas(j) = -1.0
	  else if (j .eq. 3) then
	    cell_data(i)%alphas(j) = 1.0
          end if
        end if
!=========================================================================
!	if we want to have central differences
!=========================================================================
!	cell_data(i)%alphas(j) = 0.0d0
      end do
	
      end subroutine visco_get_alpha

      end module visco_init_1
