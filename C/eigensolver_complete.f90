program eigensolver
  !-------------------------------------------------!
  ! This program is designed to illustrate the use  !
  ! of numerical methods and optimised software     !
  ! libraries to solve large Hermitian eigenvalue   !
  ! problems where only the lowest few eigenstates  !
  ! are required. Such problems arise in many       !
  ! electronic structure calculations, where we     !
  ! solve a large Schroedinger equation for just    !
  ! enough states to accommodate the number of      !
  ! electrons.                                      !
  !                                                 !
  ! In this program we will use the special case    !
  ! where the Hamiltonian is real and symmetric.    !
  ! This is a good approximation for large systems. !
  !-------------------------------------------------!
  ! Written by Phil Hasnip (University of York)     !
  ! with contributions from Dave Quigley            !
  ! (University of Warwick)                         !
  !-------------------------------------------------!
  ! Version 0.11, last modified 10th Sept. 2020     !
  !-------------------------------------------------!
  implicit none

  ! Define what we mean by double-precision
  integer, parameter                              :: dp=selected_real_kind(15,300)

  real(kind=dp),    dimension(:),   allocatable   :: H_kinetic  ! the kinetic energy operator
  real(kind=dp),    dimension(:),   allocatable   :: H_local    ! the local potential operator
  integer                                         :: num_wavevectors ! no. wavevectors in the basis
  integer                                         :: num_pw     ! no. plane-waves = 2*wavevectors+1
  integer                                         :: num_states ! no. eigenstates req'd

  complex(kind=dp), dimension(:,:), allocatable   :: trial_wvfn       ! current best guess at wvfn
  complex(kind=dp), dimension(:,:), allocatable   :: gradient         ! gradient of the energy
  complex(kind=dp), dimension(:,:), allocatable   :: search_direction ! direction to search along

  real(kind=dp),    dimension(:),   allocatable   :: eigenvalue       ! best guess at eigenvalues

  real(kind=dp), dimension(:),   allocatable      :: full_eigenvalue  ! LAPACK computed eigenvalues

  real(kind=dp)                                   :: exact_energy     ! total energy from exact eigenvalues	      
  real(kind=dp)                                   :: energy           ! current total energy	      
  real(kind=dp)                                   :: prev_energy      ! energy at last cycle	      
  real(kind=dp)                                   :: energy_tol       ! convergence tolerance on energy
  integer                                         :: status

  integer                                         :: max_iter         ! maximum number of iterations
  integer                                         :: iter,i,j,k,np,nb ! Loop counters

  ! extra variables for conjugate gradients
  complex(kind=dp), dimension(:,:), allocatable   :: prev_search_direction ! direction from prev. iteration
  real(kind=dp)                                   :: cg_gamma
  real(kind=dp)                                   :: cg_beta
  real(kind=dp)                                   :: cg_beta_old
  logical                                         :: reset_sd=.true.       ! whether to reset the CG

  ! rotation returned by diagonalise
  complex(kind=dp), dimension(:,:), allocatable   :: rotation

  ! Declare timing variables. NB intrinsic call is single-precision
  real(kind=kind(0.0))                            :: init_cpu_time
  real(kind=kind(0.0))                            :: curr_cpu_time
  real(kind=kind(0.0))                            :: exact_cpu_time
  real(kind=kind(0.0))                            :: iter_cpu_time

  ! BLAS
  complex(kind=dp),external :: zdotc

  ! No. nonzero wavevectors "G" in our wavefunction expansion
  num_wavevectors = 400

  ! No. plane-waves in our wavefunction expansion. One plane-wave has
  ! wavevector 0, and for all the others there are plane-waves at +/- G
  num_pw = 2*num_wavevectors+1

  ! No. eigenstates to compute
  num_states = 2

  ! Catch any nonsensical combinations of parameters
  if(num_states>=num_pw) stop 'Error, num_states must be less than num_pw'

  ! Set tolerance on the eigenvalue sum when using an iterative search. 
  ! The iterative search will stop when the change in the eigenvalue sum
  ! per iteration is less than this tolerance.
  energy_tol = 1.0e-10_dp

  ! Initialise the random number generator
  call init_random_seed()

  ! Now we need to allocate space for the Hamiltonian terms, and call
  ! the initialisation subroutine
  allocate(H_kinetic(num_pw),stat=status)
  if(status/=0) stop 'Error allocating RAM to H_kinetic'

  allocate(H_local(num_pw),stat=status)
  if(status/=0) stop 'Error allocating RAM to H_local'

  write(*,*) 'Initialising Hamiltonian...'

  ! Initialise and build the Hamiltonian, comprising two terms: the kinetic energy,
  ! which is a diagonal matrix in the plane-wave basis (Fourier space); and the local
  ! potential energy, which is a diagonal matrix in real space (direct space).
  !
  ! The nonzero elements of each of these terms are stored in H_kinetic and H_local
  call init_H(num_pw,H_kinetic,H_local)

  ! Perform an exact diagonalisation for comparison
  call cpu_time(init_cpu_time)

  allocate(full_eigenvalue(num_pw),stat=status)
  if(status/=0) stop 'Error allocating RAM to full_eigenvalue'

  write(*,*) 'Starting full diagonalisation...'
  write(*,*) ' '
  full_eigenvalue(:) = 0.0_dp

  !---------------------------------------------------------------------!
  ! Perform full diagonalisation using LAPACK.                          !
  !---------------------------------------------------------------------!
  call exact_diagonalisation(num_pw,H_kinetic,H_local,full_eigenvalue)
  call cpu_time(curr_cpu_time)

  exact_cpu_time = curr_cpu_time-init_cpu_time
  
  ! Write out the numerically exact results from LAPACK
  write(*,'(1x,a,t17,a,t38,a)') 'State','Eigenvalue'
  do nb=1,num_states
    write(*,'(1x,i5,5x,2g20.10)') nb,full_eigenvalue(nb)
  end do    
  write(*,*) ' '

  ! Energy is the sum of the eigenvalues of the occupied states
  exact_energy = sum(full_eigenvalue(1:num_states))
  write(*,'(1x,a,g20.10)') 'Ground state energy:   ',exact_energy

  write(*,*) 'Full diagonalisation took ',exact_cpu_time,' secs'
  write(*,*) ' '
  
  ! Allocate memory for iterative eigenvector search
  ! Each column contains the plane wave co-efficients for a single particle wavefunction
  ! and there are num_states particles.
  allocate(trial_wvfn(num_pw,num_states),gradient(num_pw,num_states),search_direction(num_pw,num_states), &
           prev_search_direction(num_pw,num_states),stat=status)
  if(status/=0) stop 'Error allocating RAM to trial_wvfn, gradient, search_direction and prev_search_direction'

  ! We have num_states eigenvalue estimates
  allocate(eigenvalue(num_states),stat=status)
  if(status/=0) stop 'Error allocating RAM to eigenvalues'

  ! Allocate rotation matrix
  allocate(rotation(num_states,num_states),stat=status)
  if(status/=0) stop 'Error allocating RAM to rotation'

  write(*,*) 'Starting iterative search for eigenvalues'
  write(*,*) ' '
  write(*,*) '+-----------+----------------+-----------------+'
  write(*,*) '| iteration |     energy     |  energy change  |'
  write(*,*) '+-----------+----------------+-----------------+'

  call cpu_time(init_cpu_time)

  ! We start from a random guess for trial_wvfn
  ! this routine is in libhamiltonian
  call randomise_state(num_pw,num_states,trial_wvfn)

  ! All the wavefunctions should be normalised and orthogonal to each other
  ! at every iteration. We enforce this in the initial random state here.
  call orthonormalise(num_pw,num_states,trial_wvfn)

  ! Apply the H to this state, store the result H.wvfn in gradient. As yet this is
  ! unconstrained, i.e. following this gradient will break orthonormality.
  call apply_H(num_pw,num_states,trial_wvfn,H_kinetic,H_local,gradient)

  ! Compute the eigenvalues, i.e. the Rayleigh quotient for each eigenpair
  ! Note that we don't compute a denominator here because our trial states
  ! are normalised.
  do nb=1,num_states
    eigenvalue(nb) = real(zdotc(num_pw,trial_wvfn(1,nb),1,gradient(1,nb),1),dp)
  end do

  ! Energy is the sum of the eigenvalues.
  energy = sum(eigenvalue)

  write(*,'(a,g15.8,a,t49,a)') ' |  Initial  | ',energy,'|','|'

  ! In case of problems, we cap the total number of iterations
  max_iter = 40000

  !----------------------------------------------------!
  ! Begin the iterative search for eigenvalues         !
  !----------------------------------------------------!
  main_loop: do iter=1,max_iter

    prev_energy = energy

    !---------------------------------------------------------------------!
    ! The constrained gradient is H.wvfn - (wvfn+.H.wvfn)*wvfn            !
    ! -- i.e. it is orthogonal to wvfn which we enforce by                !
    ! calling the routine below. Remember H.wvfn is already               !
    ! stored as gradient.                                                 !
    !---------------------------------------------------------------------!
    call orthogonalise(num_pw,num_states,gradient,trial_wvfn)

    ! The steepest descent direction is minus the gradient
    call zcopy(num_pw*num_states,gradient,1,search_direction,1)
    call zdscal(num_pw*num_states,-1.0_dp,search_direction,1)

    !---------------------------------------------------------------------!
    ! Any modifications to the search direction go here, e.g.             !
    ! preconditioning, implementation of conjugate gradients etc.         !
    !---------------------------------------------------------------------!

    ! Precondition, which breaks orthogonalisation so also re-orthogonalise
    call precondition(num_pw,num_states,search_direction,trial_wvfn,H_kinetic)
    call orthogonalise(num_pw,num_states,search_direction,trial_wvfn)    

    ! Use Fletcher-Reeves conjugate gradients (CG)
    cg_beta = 0.0_dp
    do i=1,num_states
       cg_beta = cg_beta + real(zdotc(num_pw,search_direction(:,i),1,gradient(:,i),1),dp)
    end do

    if(.not.reset_sd) then  ! this is false on the first call, so cg_beta_old will be defined
      cg_gamma         = cg_beta/cg_beta_old
      cg_beta_old      = cg_beta
      call zaxpy(num_pw*num_states,cmplx(cg_gamma,0.0_dp,dp),prev_search_direction,1,search_direction,1)

      call orthogonalise(num_pw,num_states,search_direction,trial_wvfn)    
    
    end if

    cg_beta_old = cg_beta

    call zcopy(num_pw*num_states,search_direction,1,prev_search_direction,1);

    ! Search along this direction for the best approx. eigenvector, i.e. the lowest energy
    call line_search(num_pw,num_states,trial_wvfn,H_kinetic,H_local,search_direction,gradient,eigenvalue,energy)

     ! Check convergence
    if(abs(prev_energy-energy)<energy_tol) then
      if(reset_sd) then 
        write(*,*) '+-----------+----------------+-----------------+'
        write(*,*) 'Eigenvalues converged'
        exit
      else
        reset_sd = .true.
      end if
    else
      reset_sd = .false.
    end if

    ! Reset the CG every 5 steps, to prevent it stagnating
    if(mod(iter,5)==0) reset_sd=.true.

    ! Energy is the sum of the eigenvalues
    energy = sum(eigenvalue)

    write(*,'(a,i5,1x,a,g15.8,a,g15.8,a)') ' |    ',iter,' | ',energy,'| ',prev_energy-energy,' |'

  end do main_loop

  ! If you have multiple states, you may get a linear combination of them rather than the
  ! pure states. This can be fixed by computing the Hamiltonian matrix in the basis of the
  ! trial states (from trial_wvfn and gradient), and then diagonalising that matrix.
  !
  ! In other words, we rotate the states to be as near as possible to the true eigenstates 
  !
  ! This can be done in the "diagonalise" routine, BUT YOU NEED TO COMPLETE IT

  call diagonalise(num_pw,num_states,trial_wvfn,gradient,eigenvalue,rotation)

  call cpu_time(curr_cpu_time)

  iter_cpu_time = curr_cpu_time-init_cpu_time

  write(*,*) 'Iterative search took ',iter_cpu_time,' secs'
  write(*,*) ' '

  ! Finally, summarise the results
  write(*,*) '=============== FINAL RESULTS ==============='
  write(*,'(a,t28,a)') ' State','Eigenvalue'
  write(*,'(t18,a,t39,a)') 'Iterative','Exact'
  do nb=1,num_states
    write(*,'(i5,5x,2g20.8)') nb,eigenvalue(nb),full_eigenvalue(nb)
  end do    
  write(*,*) '---------------------------------------------'
  write(*,'(a,t17,g14.8,6x,g14.8)') ' Energy   ',energy,exact_energy
  write(*,*) ' '
  write(*,*) '---------------------------------------------'
  write(*,'(a,t19,g11.5,6x,g11.5)') ' Time taken (s) ',iter_cpu_time,exact_cpu_time
  write(*,*) '============================================='
  write(*,*) ' '

  call output_results(num_pw,num_states,H_local,trial_wvfn)

  ! Deallocate memory
  deallocate(full_eigenvalue,stat=status)
  if(status/=0) stop 'Error deallocating RAM from full_eigenvalue'

  deallocate(search_direction,stat=status)
  if(status/=0) stop 'Error deallocating RAM from search_direction'

  deallocate(gradient,stat=status)
  if(status/=0) stop 'Error deallocating RAM from gradient'

  deallocate(trial_wvfn,stat=status)
  if(status/=0) stop 'Error deallocating RAM from trial_wvfn'

  deallocate(prev_search_direction,stat=status)
  if(status/=0) stop 'Error deallocating RAM from prev_search_direction'

  deallocate(rotation,stat=status)
  if(status/=0) stop 'Error deallocating RAM from rotation'


end program eigensolver

  !---------------------------------------------------------------------!
  !           -- THESE ARE THE COMPLETED SUBROUTINES --           !
  !---------------------------------------------------------------------!

  subroutine exact_diagonalisation(num_pw,H_kinetic,H_local,full_eigenvalue)
    !-------------------------------------------------!
    ! This subroutine takes a compact representation  !
    ! of the matrix H, constructs the full H, and     !
    ! diagonalises to get the whole eigenspectrum.    !
    !-------------------------------------------------!
    implicit none

    ! Define what we mean by double-precision
    integer, parameter                           :: dp=selected_real_kind(15,300)

    integer,                                    intent(in)  :: num_pw
    real(kind=dp),  dimension(num_pw),          intent(in)  :: H_kinetic
    real(kind=dp),  dimension(num_pw),          intent(in)  :: H_local
    real(kind=dp),  dimension(num_pw),          intent(out) :: full_eigenvalue

    complex(kind=dp), dimension(:,:), allocatable :: full_H
    real(kind=dp),    dimension(:),   allocatable :: lapack_real_work
    complex(kind=dp), dimension(:),   allocatable :: lapack_cmplx_work
    integer                                       :: status,np

    ! First we allocate and construct the full num_pw x num_pw Hamiltonian
    allocate(full_H(num_pw,num_pw),stat=status)
    if(status/=0) stop 'Error allocating RAM to full_H in exact_diagonalisation'

    call construct_full_H(num_pw,H_kinetic,H_local,full_H)

    !---------------------------------------------------------------------!
    ! Use LAPACK's zheev to get the eigenvalues and eigenvectors of       !
    ! full_H (NB the matrix is Hermitian)                                 !
    !---------------------------------------------------------------------!
    allocate(lapack_real_work(3*num_pw-2),stat=status)
    if(status/=0) stop 'Error allocating RAM to lapack_real_work in exact_diagonalisation'

    allocate(lapack_cmplx_work(2*num_pw-1),stat=status)
    if(status/=0) stop 'Error allocating RAM to lapack_cmplx_work in exact_diagonalisation'

    lapack_real_work  = 0.0_dp
    lapack_cmplx_work = cmplx(0.0_dp,0.0_dp,dp)

    call zheev('V','U',num_pw,full_H,size(full_H,1),full_eigenvalue,lapack_cmplx_work, &
             & size(lapack_cmplx_work,1),lapack_real_work,status)
    if(status/=0) stop 'Error with zheev in exact_diagonalisation'

    deallocate(lapack_real_work,stat=status)
    if(status/=0) stop 'Error deallocating RAM from lapack_real_work in exact_diagonalisation'

    deallocate(lapack_cmplx_work,stat=status)
    if(status/=0) stop 'Error deallocating RAM from lapack_cmplx_work in exact_diagonalisation'

    ! Deallocate memory
    deallocate(full_H,stat=status)
    if(status/=0) stop 'Error deallocating RAM from full_H in exact_diagonalisation'

    return
  end subroutine exact_diagonalisation

  subroutine orthogonalise(num_pw,num_states,state,ref_state)
    !-------------------------------------------------!
    ! This subroutine takes a set of states and       !
    ! orthogonalises them to a set of reference       !
    ! states.                                         !
    !-------------------------------------------------!
    implicit none

    ! Define what we mean by double-precision
    integer, parameter                           :: dp=selected_real_kind(15,300)

    integer,                                        intent(in)    :: num_pw
    integer,                                        intent(in)    :: num_states
    complex(kind=dp), dimension(num_pw,num_states), intent(inout) :: state
    complex(kind=dp), dimension(num_pw,num_states), intent(in)    :: ref_state

    complex(kind=dp)                             :: overlap
    integer                                      :: nb1,nb2,np
    integer                                      :: status

    do nb2=1,num_states
      do nb1=1,num_states
        overlap = cmplx(0.0_dp,0.0_dp,dp)
        do np=1,num_pw
          overlap = overlap + conjg(ref_state(np,nb1))*state(np,nb2)
        end do

        do np=1,num_pw
          state(np,nb2) = state(np,nb2) - overlap*ref_state(np,nb1)
        end do
      end do
    end do

    return
    
  end subroutine orthogonalise

  subroutine precondition(num_pw,num_states,search_direction,trial_wvfn,H_kinetic)
    !-------------------------------------------------!
    ! This subroutine takes a search direction and    !
    ! applies a simple kinetic energy-based           !
    ! preconditioner to improve the conditioning of   !
    ! the eigenvalue search.                          !
    !-------------------------------------------------!
    implicit none

    ! Define what we mean by double-precision
    integer, parameter                           :: dp=selected_real_kind(15,300)

    integer,                                        intent(in)    :: num_pw
    integer,                                        intent(in)    :: num_states
    complex(kind=dp), dimension(num_pw,num_states), intent(inout) :: search_direction
    complex(kind=dp), dimension(num_pw,num_states), intent(in)    :: trial_wvfn
    real(kind=dp),    dimension(num_pw),            intent(in)    :: H_kinetic

    integer                                      :: np,nb
    real(kind=dp)                                :: kinetic_eigenvalue,x,temp

    do nb=1,num_states
       !---------------------------------------------------------------------!
       ! Compute the kinetic energy "eigenvalue" for state nb.               !
       ! We don't have the true wavefunction yet, but our best guess is in   !
       ! trial_wvfn so we estimate the kinetic energy Ek as:                 !
       !                                                                     !
       !     E_k = trial_wvfn^+ H_kinetic trial_wvfn                         !
       !                                                                     !
       ! where "^+" means take the Hermitian conjugate (transpose it and     !
       ! take the complex conjugate of each element). H_kinetic is a         !
       ! diagonal matrix, so rather than store a num_pw x num_pw matrix with !
       ! most elements being zero, we instead just store the num_pw non-zero !
       ! elements in a 1D array. Thus the nth element of the result of       !
       ! operating with the kinetic energy operator on the wavefunction is:  !
       !                                                                     !
       !     (H_kinetic trial_wvfn)(n) = H_kinetic(n)*trial_wvfn(n)          !
       !                                                                     !
       !---------------------------------------------------------------------!
       kinetic_eigenvalue = sum(H_kinetic(:)*abs(trial_wvfn(:,nb))**2)

       do np=1,num_pw

          !---------------------------------------------------------------------!
          ! Compute and apply the preconditioning, using the estimate of        !
          ! trial_wvfn's kinetic energy computed above and the kinetic energy   !
          ! associated with each plane-wave basis function                      !
          !---------------------------------------------------------------------!
          x=H_kinetic(np)/kinetic_eigenvalue
          temp = 8.0_dp + x*(4.0_dp + x*(2.0_dp + 1.0_dp*x))

          search_direction(np,nb) = search_direction(np,nb)*temp/(temp+x**4)
          
       end do

    end do

    return

  end subroutine precondition

  subroutine diagonalise(num_pw,num_states,state,H_state,eigenvalues,rotation)
    !-------------------------------------------------!
    ! This subroutine takes a set of states and       !
    ! H acting on those states, and transforms the    !
    ! states to diagonalise <state|H|state>.          !
    !-------------------------------------------------!
    implicit none

    ! Define what we mean by double-precision
    integer, parameter                           :: dp=selected_real_kind(15,300)

    integer,                                            intent(in)    :: num_pw
    integer,                                            intent(in)    :: num_states
    complex(kind=dp), dimension(num_pw,num_states),     intent(inout) :: state
    complex(kind=dp), dimension(num_pw,num_states),     intent(inout) :: H_state
    real(kind=dp), dimension(num_states),               intent(inout) :: eigenvalues
    complex(kind=dp), dimension(num_states,num_states), intent(out)   :: rotation

    integer                                      :: nb1,nb2
    integer                                      :: optimal_size,status

    real(kind=dp),    dimension(:),  allocatable :: lapack_real_work
    complex(kind=dp), dimension(:),  allocatable :: lapack_cmplx_work

    complex(kind=dp), external :: zdotc

    ! Compute the subspace H matrix and store in rotation array
    do nb2=1,num_states
      do nb1=1,num_states
        rotation(nb1,nb2) = zdotc(num_pw,state(:,nb1),1,H_state(:,nb2),1)
      end do
    end do

    ! Compute diagonalisation
    !
    ! allocate and zero the LAPACK work arrays 
    allocate(lapack_real_work(3*num_states-2),stat=status)
    if(status/=0) stop 'Error allocating RAM to lapack_real_work in exact_diagonalisation'

    allocate(lapack_cmplx_work(2*num_states-1),stat=status)
    if(status/=0) stop 'Error allocating RAM to lapack_cmplx_work in exact_diagonalisation'

    lapack_real_work  = 0.0_dp
    lapack_cmplx_work = cmplx(0.0_dp,0.0_dp,dp)

    ! Use LAPACK to diagonalise the Hamiltonian in this subspace
    ! NB H is Hermitian
    call zheev('V','U',num_states,rotation,size(rotation,1),eigenvalues,lapack_cmplx_work, &
             & size(lapack_cmplx_work,1),lapack_real_work,status)
    if(status/=0) stop 'Error with zheev in exact_diagonalisation'

    deallocate(lapack_real_work,stat=status)
    if(status/=0) stop 'Error deallocating RAM from lapack_real_work in exact_diagonalisation'

    deallocate(lapack_cmplx_work,stat=status)
    if(status/=0) stop 'Error deallocating RAM from lapack_cmplx_work in exact_diagonalisation'

    ! Finally apply the diagonalising rotation to state
    ! (and also to H_state, to keep it consistent with state)
    call transform(num_pw,num_states,state,rotation)
    call transform(num_pw,num_states,H_state,rotation)

    return
    
  end subroutine diagonalise

  ! -- THE FOLLOWING SUBROUTINES ARE ALREADY WRITTEN --
  !       (you may wish to optimise them though)

  subroutine orthonormalise(num_pw,num_states,state)
    !-------------------------------------------------!
    ! This subroutine takes a set of states and       !
    ! orthonormalises them.                           !
    !-------------------------------------------------!
    implicit none

    ! Define what we mean by double-precision
    integer, parameter                           :: dp=selected_real_kind(15,300)

    integer,                                        intent(in)    :: num_pw
    integer,                                        intent(in)    :: num_states
    complex(kind=dp), dimension(num_pw,num_states), intent(inout) :: state

    complex(kind=dp), dimension(:,:), allocatable   :: overlap
    complex(kind=dp)                                :: tmp_overlap
    integer                                      :: nb2,nb1,np
    integer                                      :: status

    complex(kind=dp), external :: zdotc

    allocate(overlap(num_states,num_states),stat=status)
    if(status/=0) stop 'Error allocating RAM to overlap in orthonormalise'

    ! Compute the overlap matrix
    do nb2=1,num_states
      do nb1=1,num_states
        overlap(nb1,nb2) = zdotc(num_pw,state(:,nb1),1,state(:,nb2),1)
      end do
    end do

    ! Compute orthogonalising transformation

    ! First compute Cholesky (U.U^H) factorisation of the overlap matrix
    call zpotrf('U',num_states,overlap,num_states,status)
    if(status/=0) stop 'zpotrf failed in orthonormalise'

    ! invert this upper triangular matrix                                       
    call ztrtri('U','N',num_states,overlap,num_states,status)
    if(status/=0) stop 'ztrtri failed in orthonormalise'

    ! Set lower triangle to zero                                                
    do nb2 = 1,num_states
      do nb1 = nb2+1,num_states
        overlap(nb1,nb2)=cmplx(0.0_dp,0.0_dp,dp)
      end do
    end do

    ! overlap array now contains the (upper triangular) orthonormalising transformation
    call transform(num_pw,num_states,state,overlap)

    deallocate(overlap,stat=status)
    if(status/=0) stop 'Error deallocating RAM from overlap in orthonormalise'

    return
    
  end subroutine orthonormalise

  subroutine transform(num_pw,num_states,state,transformation)
    !-------------------------------------------------!
    ! This subroutine takes a set of states and       !
    ! applies a linear transformation to them.        !
    !-------------------------------------------------!
    implicit none
 
    ! Define what we mean by double-precision
    integer, parameter                           :: dp=selected_real_kind(15,300)

    integer,                                            intent(in)    :: num_pw
    integer,                                            intent(in)    :: num_states
    complex(kind=dp), dimension(num_pw,num_states),     intent(inout) :: state
    complex(kind=dp), dimension(num_states,num_states), intent(inout) :: transformation

    integer                                        :: nb2,nb1,np
    integer                                        :: status
    complex(kind=dp), dimension(:,:), allocatable  :: new_state

    allocate(new_state(size(state,1),size(state,2)),stat=status)
    if(status/=0) stop 'Error allocating RAM to new_state in transform'

    ! Apply transformation to state, and put in new_state
    new_state = cmplx(0.0_dp,0.0_dp,dp)
    do nb2=1,num_states
      do nb1=1,num_states
        do np=1,num_pw
          new_state(np,nb1) = new_state(np,nb1) + state(np,nb2)*transformation(nb2,nb1)
        end do
      end do
    end do

    ! Update state with the transformed data and deallocate memory
    state = new_state

    deallocate(new_state,stat=status)
    if(status/=0) stop 'Error deallocating RAM from new_state in transform'

    return
    
  end subroutine transform

  subroutine line_search(num_pw,num_states,approx_state,H_kinetic,H_local,direction, &
                       & gradient,eigenvalue,energy)
    !-------------------------------------------------!
    ! This subroutine takes an approximate eigenstate !
    ! and searches along a direction to find an       !
    ! improved approximation.                         !
    !-------------------------------------------------!
    implicit none

    ! Define what we mean by double-precision
    integer, parameter                           :: dp=selected_real_kind(15,300)

    integer,                                        intent(in)    :: num_pw
    integer,                                        intent(in)    :: num_states
    complex(kind=dp), dimension(num_pw,num_states), intent(inout) :: approx_state
    real(kind=dp), dimension(num_pw),               intent(in)    :: H_kinetic
    real(kind=dp), dimension(num_pw),               intent(in)    :: H_local
    complex(kind=dp), dimension(num_pw,num_states), intent(inout) :: direction
    complex(kind=dp), dimension(num_pw,num_states), intent(inout) :: gradient
    real(kind=dp),                                  intent(inout) :: energy
    real(kind=dp), dimension(num_states),           intent(inout) :: eigenvalue

    real(kind=dp)                                 :: init_energy
    real(kind=dp)                                 :: tmp_energy
!    real(kind=dp), save                           :: trial_step=0.00004_dp
    real(kind=dp), save                           :: trial_step=0.4_dp
    real(kind=dp)                                 :: step
    real(kind=dp)                                 :: opt_step
    complex(kind=dp), dimension(:,:), allocatable :: tmp_state
    real(kind=dp)                                 :: d2E_dstep2
    real(kind=dp)                                 :: best_step
    real(kind=dp)                                 :: best_energy
    real(kind=dp)                                 :: denergy_dstep
    real(kind=dp)                                 :: mean_norm
    integer                                       :: loop
    integer                                       :: nb
    integer                                       :: status

    ! BLAS external functions
    real(kind=dp),    external :: dznrm2
    complex(kind=dp), external :: zdotc

    ! To try to keep a convenient step length, we reduce the size of the search direction
    mean_norm = 0.0_dp
    do nb=1,size(approx_state,2)
      mean_norm = mean_norm + dznrm2(num_pw,direction(1,nb),1)
    end do
    mean_norm = mean_norm/real(size(approx_state,2),dp)

    call zdscal(num_pw*num_states,1.0_dp/mean_norm,direction,1)

    ! The rate-of-change of the energy is just 2*Re(conjg(direction).gradient)
    denergy_dstep = 0.0_dp 
    do nb=1,size(approx_state,2)
      denergy_dstep = denergy_dstep + 2*real(zdotc(num_pw,direction(:,nb),1,gradient(:,nb),1),dp)
    end do

    allocate(tmp_state(size(approx_state,1),size(approx_state,2)),stat=status)
    if(status/=0) stop 'Error allocating RAM to tmp_state in line_search'

    best_step   = 0.0_dp
    best_energy = energy

    ! First take a trial step in the direction
    step = trial_step

    ! We find a trial step that lowers the energy:
    do loop=1,10
       
      tmp_state = approx_state + step*direction

      call orthonormalise(num_pw,num_states,tmp_state)

      ! Apply the Hamiltonian to this state
      call apply_H(num_pw,num_states,tmp_state,H_kinetic,H_local,gradient)

      ! Compute the new energy estimate
      tmp_energy = 0.0_dp 
      do nb=1,num_states
        tmp_energy = tmp_energy + real(zdotc(num_pw,tmp_state(:,nb),1,gradient(:,nb),1),dp)
      end do

      if(tmp_energy<energy) then
        exit
      else
        d2E_dstep2 = (tmp_energy - energy - step*denergy_dstep )/(step**2)
        if(d2E_dstep2<0.0_dp) then
          if(tmp_energy<energy) then
            exit
          else
            step = step/4.0_dp
          end if
        else
          step  = -denergy_dstep/(2*d2E_dstep2)
        end if
      end if

    end do

    if(tmp_energy<best_energy) then
      best_step   = step
      best_energy = tmp_energy
    end if

    ! We now have the initial eigenvalue, the initial gradient, and a trial step
    ! -- we fit a parabola, and jump to the estimated minimum position
    ! Set default step and energy
    d2E_dstep2 = (tmp_energy - energy - step*denergy_dstep )/(step**2)


    if(d2E_dstep2<0.0_dp) then
      ! Parabolic fit gives a maximum, so no good
      write(*,'(a)') '** Warning, parabolic stationary point is a maximum **'

      if(tmp_energy<energy) then
        opt_step = step
      else
        opt_step = 0.1_dp*step
      end if
    else
      opt_step  = -denergy_dstep/(2*d2E_dstep2)
    end if

    ! Compute the quadratic coefficient in a Taylor expansion of the energy, E
    !
    !    E = E0 + dE*x + c*x**2
    ! => c = (E - E0 - dE*x)/x**2
    ! => min. at -dE/(2c)
    !
    !    dE/dx = dE + 2*c*x

    call zaxpy(num_pw*num_states,cmplx(opt_step,0.0_dp,dp),direction,1,approx_state,1)

    call orthonormalise(num_pw,num_states,approx_state)

    ! Apply the H to this state
    call apply_H(num_pw,num_states,approx_state,H_kinetic,H_local,gradient)

    ! Compute the new energy estimate
    energy = 0.0_dp 
    do nb=1,size(approx_state,2)
      eigenvalue(nb) = real(zdotc(num_pw,approx_state(:,nb),1,gradient(:,nb),1),dp)
      energy = energy + eigenvalue(nb)
    end do

    ! This ought to be the best, but check...
    if(energy>best_energy) then

      if(abs(best_step-epsilon(1.0_dp))>0.0_dp) then

        call zaxpy(num_pw*num_states,cmplx(best_step,0.0_dp,dp),direction,1,gradient,1)

        call orthonormalise(num_pw,num_states,approx_state)

        ! Apply the H to this state
        call apply_H(num_pw,num_states,approx_state,H_kinetic,H_local,gradient)

        ! Compute the new energy estimate
        energy = 0.0_dp 
        do nb=1,size(approx_state,2)
          eigenvalue(nb) = real(zdotc(num_pw,approx_state(:,nb),1,gradient(:,nb),1),dp)
          energy = energy + eigenvalue(nb)
        end do
      
      else
        write(*,*) 'Oh dear:',best_step
        stop 'Problem with line search'
      end if
    end if

!    write(*,'(3f25.15,a)') opt_step,step,energy,' <- test2'

    ! We'll use this step as the basis of our trial step next time
!    trial_step = 2*opt_step

    deallocate(tmp_state,stat=status)
    if(status/=0) stop 'Error deallocating RAM from tmp_state in line_search'

    return

  end subroutine line_search

  subroutine output_results(num_pw,num_states,H_local,wvfn)
    !-------------------------------------------------!
    ! This subroutine writes the potential and the    !
    ! wavefunction to two files (pot.dat and wvfn.dat !
    ! respectively).                                  !
    !-------------------------------------------------!
    use fftw3
    
    implicit none

    ! Define what we mean by double-precision
    integer, parameter                           :: dp=selected_real_kind(15,300)

    integer,                                        intent(in)    :: num_pw
    integer,                                        intent(in)    :: num_states
    real(kind=dp), dimension(num_pw),               intent(in)    :: H_local
    complex(kind=dp), dimension(num_pw,num_states), intent(in)    :: wvfn

    integer                                                       :: np,nb,status
    character(len=255)                                            :: filename

    complex(kind=dp), dimension(:),                allocatable    :: realspace_wvfn
    complex(kind=dp), dimension(:),                allocatable    :: tmp_wvfn

    type(C_PTR) :: plan                       ! defined in fftw3

    ! First write the local potential
    open(unit=10,file='pot.dat',form='formatted')

    do np=1,num_pw
      write(10,*) real(np-1,kind=dp)/real(num_pw,kind=dp),H_local(np)
    end do

    close(10)

    ! Now FFT and write the eigenstates
    allocate(realspace_wvfn(num_pw),stat=status)
    if(status/=0) stop 'Error allocating realspace in output_results'

    allocate(tmp_wvfn(num_pw),stat=status)
    if(status/=0) stop 'Error allocating tmp_wvfn in output_results'

    ! First create a plan. We want a forward transform and we want to estimate
    ! (rather than measure) the most efficient way to perform the transform.
    plan = fftw_plan_dft_1d(num_pw,tmp_wvfn,realspace_wvfn,FFTW_FORWARD,FFTW_ESTIMATE)

    do nb=1,num_states
      write(filename,*) nb
      filename='wvfn_'//trim(adjustl(filename))//'.dat'
      open(unit=10,file=filename,form='formatted')

      tmp_wvfn = wvfn(:,nb)
      
      ! Compute the FFT of in using the current plan, store in out.
      call fftw_execute_dft(plan,tmp_wvfn,realspace_wvfn)

      do np=1,num_pw
        write(10,*) real(np-1,kind=dp)/real(num_pw,kind=dp),real(realspace_wvfn(np),dp)
      end do

      close(10)
    end do

    call fftw_destroy_plan(plan)
    deallocate(realspace_wvfn,stat=status)
    if(status/=0) stop 'Error deallocating realspace in output_results'

    return

  end subroutine output_results
