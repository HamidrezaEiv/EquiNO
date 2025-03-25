PROGRAM forpy_test
    USE forpy_mod
    IMPLICIT NONE

    INTEGER, PARAMETER                      :: NROWS = 1000
    INTEGER, PARAMETER                      :: NCOLS = 3
    REAL                                    :: X(NROWS, NCOLS)
    REAL                                    :: GE(NCOLS)
    REAL                                    :: GT(NCOLS)
    REAL                                    :: GC(NCOLS, NCOLS)
    INTEGER                                 :: ii, jj, ierror
    TYPE(list)                              :: paths
    TYPE(module_py)                         :: pred

    ! Open the file
    open(unit=NROWS, file=trim('../data/val_data.csv'), status='old', action='read')
    ! Read the data
    read(NROWS, *) ! Skip header
    DO ii = 1, NROWS
        read(NROWS, *) (X(ii, jj), jj=1, NCOLS)
    END DO
    ! Close the file
    close(NROWS)

    ierror = forpy_initialize()
    ierror = get_sys_path(paths)
    ierror = paths%append(".")
    ierror = import_py(pred, "predict")

    DO ii = 1, NROWS
        DO jj = 1, NCOLS
            GE(jj) = X(ii, jj)
        END DO
        CALL RVE(GE, GT, GC) ! 
        print *, 'Strain:', GE
        print *, 'Stress:', GT
        print *, 'Consistent tangent:', GC
    END DO

    CALL forpy_finalize

CONTAINS

    SUBROUTINE RVE(eps, stress, ct)
        real, dimension(NCOLS),     intent(in)      :: eps
        real, dimension(NCOLS),     intent(out)     :: stress
        real, dimension(NCOLS,NCOLS),   intent(out)     :: ct
        type(object)                            :: return_value
        type(tuple)                             :: args, return_value_tuple
        integer                                 :: i, j, k
        
        ierror = tuple_create(args, NCOLS)
        DO i = 1, NCOLS
        ierror = args%setitem(i-1, eps(i))
        END DO
        
        ierror = call_py(return_value, pred, "prediction", args)
        ierror = cast_nonstrict(return_value_tuple, return_value)

        DO i = 1, NCOLS
        ierror = return_value_tuple%getitem(stress(i), i-1)
        END DO

        DO i = 1, NCOLS
            DO j = 1, NCOLS
                k = i * NCOLS + j - 1   
                ierror = return_value_tuple%getitem(ct(i,j), k)
            END DO
        END DO
        
        CALL args%destroy
        CALL return_value%destroy
        CALL return_value_tuple%destroy

    END SUBROUTINE RVE

END PROGRAM