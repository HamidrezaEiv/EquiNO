# space_integration_mod.py
# module containing required information for the spatial integration
    
import numpy as np

def shp_quad(npe,coord):
    # function to evaluate the shape functions and the shape function 
    # derivatives w.r.t. local coordinates (xi,eta) at the integration points
    #
    # Implementation for 8-noded quadrilateral element
    #
    # INPUT
    #  npe - number of nodes per element
    #  coord - integration point coordinates in local coordinates (xi,eta)
    #
    # OUTPUT
    #  shp - shape functions and shape function derivatives evaluated at 
    #        integration points (npe x 3 x ngp)
    
    # check number of nodes per element
    if npe != 8:
        raise ValueError("Current implementation is valid only for 8-noded quadrilateral elements")
        
    # initialize array
    # shp(:,1,:) - shape functions evaluated at integration point
    # shp(:,2,:) - shape function derivative w.r.t. xi evaluated at integration point
    # shp(:,3,:) - shape function derivative w.r.t. eta evaluated at integration point
    shp = np.zeros((npe,3,coord.shape[0]))
    
    for ip in range(0,coord.shape[0]):

        # get current integration point coordinates
        xi = coord[ip,0]
        eta = coord[ip,1]        

        # evaluate shape functions
        shp[0,0,ip] = 0.25*(1-xi)*(1-eta)*(-xi-eta-1)
        shp[1,0,ip] = 0.25*(1+xi)*(1-eta)*(xi-eta-1)
        shp[2,0,ip] = 0.25*(1+xi)*(1+eta)*(xi+eta-1)
        shp[3,0,ip] = 0.25*(1-xi)*(1+eta)*(-xi+eta-1)
        shp[4,0,ip] = 0.5*(1-xi**2)*(1-eta)
        shp[5,0,ip] = 0.5*(1-eta**2)*(1+xi)
        shp[6,0,ip] = 0.5*(1-xi**2)*(1+eta)
        shp[7,0,ip] = 0.5*(1-eta**2)*(1-xi)
                              
        # evaluate shape function derivatives w.r.t. xi
        shp[0,1,ip] = -0.25*(1-eta)*(-2*xi-eta)
        shp[1,1,ip] = 0.25*(1-eta)*(2*xi-eta)
        shp[2,1,ip] = 0.25*(1+eta)*(2*xi+eta)
        shp[3,1,ip] = -0.25*(1+eta)*(-2*xi+eta)
        shp[4,1,ip] = -(1-eta)*xi
        shp[5,1,ip] = 0.5*(1-eta**2)
        shp[6,1,ip] = -(1+eta)*xi
        shp[7,1,ip] = -0.5*(1-eta**2)
        
        # evaluate shape function derivatives w.r.t. eta
        shp[0,2,ip] = -0.25*(1-xi)*(-2*eta-xi)
        shp[1,2,ip] = -0.25*(1+xi)*(-2*eta+xi)
        shp[2,2,ip] = 0.25*(1+xi)*(2*eta+xi)
        shp[3,2,ip] = 0.25*(1-xi)*(2*eta-xi)
        shp[4,2,ip] = -0.5*(1-xi**2)
        shp[5,2,ip] = -(1+xi)*eta
        shp[6,2,ip] = 0.5*(1-xi**2)
        shp[7,2,ip] = -(1-xi)*eta
        
    return shp


def shp_tri(npe,coord):
    # function to evaluate the shape functions and the shape function 
    # derivatives w.r.t. local coordinates (xi,eta) at the integration points
    #
    # Implementation for 6-noded triangular element
    #
    # INPUT
    #  npe - number of nodes per element
    #  coord - integration point coordinates in local coordinates (xi,eta)
    #
    # OUTPUT
    #  shp - shape functions and shape function derivatives evaluated at 
    #        integration points (npe x 3 x ngp)
    
    # check number of nodes per element
    if npe != 6:
        raise ValueError("Current implementation is valid only for 6-noded triangular elements")
    
    # initialize array
    # shp(:,1,:) - shape functions evaluated at integration point
    # shp(:,2,:) - shape function derivative w.r.t. xi evaluated at integration point
    # shp(:,3,:) - shape function derivative w.r.t. eta evaluated at integration point
    shp = np.zeros((npe,3,coord.shape[0]))
    
    for ip in range(0,coord.shape[0]):

        # get current integration point coordinates
        xi = coord[ip,0]
        eta = coord[ip,1]        

        # evaluate shape functions
        shp[0,0,ip] = (1-xi-eta)*(2*(1-xi-eta)-1)
        shp[1,0,ip] = xi*(2*xi-1)
        shp[2,0,ip] = eta*(2*eta-1)
        shp[3,0,ip] = 4*xi*(1-xi-eta)
        shp[4,0,ip] = 4*xi*eta
        shp[5,0,ip] = 4*eta*(1-xi-eta)
                              
        # evaluate shape function derivatives w.r.t. xi
        shp[0,1,ip] = -4*(1-xi-eta)+1
        shp[1,1,ip] = 4*xi-1
        shp[2,1,ip] = 0.0
        shp[3,1,ip] = 4-8*xi-4*eta
        shp[4,1,ip] = 4*eta
        shp[5,1,ip] = -4*eta
        
        # evaluate shape function derivatives w.r.t. eta
        shp[0,2,ip] = -4*(1-xi-eta)+1
        shp[1,2,ip] = 0.0
        shp[2,2,ip] = 4*eta-1
        shp[3,2,ip] = -4*xi
        shp[4,2,ip] = 4*xi
        shp[5,2,ip] = 4-4*xi-8*eta

    return shp


def get_coord_weights_gauss(order):
    # function to determine the weighting factors for the spatial integration
    # - here Gauss-integration - and the shape functions evaluated at the 
    # integration point positions as well as the shape function derivatives
    # w.r.t. local coordinates (xi,eta).
    #
    # Implementation for 8-noded quadrilateral element
    #
    # INPUT
    #  order - integration order 
    #
    # OUTPUT
    #  weight - weighting factors of integration points (ngp x 1)
    #  coord - integration point coordinates in local coordinates (xi,eta)
    
    # check integration order
    if order != 4:
        raise ValueError("Current implementation is valid only for 8-noded quadrilateral elements with integration order 4")
        
    # number of integration points
    ngp = 9
    
    # integration point coordinates
    a = np.sqrt(3./5.)
    b = 0.
    coord = np.array([[-a, -a], [a, -a], [a, a], [-a, a], [b, -a], [a, b], [b, a], [-a, b], [b, b]])

    # integration point weights
    c = 5./9.
    d = 8./9.
    weight = np.zeros((ngp))
    weight[0:4] = c*c
    weight[4:8] = c*d
    weight[8] = d*d
    
    return weight, coord


def get_coord_weights_hammer(order):
    # function to determine the weighting factors for the spatial integration
    # - here Hammer-integration - and the shape functions evaluated at the 
    # integration point positions as well as the shape function derivatives
    # w.r.t. local coordinates (xi,eta).
    #
    # Implementation for 6-noded triangular element
    #
    # INPUT
    #  order - integration order 
    #
    # OUTPUT
    #  weight - weighting factors of integration points (ngp x 1)
    #  coord - integration point coordinates in local coordinates (xi,eta)
    
    # check integration order
    if order != 2:
        raise ValueError("Current implementation is valid only for 6-noded triangular elements with integration order 2")
    
    # number of integration points
    ngp = 3
    
    # coordinates of integration points
    a = 0.5
    b = 0.
    coord = np.array([[a, a], [b, a], [a, b]])
    
    # integration point weights
    c = 1./6.
    weight = np.array([c, c, c])
    
    return weight, coord


def space_int(elemType):
    # function to evaluate required information for the spatial integration
    #
    # INPUT
    #  elemType - element type (string)
    #
    # OUTPUT
    #  weight - weighting factors of integration points (ngp x 1)
    #  shp - shape functions and shape function derivatives evaluated at 
    #        integration points (npe x 3 x ngp)
    
    if elemType == 'tri6':
        
        # set integration order
        order = 2
        # set number of nodes per element
        npe = 6
        
        # evaluate weighting factors for spatial integration and integration
        # point coordinates
        [weight,coord] = get_coord_weights_hammer(order)
        
        # evaluate shape functions and shape function derivatives w.r.t.
        # local coordinates (xi,eta) at integration points
        shp = shp_tri(npe,coord)
        
    elif elemType == 'quad8':
        
        # set integration order
        order = 4
        # set number of nodes per element
        npe = 8
        
        # evaluate weighting factors for spatial integration and integration
        # point coordinates
        [weight,coord] = get_coord_weights_gauss(order)
        
        # evaluate shape functions and shape function derivatives w.r.t.
        # local coordinates (xi,eta) at integration points
        shp = shp_quad(npe,coord)
        
    else:
        raise ValueError("Requested element type is not implemented")
    
    return weight, shp


def inv_jac(x,shpGP):
    # function to compute the determinant of the Jacobian matrix and the 
    # derivatives of the shape functions w.r.t. global coordinates
    #
    # Implementation for two-dimensional elements
    #
    # INPUT
    #  x - global coordinates (x,y) of element nodes (npe x 2)
    #  shpGP - shape function derivatives associated with current integration 
    #          point (npe x 2)
    #
    # OUTPUT
    #  detj - determinant of Jacobian matrix 
    #  shpdx - derivatives of shape functions w.r.t. global coordinates to 
    #          perform the strain computation (npe x 2)
    
    # check dimensions of arguments
    if x.shape[1] != 2:
        raise ValueError("Wrong dimension of coordinate input")
    elif shpGP.shape[1] != 2:
        raise ValueError("Wrong dimension of shape function input")
    
    # compute Jacobian matrix dx/dxi (already transposed)
    jacobian = np.matmul(np.transpose(shpGP),x)
    
    # check for small numbers
    logMat = np.abs(jacobian) < np.finfo(float).eps
    jacobian[logMat] = 0.0
    
    # compute determinant of transposed Jacobian
    detj = np.linalg.det(jacobian)
    
    # compute transpose inverse of Jacobian matrix
    fac = 1.0/detj
    invJac = np.zeros((2,2))
    invJac[0,0] = fac*jacobian[1,1]
    invJac[0,1] = -fac*jacobian[0,1]
    invJac[1,0] = -fac*jacobian[1,0]
    invJac[1,1] = fac*jacobian[0,0]
    
    # compute global derivatives of shape functions
    shpdx = np.matmul(shpGP,np.transpose(invJac))
    
    return detj, shpdx