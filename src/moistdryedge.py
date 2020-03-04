"""@package moistdryedge
Documentation for module moistdryedge.

Class Edge contains (x,y) coordinates of the points found on the boundary between
the dry and the moist region, defined as the location of maximum gradient of a
reference variable (PW, CRH, etc.).
"""

import numpy as np
from scipy.ndimage.filters import gaussian_filter,convolve

class Edge():

    """Class Edge contains (x,y) coordinates of the points found on the boundary between
    the dry and the moist region, defined as the location of maximum gradient of a
    reference variable (PW, CRH, etc.)."""

    def __init__(self,mask_mode='thres',sigma_smooth=8,coef_mask=0.6,thres=0.5,\
    epsilon=1e-3):

        """Arguments:
        - mask_mode: method to select a rough zone around the edge, either using
        a fixed threshold for the norm of the gradient ('thres') or a fixed fraction 
        of the maximum horizontal gradient ('percmax'). Default: 'thres'.
        - coef_mask: fraction of maximum gradient for mask_mode='percmax'
        - thres: threshold value for mask_mode='thres'. Default: 0.5.
        - epsilon: condition chosen to pick grad == 0 when computing the local 
        maxima of the gradient parallel to the gradient. Default: 1e-3.
        - sigma_smooth: number of pixels for width of gaussian filter (see 
        scipy.ndimage.filters.gaussian_filter)
        - """

        self.mask_mode = mask_mode
        self.coef_mask = coef_mask
        self.thres = thres
        self.sigma_smooth = sigma_smooth
        self.epsilon = epsilon

    def deriv(self,arr,axis=0):

        """Compute derivative of numpy array in argument.
        
        Arguments:
        - arr: numpy array
        - axis: direction
        
        Returns:
        - numpy array of same shape as arr"""
        
        stencil = [1/12,-2/3,0,2/3,-1/12]
        f = lambda x: convolve(x,stencil,mode='wrap')
        
        return np.apply_along_axis(f,axis,arr)

    def grad2Dnorm(self,arr):
        
        """Compute norm of the gradient of numpy array in argument.
        
        Arguments:
        - arr: 2D numpy array
        
        Returns:
        - numpy array of same shape as arr"""

        d_x = self.deriv(arr,axis=0)
        d_y = self.deriv(arr,axis=1)
        
        return np.sqrt(d_x**2+d_y**2)

    def compute(self,arr,out=False):

        """Main algorithm for edge detection. Assumes 2D periodic BCs.
        
        Arguments:
        - arr: 2D numpy array"""

        # smooth field
        arr_smooth = gaussian_filter(arr,self.sigma_smooth,mode='wrap')
        # grad(PW)
        arr_gradi = self.deriv(arr_smooth,axis=0)
        arr_gradj = self.deriv(arr_smooth,axis=1)
        # norm(grad(PW))
        arr_gradnorm = self.grad2Dnorm(arr_smooth)

        # exit if flat field
        if np.all(arr_gradnorm == 0):
            snapshots = [arr_smooth,arr_gradnorm,np.array([]),np.array([])]
            return np.array([]),np.array([]),snapshots
        
        # smoothed area around the border (mask)
        if self.mask_mode == 'thres':
            arr_area = arr_gradnorm > self.thres
        elif self.mask_mode == 'percmax':
            arr_area = arr_gradnorm > np.max(arr_gradnorm)*self.coef_mask
        # grad(norm(grad(PW)))
        arr_gradnorm_gradi = self.deriv(arr_gradnorm,axis=0)
        arr_gradnorm_gradj = self.deriv(arr_gradnorm,axis=1)
        # unit vector n in the direction of the gradient
        e_gradi = arr_gradi/arr_gradnorm
        e_gradj = arr_gradj/arr_gradnorm
        # grad(norm(grad(PW))).n
        arr_dotprod = (arr_gradnorm_gradi*e_gradi+arr_gradnorm_gradj*e_gradj)/arr_gradnorm
        # local maxima
        arr_maxima = np.absolute(arr_dotprod)<self.epsilon
        # local maxima at boundary
        arr_border = np.logical_and(arr_maxima,arr_area)
        # indices of found maxima
        self.border_i,self.border_j = np.where(arr_border)

        snapshots = [arr_smooth,arr_gradnorm,arr_area,arr_maxima]
    
        if out:
            return snapshots

    def getValuesOnEdge(self,arr):

        """Extracts values of arr on moist-dry boundary.
        
        Arguments:
        - arr: 2D numpy array"""

        values = []
        for k in range(len(self.border_i)):
            i = self.border_i[k]
            j = self.border_j[k]
            values.append(arr[i,j])
            
        return values

    def computeStatOnEdge(self,arr,varid,fname='mean',out=False):

        """Applies function f to values on moist-dry boundary and stores the 
        result as attribute in self.${varid}_${fname}
        
        Arguments:
        - arr: 2D numpy array
        - varid: str
        - fname: str, assumes it is a unary numpy method"""

        values = self.getValuesOnEdge(arr)
        result = getattr(np,fname)(values)
        setattr(self,"%s_%s"%(varid,fname),result)

        if out:
            return result

    def computeGradNormStatOnEdge(self,arr,varid,fname='mean',out=False):

        """Applies function f to norm(grad(arr)) and stores the result as
        a new attribute self.${varid}_gradnorm_mean

        Arguments:
        - arr: 2D numpy array
        - varid: str
        - fname: str, assumes it is a unary numpy method"""

        # calculate gradnorm of smoothed field
        arr_smooth = gaussian_filter(arr,self.sigma_smooth,mode='wrap')
        gradnorm = self.grad2Dnorm(arr_smooth)

        # calculate stat on edge
        result = self.computeStatOnEdge(gradnorm,varid+'_gradnorm',fname,out)

        if out:
            return result

    def computeFractionArea(self,arr,varid,out=False):

        """Creates a timeseries for fraction area of the dry region based on a 
        threshold stored as attribute. 

        Arguments:
        - arr: 2D numpy array of variable on which the threshold is applied
        - varid: str for attribute ${varid}_mean used as threshold"""

        thres = getattr(self,"%s_mean"%varid)
        fracarea = np.sum(arr<thres)/arr.size
        setattr(self,"fracarea_%s"%varid,fracarea)

        if out:
            return fracarea


class EdgeOverTime(Edge):

    def __init__(self,mask_mode='thres',sigma_smooth=8,coef_mask=0.6,thres=0.5,\
    epsilon=1e-3):

        """Constructor of class EdgeOverTime.
        See Edge.__init__ for details."""

        # super().__init__(mask_mode=mask_mode,sigma_smooth=sigma_smooth,\
        #     coef_mask=coef_mask,thres=thres,\
        #     epsilon=epsilon)
        super().__init__(mask_mode=mask_mode,sigma_smooth=sigma_smooth,\
            coef_mask=coef_mask,thres=thres,epsilon=epsilon)
        
    def compute(self,arr):

        """Compute coordinates of moist-dry boundary at all times.
        
        Arguments:
        - arr: 3D numpy array with time in 1st dimension"""

        self.Nt = arr.shape[0]

        self.edges = []
        for i_t in range(self.Nt):

            # initialize
            e = Edge(mask_mode=self.mask_mode,sigma_smooth=self.sigma_smooth,\
            coef_mask=self.coef_mask,thres=self.thres,epsilon=self.epsilon)
            # compute
            e.compute(arr=arr[i_t])
            # store
            self.edges.append(e)

    def computeStatOnEdge(self, arr, varid, fname='mean'):

        """Applies function f to values on moist-dry boundary and stores the 
        result as attribute in self.${varid}_${fname}
        
        Arguments:
        - arr: 2D numpy array
        - varid: str
        - fname: str, assumes it is a unary numpy method"""

        # initalize object attribute and temp array
        attrname = "%s_%s"%(varid,fname)
        vals = np.nan*np.zeros((self.Nt,))
        
        for i_t in range(self.Nt):

            vals[i_t] = self.edges[i_t].computeStatOnEdge(arr[i_t], varid, fname=fname,\
                out=True)

        # store 
        setattr(self,attrname,vals)

    def computeGradNormStatOnEdge(self, arr, varid, fname='mean'):

        """Applies function f to norm(grad(arr)) and stores the result as
        a new attribute self.${varid}_gradnorm_mean

        Arguments:
        - arr: 2D numpy array
        - varid: str
        - fname: str, assumes it is a unary numpy method"""

        attrname = "%s_gradnorm_%s"%(varid,fname)
        vals = np.nan*np.zeros((self.Nt,))
        
        for i_t in range(self.Nt):

            vals[i_t] = self.edges[i_t].computeGradNormStatOnEdge(arr[i_t], varid, fname=fname,\
                out=True)
            
        # stores timeseries
        setattr(self,attrname,vals)

    def computeFractionArea(self,arr,varid):

        """Creates a timeseries for fraction area of the dry region based on a 
        threshold stored as attribute. 

        Arguments:
        - arr: 2D numpy array of variable on which the threshold is applied
        - varid: str for attribute ${varid}_mean used as threshold"""

        attrname = "fracarea_%s"%varid
        vals = np.nan*np.zeros((self.Nt,))
        
        for i_t in range(self.Nt):

            vals[i_t] = self.edges[i_t].computeFractionArea(arr[i_t],varid,out=True)

        setattr(self,attrname,vals)

        


        

