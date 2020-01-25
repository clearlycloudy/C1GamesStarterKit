import gamelib

import numpy as np
import torch
from scipy import fftpack
from scipy import signal
from scipy import misc
# from sklearn import mixture

def div(a,b):
    if b != 0:
        value = float(a) / float(b)
    else:
        value = float('Inf')
    return value

def create_line(start, end):
    # inclusive
        
    dx = end[0]-start[0]
    dy = end[1]-start[1]

    if dx==0 and dy==0:
        return [start]
    elif dx == 0:
        extension,step = (1,1) if dy > 0 else (-1,-1)
        loc = list(map( lambda x: [start[0], start[1]+x], range(0,dy+extension,step) ))
        return loc
    elif dy == 0:
        extension,step = (1,1) if dx > 0 else (-1,-1)
        loc = list(map( lambda x: [start[0]+x, start[1]], range(0,dx+extension,step) ))
        return loc
    else:
        slope = float(dy)/float(dx)
        loc = []
        prev = start[1]
        extension,step = (1,1) if dx > 0 else (-1,-1)
        for i in range(0,dx+extension,step):
            x = start[0] + i
            y = int(start[1] + float(i)*slope)

            if prev is not None and abs(y-prev)>1:
                for j in range(abs(y-prev)):
                    sign = 1. if y-prev > 0 else -1.
                    yy = int(float(prev)+float(j)*sign)
                    # gamelib.debug_write('line: {}, {}, {}'.format(x,yy, sign))
                    loc.append([x,yy])
            else:
                # gamelib.debug_write('line: {}, {}'.format(x,y))
                loc.append([x,y])
            prev = y

        return loc
    
# # from scipy documentation:
# def gaussian(self, height, center_x, center_y, width_x, width_y, rotation):
#         """Returns a gaussian function with the given parameters"""
#         width_x = float(width_x)
#         width_y = float(width_y)

#         rotation = np.deg2rad(rotation)
#         center_x = center_x * np.cos(rotation) - center_y * np.sin(rotation)
#         center_y = center_x * np.sin(rotation) + center_y * np.cos(rotation)

#         def rotgauss(x,y):
#             xp = x * np.cos(rotation) - y * np.sin(rotation)
#             yp = x * np.sin(rotation) + y * np.cos(rotation)
#             g = height*np.exp(
#                 -(((center_x-xp)/width_x)**2+
#                   ((center_y-yp)/width_y)**2)/2.)
#             return g
#         return rotgauss

# def moments(self, data):
#     """Returns (height, x, y, width_x, width_y)
#     the gaussian parameters of a 2D distribution by calculating its
#     moments """
#     total = data.sum()
#     X, Y = np.indices(data.shape)
#     x = (X*data).sum()/total
#     y = (Y*data).sum()/total
#     col = data[:, int(y)]
#     width_x = np.sqrt(abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
#     row = data[int(x), :]
#     width_y = np.sqrt(abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
#     height = data.max()
#     return height, x, y, width_x, width_y, 0.0

    
# def fitgaussian(self, data):
#     """Returns (height, x, y, width_x, width_y)
#     the gaussian parameters of a 2D distribution found by a fit"""
#     params = self.moments(data)
#     errorfunction = lambda p: np.ravel(self.gaussian(*p)(*np.indices(data.shape)) - data)
#     p, success = scipy.optimize.leastsq(errorfunction, params)
#     return p

def create_gaussian_distr(map2d):

    samples = []
    for x in range(map2d.shape[0]):
        for y in range(map2d.shape[1]):
            for i in range(int(map2d[x,y])):
                samples.append([x,y])

    if len(samples)>3:
        samp = np.zeros((len(samples),2))
        for i in range(len(samples)):
            samp[i,:] = [ samples[i][0], samples[i][1] ]

        #calc mean and covariance from data
        n = samp.shape[0]
        cov = np.cov(samp, y=None, rowvar=False, bias=False, ddof=None, fweights=None, aweights=None)
        noise = np.random.normal() * 0.2
        cov = cov + noise
        means = np.sum(samp, axis=0) / n

        # gamelib.debug_write(means)
        # gamelib.debug_write(cov)

        # #1 component in mixture for now
        # commented out since sklearn is not supported in online mode
        # gmm = mixture.GaussianMixture(n_components=1, covariance_type='full').fit(samp)
                    
        # gamelib.debug_write("samp length: {}".format(len(samples)))
        # gamelib.debug_write("gmm means:")
        # gamelib.debug_write(gmm.means_)
        # gamelib.debug_write("gmm covs:")
        # gamelib.debug_write(gmm.covariances_)

        # X = gmm.sample(n_samples=1)
        # return gmm
        return means, cov
    else:
        return None

def generate_sample_from_gaussian(gmm, samples=1):
    output = []
    for j in range(samples):
        # just take the first component of the gaussian fixture for now
        # X = np.random.multivariate_normal(gmm.means_[0], gmm.covariances_[0])
        X = np.random.multivariate_normal(gmm[0], gmm[1])
        output.append(X)
    return output
