import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
import progressbar as pb
from tqdm import tqdm

def get_iter(c, max_steps, thresh=2):
    """
    Calculate the number of iterations of z->z^2+c needed to escape.

    Iteratively calculates
        z = z^2 + c
    until either:
        |z| > thresh, or
        the number of iterations exceeds max_steps
    with an initial value of z = c.

    Parameters
    ----------
    c : complex
        Initial value of iteration and additive constant of iteration
    max_steps : positive int
        Maximum number of iterations to complete.
    thresh : positive float, optional
        Threshold distance from the origin, beyond which the iteration is said
        to have escaped. Sensibly should be greater than 2. Default is 2.
    """
    #assert type(c) is complex, "c must be complex"
    #assert max_steps > 0, "max_steps must be positive"
    #assert thresh > 0, "thresh must be positive"

    # initial values
    x0 = c.real
    y0 = c.imag

    # current values
    x = 0
    y = 0

    # squared values
    x2 = 0
    y2 = 0

    i = 1   # number of iterations

    thresh2 = thresh*thresh

    while i < max_steps and x2 + y2 < thresh2:
        # This calculates z^2 + c with less multiplications than doing z*z+c.
        # It uses the identity z^2 = (x + yi)^2 = x^2 - y^2 + 2xyi
        y = (x+x)*y + y0
        x = x2 - y2 + x0
        x2 = x*x
        y2 = y*y
        i += 1

    return i

def get_iter_area(xs, ys, max_steps, cume_area, bar, thresh=2):
    """
    Calculates the escape iterations in the domain defined by xs and ys.

    Makes use of the fact that the Mandelbrot set is simply connected. This
    means that any closed boundary along which all points are in the Mandelbrot
    set, all of the points within that boundary are also in the set. This can
    be extended to the escape times, so long as the boundary you draw isn't
    entirely outside of the escape threshold.

    This function applies this property to recursively calculate the escape
    times. The domain is split into four quadrants, and the escape times are
    calculated along the edges of each quadrant. Then for each quadrant:
        If all values on the edge are equal, fill the quadrant with that value.
        Otherwise, split the quadrant again and repeat.
    
    If the width of a quadrant is 2 or less in either dimension, the values
    are all directly calculated.

    Parameters
    ----------
    xs : array of floats
        The x values of the domain.
    ys : array of floats
        The y values of the domain.
    max_steps : positive int
        Maximum number of iterations to complete.
    cume_area : int
        Cumulative area calculated so far. Used to print a progress bar.
    bar : ProgressBar
        ProgressBar instance to display a progress bar.
    """
    # Split domain into four rectangles and calculate n_iter along perimeter.
    # If all values along the perimeter are equal, all points inside the
    # rectangle also have that value.
    res = (xs.shape[0], ys.shape[0])
    
    # if area is < 100 pixels, don't split and just calculate values straight
    ret_arr = np.zeros(res)
    if res[0] <= 2 or res[1] <= 2:
        for xi, x in enumerate(xs):
            for yi, y in enumerate(ys):
                ret_arr[xi, yi] = get_iter(x+1j*y, thresh=thresh, max_steps=max_steps)
        bar.update(cume_area)
        cume_area += res[0]*res[1]
        return cume_area, ret_arr

    sub_res = [ # resolutions of rectangles
        (res[0]//2, res[1]//2),
        (res[0] - res[0]//2, res[1]//2),
        (res[0]//2, res[1] - res[1]//2),
        (res[0] - res[0]//2, res[1] - res[1]//2)
    ]
    
    x_subslices = [
        slice(None, sub_res[1][0]),
        slice(sub_res[1][0], None)
    ]
    y_subslices = [
        slice(None, sub_res[2][1]),
        slice(sub_res[2][1], None)
    ]
    
    for i in range(2):
        for j in range(2):
            sub_x = xs[x_subslices[i]]
            sub_y = ys[y_subslices[j]]
            #print(f'x: {sub_x[0]:0.10f}\t\t{sub_x[-1]:0.10f}\t\ty: {sub_y[0]:0.10f}\t\t{sub_y[-1]:0.10f}', end='\r')
            # for each rectangle, calculate n_iter along perimeter
            sub_arr = ret_arr[x_subslices[i], y_subslices[j]]
            #print(sub_arr.shape)
            #print((sub_x.shape[0], sub_y.shape[0]))
            # top and bottom edges
            for xi, x in enumerate(sub_x):
                sub_arr[xi, 0] = get_iter(x + 1j*sub_y[0], thresh=thresh, max_steps=max_steps)
                sub_arr[xi, -1] = get_iter(x + 1j*sub_y[-1], thresh=thresh, max_steps=max_steps)
                
            # left and right edges
            for yi, y in enumerate(sub_y):
                sub_arr[0, yi] = get_iter(sub_x[0] + 1j*y, thresh=thresh, max_steps=max_steps)
                sub_arr[-1, yi] = get_iter(sub_x[-1] + 1j*y, thresh=thresh, max_steps=max_steps) 
                
            # check if all calculated values are equal
            if np.all(sub_arr[(sub_arr > 0)] == sub_arr[0, 0]):
                # fill array with perimeter value
                #print('fill')
                ret_arr[x_subslices[i], y_subslices[j]].fill(sub_arr[0, 0])
                bar.update(cume_area)
                cume_area += sub_x.shape[0] * sub_y.shape[0]
            else:
                # recursively send this sub_array to this function to fill it in
                cume_area, ret_arr[x_subslices[i], y_subslices[j]] = get_iter_area(sub_x, sub_y, max_steps, cume_area, bar, thresh=thresh)
                #print(i, j)
                
    # return sub arrays glued back together
    return cume_area, ret_arr
    
def plot_recursive(xs, ys, zoom=0, thresh=4, max_steps=25, total_area=None):
    with pb.ProgressBar(max_value=total_area) as bar:
        area, arr = get_iter_area(xs, ys, max_steps, 0, bar, thresh=thresh)

    return arr

def doit_rec(centre, zoom, max_steps, cmap, n, n_jobs, ax):
    ax.clear()
    res=(3040//n, 1440//n)
    extent = (19 / 10**zoom, 9 / 10**zoom)
    domain = ((centre[0]-extent[0], centre[0]+extent[0]), (centre[1]-extent[1], centre[1]+extent[1]))
    xs = np.linspace(domain[0][0], domain[0][1], res[0])
    ys = np.linspace(domain[1][0], domain[1][1], res[1])
    sub_xs = np.array_split(xs, n_jobs)
    arrs = Parallel(n_jobs=n_jobs)(delayed(plot_recursive)(sub_x, ys, max_steps=max_steps, zoom=zoom, total_area=res[0]*res[1]) for sub_x in sub_xs)
    arrs_joined = np.concatenate(arrs, axis=0)
    ax.imshow(arrs_joined.transpose(), origin='lower', interpolation='none', cmap=cmap)
    ax.set_axis_off()
    return xs, ys, arrs_joined
