import numpy as np
import matplotlib.pyplot as plt

# define a quadratic function

def f(x,a,b):
    return a + b*np.power(x,2)

# defining the exact derivative
def df(x,a,b):
    return b*2*x

#defing forward numerical derivative
def dfn(x,a,b,h):
    return (f(x+h,a,b)-f(x,a,b))/h

#defing central difference numerical derivative
def dfc(x,a,b,h):
    return (f(x+h,a,b)-f(x-h,a,b))/(2*h)

# selecting a and b
a = 1
b = 1

#defing array of x values
X = np.linspace(1,5,5)

# defining array of h values
H = np.linspace(1e-6,1,100)

#initilaizing the figure
fig,ax = plt.subplots(1,2,figsize=(15,5), dpi=100)

#calulting the require results for differnt values of x
for x in X:
    #calulating array of exact derivative for given x
    df_exact = df(x,a,b)

    error_array = np.zeros(len(H))
    
    i=0
    for h in H:
        dfn_exact = dfn(x,a,b,h) #Numerical Derivative
        #calulating (relative error) of forward numerical derivatives
        dferr_exact = abs((dfn_exact-df_exact)/df_exact)
        
        #storing the error for ploting
        error_array[i] = dferr_exact
        i=i+1
    log_error_array = np.log10(error_array) #calulating log(Relative error)

    #plotting error vs step Size(h)
    ax[0].plot(H, error_array,label='x='+str(x), lw=1.2)
    ax[1].plot(H, log_error_array,label='x='+str(x), lw=1.2)

# formatting the output
ax[0].legend()
ax[1].legend()
ax[0].set_xlabel("Step Size $(h)$", fontsize=15);
ax[1].set_xlabel("Step Size $(h)$", fontsize=15);
ax[0].set_ylabel("Relative Error $(\epsilon)$", fontsize=15);
ax[1].set_ylabel("$log_{10}(\epsilon)$", fontsize=15);
ax[0].set_title("Plot of $Error$ vs h")
ax[1].set_title("Plot of $log(Error)$ vs h");

#plt.savefig('error_vs_h_a.jpg', bbox_inches='tight', dpi=200)
plt.show()