import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# defing functions for 3rd part of worksheet

#defing exponential function
def f(x):
    return np.exp(x)

#defining the numerical second derivative
def df_2n(x,h):
    return (f(x+h)+f(x-h) - 2*f(x))/(h**2)

#defining x values
X = np.linspace(1,10,10)

# defining array of h values
H = np.array([1e-6,1e-5,1e-4,1e-3,1e-2,1e-1])

df_array = [] #array for creating the dataframe for shoing the table
er_array = [] #array for storing the error values
rel_array = [] #array for storing the relative error values

for x in X:
    df_exact = f(x)
    val_array = [df_exact] #values array with first element being the exact second derivative
    error_array = [] #error array for storing the error values for fixed x and varying h
    rel_error_array = [] #relative error array for storing the relative error values for fixed x and varying h

    for h in H:
        df_numerical = df_2n(x,h)
        val_array.append(df_numerical) #appending the numerical second derivative
        error_array.append(abs((df_numerical-df_exact))) #getting absolute error
        rel_error_array.append(abs((df_numerical-df_exact)/df_exact)) #getting relative error
        
    df_array.append(val_array)
    er_array.append(error_array)
    rel_array.append(rel_error_array)

index_val = ['x='+str(x) for x in X]
column_val = ['Exact','h=1e-6','h=1e-5','h=1e-4','h=1e-3','h=1e-2','h=1e-1']

df = pd.DataFrame(df_array, index=index_val, columns=column_val)
df_er = pd.DataFrame(er_array, index=index_val, columns=column_val[1:])
df_rel_er = pd.DataFrame(rel_array, index=index_val, columns=column_val[1:])

#saving the dataframe as a csv file
df.to_csv('df_table.csv') #second derivative table
df_er.to_csv('df_er_table.csv') #absolute error table
df_rel_er.to_csv('df_rel_er_table.csv') #relative error table

#printing the dataframe for second derivative
print("Second derivtive table\n", df)

# plotting error and relative error vs h
fig, ax = plt.subplots(1,2,figsize=(20,6), dpi=100)
for i in range(len(X)):
    ax[0].plot(H,np.log10(er_array[i]),label=index_val[i], marker='o', lw=1.2)
    ax[1].plot(H,np.log10(rel_array[i]),label=index_val[i])


ax[0].legend()
ax[1].legend()
ax[0].set_xlabel("Step Size $(h)$", fontsize=15);
ax[1].set_xlabel("Step Size $(h)$", fontsize=15);
ax[0].set_ylabel("$log_{10}(Absolute Error)$", fontsize=15);
ax[1].set_ylabel("$log_{10}(Relative Error)$", fontsize=15);
ax[0].set_title("Plot of $Absolute Error$ vs h")
ax[1].set_title("Plot of $Relative Error$ vs h");
#plt.savefig('error_vs_h.jpg', bbox_inches='tight', dpi=200)


#showing image of er_array and rel_array for dereming optimal h
fig, ax = plt.subplots(1,2,figsize=(16,6), dpi=100)
im0 = ax[0].imshow(np.log10(np.array(er_array)),cmap='ocean',origin='lower',
            extent=[1e-6,1e-1,1,10], aspect='auto')

im1 = ax[1].imshow(np.log10(np.array(rel_array)),cmap='ocean',origin='lower',
            extent=[1e-6,1e-1,1,10], aspect='auto')
plt.colorbar(im0, ax=ax[0]);
plt.colorbar(im1, ax=ax[1]);

#formatting the output
ax[0].set_title("Plot of log($Absolute~Error$) vs h for varying x")
ax[1].set_title("Plot of log($Relative~Error$) vs h for varying x");
ax[0].set_ylabel('x',fontsize=15)
ax[1].set_xlabel('h',fontsize=15)
ax[0].set_xlabel('h',fontsize=15)
ax[1].set_ylabel('x',fontsize=15)

#plt.savefig('error_vs_h_image.jpg', bbox_inches='tight', dpi=200)
plt.show()