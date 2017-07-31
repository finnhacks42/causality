        
bn = DiscreteBN()
bn.add_var('Z',2,prob_table=[.1,.9])
bn.add_var('X',2,['Z'])
bn.add_var('Y',2,['X'])  
for c in bn._conditionals:
    print "c",c
print bn.joint    

#############################################################
model = LinearGaussianBN()
model.add_var('X',weights = [0])
model.add_var('Y',['X'], weights = [0,"w_YX"])
display((model.mu,model.cov))


# check
n = 10000000
V_x = 2.0
w_YX = 2.3
V_y = 3.0
observed =('Y',3,.001)

conditional = model.observe([observed[0]],[observed[1]])
display((YgivenX.mu,YgivenX.cov))

# by direct simulation
X = np.random.normal(0,np.sqrt(V_x),size = n)
Y = w_YX*X + np.random.normal(0,np.sqrt(V_y),size = n)
df = pd.DataFrame(columns = ['X','Y'],data = np.vstack((X,Y)).T)
samples = df.loc[(df[observed[0]] > observed[1]-observed[2]) & (df[observed[0]] < observed[1]+observed[2]), ['X','Y']]
print(samples['X'].values.mean(),samples['X'].values.var(ddof=0),"based on {0} samples".format(len(samples)))

# by substitution 2
mu,cov = conditional.set_params({'X':([0],V_x),'Y':([0,w_YX],V_y)}).parameterized_mean_cov()
display((mu,cov))

##################################################################
model = LinearGaussianBN()
model.add_var('Z',weights=[0])
model.add_var('X',['Z'],weights = [0,"w_XZ"])
model.add_var('Y',['Z','X'],weights = [0,"w_YZ","w_YX"])


# do some basic checking with samples
n = 1000000
V_z = 2
V_x =  2
w_xz = .8
V_y = .1
w_yz = .8
w_yx = 2.2
x = (3,.01)

Z = np.random.normal(0,np.sqrt(V_z),size = n)
X = w_xz*Z + np.random.normal(0,np.sqrt(V_x),size = n)
Y = w_yz*Z + w_yx*X + np.random.normal(0,np.sqrt(V_y),size = n)
df = pd.DataFrame(columns = ['Z','X','Y'],data = np.vstack((Z,X,Y)).T)
samples = df.loc[(df["X"] > x[0]-x[1]) & (df["X"] < x[0]+x[1]), ['Z','X','Y']]
print(samples['Y'].values.mean(),samples['Y'].values.var(ddof=0),"based on {0} samples".format(len(samples)))
#plot = sns.jointplot(x="X", y="Y", data=df.head(1000), kind="reg")

YgivenX = model.observe(['X'],[x[0]]).marginal('Y')
display((YgivenX.mu,YgivenX.cov))

mu,cov = YgivenX.set_params({'Z':([0],V_z),'X':([0,w_xz],V_x),'Y':([0,w_yz,w_yx],V_y)}).parameterized_mean_cov()
display((mu,cov))

####################################################################

V_z = .5
w_xz = 2.5
V_x =  5

model = LinearGaussianBN()
model.add_var('Z')
model.add_var('X',['Z'])
conditional = model.observe(['X'],['x'])
mu,cov = conditional.set_params({'Z':([0],"V_z"),'X':([0,"w_xz"],"V_x")}).parameterized_mean_cov()
display((mu,cov))

model.set_var_params("Z",[0],V_z)
model.set_var_params("X",[0,w_xz],V_x)
data = model.sample(10000)
df = pd.DataFrame(columns=["Z","X"],data = data)
plot = sns.jointplot(x="X", y="Z", data=df, kind="reg")
dx = plot.ax_joint.get_lines()[0].get_xdata()[-1] - plot.ax_joint.get_lines()[0].get_xdata()[0]
dy = plot.ax_joint.get_lines()[0].get_ydata()[-1] - plot.ax_joint.get_lines()[0].get_ydata()[0]


display(mu.subs([("V_z",V_z),("w_xz",w_xz),("V_x",V_x)]))
print("slope",dy/dx)

