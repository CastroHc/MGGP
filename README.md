# Multi-Gene Genetic Programing (MGGP) toolbox


## Getting Started

### Prerequisites

To use this toolbox it is necessary the user to have installed the lybraries: numpy, scipy and deap

```
pip install deap
```

### Installing

To install the toolbox use: 

```
pip install mggp
```

## License

This project is licensed under the MIT License - see the [license.txt](license.txt) file for details

# Tutorial
## _mggpElement Class_

An `mggpElement` object is responsible to carry the attributes and functions used to create and evaluate individuals from a MGGP population. This class is able to build SISO and MISO models. Its default configuration creates an element object capable of building SISO models in which the variables are 'y1', 'u1' and (optional) 'e1'. The number '1' in the variable name indicates that it is a one-step lagged variable (y1 = y[k-1]). Also, the only function present in the primitive set is '_mul_' with arity equals 2, that is, it receives two arguments -- _mul_(x1,x2).

Three parameters should be set in an `mggpElement` object:

- maxDelay = corresponding to the maximum back-shift operator included in the primitive set. For example:
>maxDelay = 3 --> {q1, q2, q3} (default),
>maxDelay = 5 --> {q1, q2, q3, q4, q5}.
- MA = this parameter enables the use of the variable 'e1' that represents residual terms. That means, when it is set True, the functions related to extended least squares and (extended) one-step-ahead prediction will depend on the terms of 'e1'. If it is set False (default), those functions will work as a white noise output error problem.
- constant = if it is set 'True' it enables the terminal '1' and the individuals are allowed to have constant term. Otherwise, if it is set 'False' (default), the terminal '1' is not included in the primitive set.

These parameters can be changed using the function

> element.`setPset`(maxDelay,numberOfVariables,MA,constant)


### Create SISO Model
The `mggpElement` Class possesses the function `createModel(listStrings)`, that receives as argument a list of strings in which each string is a term of the model
    
Consider the following NARX system (Piroddi, 2003):

    y(k) = 0.75y(k-2) + 0.25u(k-1) - 0.20y(k-2)u(k-1)
    
Lets create an element object using the default parameters. Then create a model object that represent the aforementioned  system.

    from mggp import mggpElement
    element = mggpElement()
    listStrings = ['q1(y1)','u1','mul(q1(y1),u1)']
    model = element.createModel(listStrings)
    element.compile_model(model)

If the user wants to print the model terms in the console, just do:

    for term in model:
        print(str(term))

>Note: the `createModel()` function sets an attribute named `lagMax` in the model object that contains the maximum lag applied by the back-shift operators. That means, the model maximum lag is `maximumLag = model.lagMax + 1`. The example model has a `model.lagMax = 1`, and the maximum lag of the model is 2.

### Simulate a System
To simulate the aforementioned SISO model as a system use the free-run simulation function

> element.`predict_freeRun`(model,theta,y0,*inputs)

No matter the size of the initial conditions y0 you use, the function will work only with the size of the maximum lag in your model. For example, our model has a maximum lag equals 2, y0 must have at least size 2.
Lets say our input is 100 Gaussian distributed random values with mean of zero
and standard deviation of 1 and use the previous SISO model already built.

    u = np.random.normal(loc=0,scale=1,size=(100))
    y0 = np.zeros((2))
    theta = np.array([0.75,0.25,-0.20]) 
    y = element.predict_freeRun(model,theta,y0,u[:-1])

> Note 1: The model must be compiled to identify the regressors. 
Note 2: _u[:-1]_ is used to neglect the last sample of u for the prediction so that y and u have the same sizes.

To plot the output just do:

    import matplotlib.pyplot as plt
    plt.plot(y)
    plt.title('Simulation of the example model')    

### Create a MISO Model with Constant Term
Consider the following NARX MISO system, that has two inputs: _u_ and _h_.

    y(k) = 0.75y(k-2) + 0.25u(k-1) - 0.20y(k-2)u(k-1) - 0.5h(k-2) + 0.1

Now, the number of variables is 3 and it has a constant term. The user can change the name of the arguments using the function

> element.`renameArguments`(dictionary)

The default names are 'ARG0', 'ARG1', 'ARG2', etc. **The first argument must _always_ be the output.**
The dictionary maps the default names to the new ones.

    element = mggpElement()
    element.setPset(maxDelay=3,numberOfVariables=3,MA=False,
                    constant=True)
    element.renameArguments({'ARG0':'y1','ARG1':'u1','ARG2':'h1'})
    listStrings = ['q1(y1)','u1','mul(q1(y1),u1)','q1(h1)','1']
    model = element.createModel(listStrings)
    element.compile_model(model)

> Note 1: It is advised to name the arguments with the suffix '1'. It helps the user to remember that the variable is a one-step lagged variable.
Note 2: The constant term is always represented as '1' in the `listStrings` argument. Its value is determined by the model parameter.

### Simulate a System with White Noise Equation Error

    y(k) = 0.75y(k-2) + 0.25u(k-1) - 0.20y(k-2)u(k-1) + v(k)
>
    element = mggpElement()
    element.setPset(maxDelay=3,numberOfVariables=3,MA=False,
                    constant=False)
    element.renameArguments({'ARG0':'y1','ARG1':'u1','ARG2':'v'})
    listStrings = ['q1(y1)','u1','mul(q1(y1),u1)','v']
    model = element.createModel(listStrings)
    element.compile_model(model)
    y0 = np.zeros((2))
    u = np.random.normal(loc=0,scale=1,size=(100))
    v = np.random.normal(loc=0,scale=0.02,size=(100))
    theta = np.array([0.75,0.25,-0.20,1])
    y = element.predict_freeRun(model,theta,y0,u[:-1],v[:-1])

### Simulate a System with Colored Noise Equation Error

    y(k) = 0.75y(k-2) + 0.25u(k-1) - 0.20y(k-2)u(k-1) + 0.8v(k-1) + v(k)
>
    element = mggpElement()
    element.setPset(maxDelay=3,numberOfVariables=3,MA=False,
                    constant=False)
    element.renameArguments({'ARG0':'y1','ARG1':'u1','ARG2':'v'})
    listStrings = ['q1(y1)','u1','mul(q1(y1),u1)','q1(v)','v']
    model = element.createModel(listStrings)
    element.compile_model(model)
    y0 = np.zeros((2))
    u = np.random.normal(loc=0,scale=1,size=(100))
    v = np.random.normal(loc=0,scale=0.02,size=(100))
    theta = np.array([0.75,0.25,-0.20,0.8,1])
    y = element.predict_freeRun(model,y0,theta,u[:-1],v[:-1])

> Note: Although in the example the noise variable v do not have the suffix '1', this delay is applied on the 'regressor'. However, the noise variable is temporary. Experimental models do not have it as argument. So, it can be interpreted as a non-lagged variable. 

### The Least Squares Functions

> element.`ls`(model,y,*inputs)

Example:
For the SISO model: 

    theta = element.ls(model,y,u)
For the MISO model:

    theta = element.ls(model,y,u,v)

> element.`ls_extended`(model,y,*inputs)

Example:
For the SISO model: 

    theta = element.ls_extended(model,y,u)

For the MISO model:

    theta = element.ls_extended(model,y,u,v)

> Note: If the MA parameter is set 'True', the ELS will extend the regressor matrix with the MA part of the model. On the other hand, if it is set 'False', the extension is made as it was a white noise output error problem.

### Predictors
There are four predictors in the toolbox:
> element.`predict_freeRun`(model,theta,y0,*inputs)

Returns the free-run simulation with the initial conditions in the beginning.

> element.`predict`(model,theta,y,*inputs)

Returns the one-step-ahead prediction (without initial conditions in the beginning)

> element.`predict_extended`(model,theta,y,*inputs)

Returns the one-step-ahead prediction (without initial conditions in the beginning) from the extended regressor matrix. If the MA parameter is set 'False', the extension is made as it was a white noise output error problem.

> element.`predict_ksa`(model,theta,k,y,*inputs)

Returns a tuple with the k-steps-ahead (or multiple shooting) prediction (with initial conditions) and the 3-d batched array of y. They are 3-d arrays with the form (batches, data, 1). The number of batches are calculated dividing the data-set in several windows of (k + model maximum lag) size. The remaining data is discarded.

It can be confusing how to compare the predicted value with the desired one. There are to ways for the user remove the initial conditions from the y vector:

> yd = y[model.lagMax+1:]

and the built-in function

> yd = element.`getDesiredY`(model,y)

This function also works in the 3-d array from _ksa_ analysis.

### MSE built-in scores

> element.`score_osa`(model,theta,y,*inputs)

Returns the mse of the one-step-ahead predictor.

> element.`score_osa_ext`(model,theta,y,*inputs)

Returns the mse of the extended one-step-ahead predictor

> element.`score_freeRun`(model,theta,y,*inputs)

Returns the mse of the free-run simulation predictor

> element.`score_ksa`(model,theta,k,y,*inputs)

Returns the mse of the multiple shooting simulation predictor

### Moving Average Models
The Moving Average configuration is activated by the MA argument in the `setPset` function. If it is set 'True' the `mggpElement` object automatically includes an extra variable named 'e1'. That means, the `numberOfVariables` argument should not take into account the residual term.

Consider the following NARMAX model:

    y(k) = t1y(k-2) + t2u(k-1) - t3y(k-2)u(k-1) + t4e(k-1)
>
    element = mggpElement()
    element.setPset(maxDelay=3,numberOfVariables=2,MA=True,
                    constant=False)
    element.renameArguments({'ARG0':'y1','ARG1':'u1'})
    listStrings = ['q1(y1)','u1','mul(q1(y1),u1)','e1']
    model = element.createModel(listStrings)
    element.compile_model(model)
    
    theta = element.ls_extended(model,y,u)
    ypred = element.predict_extended(model,theta,y,u)

> Note: The free-run predictor for NARMAX models neglects the MA part. 

### Get Regressor Matrix
The user can get the regressor Matrix through the function:
> element.`makeRegressors`(model,y,*inputs,**options)

There are two arguments in options that can be set. They are
> mode: \{'default', 'extended'\}
 theta: if mode is set to 'extended', the theta values must be sent.

Examples:

    p = element.makeRegressors(model,y,u,mode='default')
    p = element.makeRegressors(model,y,u,mode='extended',theta=theta)

> Note: If MA is set 'True', the 'extended' mode must be used. If it is set 'False' the 'extended' mode will return a regressor matrix as it was a white noise output error problem.

### Orthogonal Least Squares
The built-in `ols` function apply a classical Gram-Smith Orthogonal Least Squares structure selection method.

> element.`ols`(model,tol,y,*inputs,**kwargs)

Example:

    element = mggpElement()
    element.setPset(maxDelay=3,numberOfVariables=2,MA=False,constant=False)
    element.renameArguments({'ARG0':'y1','ARG1':'u1'})
    listStrings = ['q1(y1)','u1','mul(q1(y1),u1)','y1','q1(u1)']
    model = element.createModel(listStrings)
    element.compile_model(model)
    element.ols(model,1e-3,y,u)
    for term in model:
        print(str(term))

> Note 1: If the MA mode is set 'True', the key argument 'theta' must be set. The algorithm will create a extended regressor matrix and apply ols pruning over it. Probably, all residual terms will be removed.
> element.`ols`(model,tol,y,*inputs,theta=theta)

> Note 2: The ols function perform an in-place operation, that means, it modifies the object sent as argument.
Note 3: The resultant model do not have its terms sorted by ERR coefficient.

### Include New Functions
Other functions can be included into the primitive set through the function:
> element.`addPrimitive`(function,arity)

> Note: Arity is the number of arguments the function takes.

Example 1: Exponential function

    element = mggpElement()
    element.setPset(maxDelay=1,numberOfVariables=2)
    element.renameArguments({'ARG0':'y1','ARG1':'u1'})
    element.addPrimitive(np.exp,1)
    listStrings = ['exp(u1)']
    model = element.createModel(listStrings)
    element.compile_model(model)
    u = np.linspace(-5,5)
    y0 = np.zeros(1)
    theta = np.array([1])
    y = element.predict_freeRun(model, theta, y0, u)
    plt.figure()
    plt.plot(u,y[1:])

Example 2: Sinusoidal function

    element = mggpElement()
    element.setPset(maxDelay=1,numberOfVariables=2)
    element.renameArguments({'ARG0':'y1','ARG1':'u1'})
    element.addPrimitive(np.sin,1)
    listStrings = ['sin(u1)']
    model = element.createModel(listStrings)
    element.compile_model(model)
    u = np.linspace(-5,5)
    y0 = np.zeros(1)
    theta = np.array([1])
    y = element.predict_freeRun(model, theta, y0, u)
    plt.figure()
    plt.plot(u,y[1:])

### Handling constraints with built-in functions
There are cases in which the structure of a model has some restrictions. It is possible to create constraints in functions arguments through the function:

> element.`constraint_funcs`(model,funcs,consts,values)

The arguments present in the list of string 'consts' will be removed from the functions present in the list of strings 'funcs' and replaced by values.

- model: model object to be constraint
- funcs: list of strings with functions names
- consts: list of strings with functions or arguments names. The latter must be the default names - 'ARG0', 'ARG1', etc
- values: list of terminals objects to be used as replacement

The terminals list can be gotten using the function:

> terminals = element.`getTerminalsObjects`()

For example, if the user wants to limit the 'mul' function to be used only with 'u1' variables:

    element.constraint_funcs(model, 'mul', 'ARG0',terminals[1])

> Note: to check terminals names use 
    terminals[index].name

Example 1: include the _sign_ function, restrain it to not have 'mul' nor 'sign' as arguments, and replace them by any terminal

    def sgn(x1):
        return np.sign(x1)
    
    element = mggpElement()
    element.setPset(maxDelay=1,numberOfVariables=2,constant=True)
    element.renameArguments({'ARG0':'y1','ARG1':'u1'})
    element.addPrimitive(sgn,1)
    listStrings = ['sgn(y1)','sgn(mul(u1,u1))','sgn(sgn(u1))']
    model = element.createModel(listStrings)
    element.compile_model(model)
    terminals = element.getTerminalsObjects()
    element.constraint_1arityFuncs(model, ['sgn'], 
                                ['mul','sgn'],terminals)

Example 2: include a _sign_ function of arity 2 that return sign(x1-x2) and restrain it to not have 'mul', 'sign' nor 'y1' as arguments, and replace them by 'u1'

    def sgn(x1,x2):
        return np.sign(x1-x2)
    
    element = mggpElement()
    element.setPset(maxDelay=1,numberOfVariables=2,constant=True)
    element.renameArguments({'ARG0':'y1','ARG1':'u1'})
    element.addPrimitive(sgn,2)
    listStrings = ['sgn(u1,y1)','sgn(u1,mul(u1,u1))',
                    'sgn(sgn(y1,u1),y1)']
    model = element.createModel(listStrings)
    element.compile_model(model)
    terminals = element.getTerminalsObjects()
    element.constraint_2arityFuncs(model, ['sgn'], 
                                  ['mul','sgn','ARG0'],terminals[1])


### Save and Load built-in functions
To make it easier to the user, the `mggpElement` class implements a save and a load functions (using the _pickle_ package).

> element.`save`(filename,dictionary)
> element.`load`(filename)

It can be used with dictionary objects. For example, if the user wants to save a model, a theta value, and training data:

    modelListString = element.model2List(model)
    dictionary = {'model':modelListString,
                  'theta':theta,
                  'y_train':y,
                  'u_train':u}
    element.save('modelInfo.pkl',dictionary)

> Note: It is advised to save models as list of strings. The `Individual` instances can generate conflicts with another `base.creator` modules.

The user can get it from the function:

> element.`model2List`(model)

And to retrieve the saved objects:

    dictionary = element.load('modelInfo.pkl')
    modelListString = dictionary['model']
    model = element.createModel(modelListString)
    theta = dictionary['theta']
    y_train = dictionary['y_train']
    u_train = dictionary['u_train']

## The mggpEvolver Class
The `mggpEvolver` class is responsible to execute the evolution of a population. The individuals from this population are created according to the primitive set defined in the `mggpElement` object. The following parameters can be set:

- popSize: population size (default = 100)
- CXPB: crossover probability (default = 0.9)
- MTPB: mutation probability (default 0.1)
- n_gen: number of generations (default = 50)
- maxHeight: maximum heiht of GP elements (default = 3)
- maxTerms: maximum number of model terms (default = 5)
- elite: percentage of the population to be included into the \textit{hall of fame} object and be kept in the population (default = 5)
- verbose: print statistics at each generation (default = True)
- element: \textit{mggpElement} object with the information needed to create individuals.

The `run` function have two arguments:

- evaluate: function that returns the individual fitness to be minimized. It must posses one single argument that is the individual to be evaluated. The function must return a tuple (value,) -- with the comma after value.
- seed: list of valid models (created by `element.createModel` function). If 'None' (default), no seed is included into the population.

The `run` function return a _hall of fame_ object.

### Simple Example
Consider the system:

    y(k) = 0.75y(k-2) + 0.25u(k-1) - 0.20y(k-2)u(k-1)

where u=WGN(0,1) with an output Gaussian noise with mean of zero and standard deviation of 0.08.

    from mggp import mggpElement, mggpEvolver
    import numpy as np
    
    # simulate the system
    element = mggpElement()
    element.setPset(maxDelay=1,numberOfVariables=2)
    element.renameArguments({'ARG0':'y1','ARG1':'u1'})
    listStrings = ['q1(y1)','u1','mul(q1(y1),u1)']
    model = element.createModel(listStrings)
    element.compile_model(model)
    u = np.random.normal(loc=0,scale=1,size=(500))
    y0 = np.zeros((2))
    theta = np.array([0.75,0.25,-0.20]) 
    y = element.predict_freeRun(model,theta,y0,u[:-1])
    y += np.random.normal(loc=0,scale=0.08,size=(500,1))
    
    # Create the element object to be used in MGGP 
    # in this case, it is the same used to create training data.
    
    mggp = mggpEvolver(popSize=500,CXPB=0.9,MTPB=0.1,n_gen=50,maxHeight=3,
                      maxTerms=5,verbose=True,elite=5,element=element)
    
    def evaluate(ind):
        try:
            element.compile_model(ind)
            theta = element.ls(ind,y,u)
            ind.theta = theta
            SE = element.score_osa(ind, theta, y, u)
            return SE,
        # exception treatment for cases of Singular Matrix
        except np.linalg.LinAlgError:
            return np.inf,
    hof = mggp.run(evaluate=evaluate,seed=None)
    model = hof[0]
    
    for term in model:
        print(str(term))


