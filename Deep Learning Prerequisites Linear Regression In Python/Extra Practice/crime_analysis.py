import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# X1 = total overall reported crime rate per 1 million residents
# X2 = reported violent crime rate per 100,000 residents
# X3 = annual police funding in $/resident
# X4 = % of people 25 years+ with 4 yrs. of high school
# X5 = % of 16 to 19 year-olds not in highschool and not highschool graduates.
# X6 = % of 18 to 24 year-olds in college
# X7 = % of people 25 years+ with at least 4 years of college

def make_poly(X, deg):
    n = len(X)
    data = [np.ones(n)]
    for d in range(0, deg):
        data.append(X**(d+1))
    return np.vstack(data).T

def fit(X, Y):
    return np.linalg.solve(X.T.dot(X), X.T.dot(Y))

def getTitle(c):
    if c == 0:
        return "total overall reported crime rate per 1 million residents"
    elif c == 1:
        return "violent crime rate per 100,000"
    elif c == 2:
        return "anual police funding $/resident"
    elif c == 3:
        return "% of people 25 years+ with 4 yrs. of high school"
    elif c == 4:
        return "% of 16 to 19 year-olds not in highschool and not highschool graduates."
    elif c == 5:
        return "% of 18 to 24 year-olds in college"
    elif c == 6:
        return "% of people 25 years+ with at least 4 years of college"

def fit_and_display(X, Y, title, sample, deg):
    N = len(X)

    train_idx = np.random.choice(N, sample)
    Xtrain = X[train_idx]
    Ytrain = Y[train_idx]

    plt.scatter(X, Y)
    plt.title(title)
    plt.show()

    # fit polynomial
    Xtrain_poly = make_poly(Xtrain, deg)
    print("Xtrain_poly", Xtrain_poly)
    print("Ytrain", Ytrain)
    w = fit(Xtrain_poly, Ytrain)

    #display the polynomial
    X_poly = make_poly(X,deg)
    Y_hat = X_poly.dot(w)
    plt.plot(sorted(X), sorted(Y))
    plt.plot(sorted(X), sorted(Y_hat))
#    plt.plot(X, Y)
#    plt.plot(X, Y_hat)
    plt.scatter(Xtrain, Ytrain)
    plt.title("deg = %d" % deg)
    plt.show()
    

def get_mse(Y, Yhat):
    d = Y-Yhat
    return d.dot(d)/len(d)

def plot_train_vs_test_curves(X, Y, title,sample=20, max_deg=20):
    N = len(X)
    
    # Pick a random sample of (sample many) numbers to use as train
    train_idx = np.random.choice(N, sample)
    # build an array of Xtrain
    Xtrain = X[train_idx]
    # build an array of Ytrain
    Ytrain = Y[train_idx]

    test_idx = [idx for idx in range(0,N) if idx not in train_idx]
    # build an array of non train terms to test aka Xtest
    Xtest = X[test_idx]
    # build an array of non-train terms to test aka Ytest
    Ytest = Y[test_idx]

    mse_trains = []
    mse_tests = []

    for deg in range(0, max_deg+1):
        Xtrain_poly = make_poly(Xtrain, deg)
        w = fit(Xtrain_poly, Ytrain)
        Yhat_train = Xtrain_poly.dot(w)
        mse_train = get_mse(Ytrain, Yhat_train)

        Xtest_poly = make_poly(Xtest, deg)
        Yhat_test = Xtest_poly.dot(w)
        mse_test = get_mse(Ytest, Yhat_test)

        mse_trains.append(mse_train)
        mse_tests.append(mse_test)

    plt.plot(mse_trains, label='train_mse')
    plt.plot(mse_tests, label='test_mse')
    plt.title(title)
    plt.legend()
    plt.show()

    plt.plot(mse_trains, label='train_mse')
    plt.title(title)
    plt.legend()
    plt.show()

def fit_and_plot(X,Y, r2, label):
    denominator = X.dot(X) - X.mean() * X.sum()
    a = (X.dot(Y) - Y.mean()*X.sum())/denominator
    b = (X.dot(X)*Y.mean()-X.dot(Y)*X.mean())/denominator

    Yhat = a*X+b

    plt.scatter(X,Y)
    plt.plot(X,Yhat)
    plt.title(label + "    r2: " + str(r2))
    plt.show()

def get_r2(X,Y, Yhat=None):
    w = np.linalg.solve(X.T.dot(X), X.T.dot(Y))

    if Yhat == None:
        Yhat = X.dot(w)
    else:
        None
        
    d1 = Y- Yhat
    d2 = Y - Y.mean()
    r2 = 1 - d1.dot(d1)/d2.dot(d2)
    
    return r2

def l1_regularization(X, Y):
    costs = []

    w= np.random.randn(5) / np.sqrt(5)
    learning_rate = 0.001

    #l1 regularization term
    l1 = -25.0
    
    for t in range(0,200):
        Yhat = X.dot(w)
        delta = Yhat - Y
        w = w - learning_rate*(X.T.dot(delta) + l1*np.sign(w))

        mse = delta.dot(delta) / N
        costs.append(mse)

    plt.plot(costs)
    plt.show()

    print("Yhat", Yhat)
    print("Y", Y)

    plt.plot(Yhat, label = 'yhat')
    plt.plot(Y, label = 'y')
    plt.legend()
    plt.title("L1 Regularization")
    plt.show()


def l2_regularization(X, Y, title):
    X = np.vstack([np.ones(len(X)), X]).T

    #calculate the maximum likelihood solution
    w_ml = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
    Yhat_ml = X.dot(w_ml)
    plt.scatter(X[:,1], Y)
    plt.plot(X[:,1], Yhat_ml)
    plt.title(title)
    plt.show()

    #set the l2 penality
    l2 = 100.0

    w_map = np.linalg.solve(l2*np.eye(2) + X.T.dot(X), X.T.dot(Y))
    Yhat_map = X.dot(w_map)
    plt.scatter(X[:,1],Y)
    plt.plot(X[:,1], Yhat_ml, label=('maximum likelihood' + " r2: " + str(get_r2(X, Y, Yhat_ml))))
    print("Yhat_ml slope:",get_slope(X[:,1], Yhat_ml))
    plt.plot(X[:,1], Yhat_map, label=('map' + " r2: " + str(get_r2(X, Y, Yhat_map))))
    print("Yhat_map slope:",get_slope(X[:,1], Yhat_map))
    plt.title(title)
    plt.legend()
    plt.show()

def get_slope(X, Y):
    denominator = X.dot(X) - X.mean()*X.sum()
    return (X.dot(Y) - Y.mean()*X.sum())/denominator

def normalize(X):
    return (X - X.mean()) / np.std(X)
    

df = pd.read_excel('mlr06.xls')
df['ones'] = 1

overallReportedCrime = normalize(df['X1'])
violentCrimeRate = normalize(df['X2'])
annualPoliceFunding = normalize(df['X3'])
highSchoolGrads = normalize(df['X4'])
NotInHighSchool = normalize(df['X5'])
InCollege = normalize(df['X6'])
CollegeGrad = normalize(df['X7'])

print("overallReportedCrime:\n", overallReportedCrime,"\n")
print("violentCrimeRate:\n", violentCrimeRate,"\n")

print("annualPoliceFunding:\n", annualPoliceFunding,"\n")
plt.scatter(annualPoliceFunding, overallReportedCrime, label="overallReportedCrime")
plt.plot(sorted(annualPoliceFunding), sorted(overallReportedCrime), label="sorted(overallReportedCrime)")
plt.scatter(annualPoliceFunding, violentCrimeRate, label="violentCrimeRate")
plt.plot(sorted(annualPoliceFunding), sorted(violentCrimeRate), label="sorted(violentCrimeRate)")
plt.title("annualPoliceFunding")
plt.legend()
plt.show()
#fit_and_plot(annualPoliceFunding, overallReportedCrime, get_r2(df[['X3', 'ones']], overallReportedCrime), "Annual Police Funding vs. Overall Reported Crime")
#fit_and_plot(annualPoliceFunding, violentCrimeRate, get_r2(df[['X3', 'ones']], violentCrimeRate), "Annual Police Funding vs. Violent Crime Rate")
l2_regularization(annualPoliceFunding, overallReportedCrime, "Annual Police Funding vs. Overall Reported Crime")
l2_regularization(annualPoliceFunding, violentCrimeRate, "Annual Police Funding vs. Violent Crime Rate")
for deg in range(5,25):
    fit_and_display(annualPoliceFunding, overallReportedCrime, "Annual Police Funding vs. Overall Reported Crime", 20, deg)
plot_train_vs_test_curves(annualPoliceFunding,overallReportedCrime, "Annual Police Funding vs. Overall Reported Crime", sample=20, max_deg=25)

print("highSchoolGrads:\n", highSchoolGrads,"\n")
plt.scatter(highSchoolGrads, overallReportedCrime, label="overallReportedCrime")
plt.scatter(highSchoolGrads, violentCrimeRate, label="violentCrimeRate")
plt.title("highSchoolGrads")
plt.legend()
plt.show()
#fit_and_plot(highSchoolGrads, overallReportedCrime, get_r2(df[['X4', 'ones']], overallReportedCrime), "High School Grad Rate vs. Overall Reported Crime")
#fit_and_plot(highSchoolGrads, violentCrimeRate, get_r2(df[['X4', 'ones']], violentCrimeRate), "High School Grad Rate vs. Violent Crime Rate")
l2_regularization(highSchoolGrads, overallReportedCrime, "highSchoolGrads vs. Overall Reported Crime")
l2_regularization(highSchoolGrads, violentCrimeRate, "highSchoolGrads vs. Violent Crime Rate")

print("16to19NotInHighSchool:\n", NotInHighSchool,"\n")
plt.scatter(NotInHighSchool, overallReportedCrime, label="overallReportedCrime")
plt.scatter(NotInHighSchool, violentCrimeRate, label="violentCrimeRate")
plt.title("16to19NotInHighSchool")
plt.legend()
plt.show()
#fit_and_plot(NotInHighSchool, overallReportedCrime, get_r2(df[['X5', 'ones']], overallReportedCrime), "Teens Not In High School vs. Overall Reported Crime")
#fit_and_plot(NotInHighSchool, violentCrimeRate, get_r2(df[['X5', 'ones']], violentCrimeRate), "Teens Not In High School vs. Violent Crime Rate")
l2_regularization(NotInHighSchool, overallReportedCrime, "16to19NotInHighSchool vs. Overall Reported Crime")
l2_regularization(NotInHighSchool, violentCrimeRate, "16to19NotInHighSchool vs. Violent Crime Rate")

print("18to24InCollege:\n", InCollege,"\n")
plt.scatter(InCollege, overallReportedCrime, label="overallReportedCrime")
plt.scatter(InCollege, violentCrimeRate, label="violentCrimeRate")
plt.title("18to24InCollege")
plt.legend()
plt.show()
#fit_and_plot(InCollege, overallReportedCrime, get_r2(df[['X6', 'ones']], overallReportedCrime), "18to24InCollege vs. Overall Reported Crime")
#fit_and_plot(InCollege, violentCrimeRate, get_r2(df[['X6', 'ones']], violentCrimeRate), "18to24InCollege vs. Violent Crime Rate")
l2_regularization(InCollege, overallReportedCrime, "18to24InCollege vs. Overall Reported Crime")
l2_regularization(InCollege, violentCrimeRate, "18to24InCollege vs. Violent Crime Rate")

print("25PlusCollegeGradRate:\n", CollegeGrad,"\n")
plt.scatter(CollegeGrad, overallReportedCrime, label="overallReportedCrime")
plt.scatter(CollegeGrad, violentCrimeRate, label="violentCrimeRate")
plt.title("25PlusCollegeGradRate")
plt.legend()
plt.show()
#fit_and_plot(CollegeGrad, overallReportedCrime, get_r2(df[['X7', 'ones']], overallReportedCrime), "25PlusCollegeGradRate vs. Overall Reported Crime")
#fit_and_plot(CollegeGrad, violentCrimeRate, get_r2(df[['X7', 'ones']], violentCrimeRate), "25PlusCollegeGradRate vs. Violent Crime Rate")
l2_regularization(CollegeGrad, overallReportedCrime, "25PlusCollegeGradRate vs. Overall Reported Crime")
l2_regularization(CollegeGrad, violentCrimeRate, "25PlusCollegeGradRate vs. Violent Crime Rate")
