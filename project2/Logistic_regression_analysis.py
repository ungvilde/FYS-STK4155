import numpy as np

from common import *
from LogisticRegression import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from GridSearch import GridSearch_LogReg, GridSearch_FFNN_classification_architecture, GridSearch_FFNN_classifier, GridSearch_LogReg_Sklearn, GridSearch_LogReg_epochs_batchsize
from ResampleMethods import *

from FFNN import FFNN
from activation_functions import *
from sklearn.neural_network import MLPClassifier



sns.set_theme("notebook", "whitegrid")

np.random.seed(123)

X, y = load_breast_cancer(return_X_y=True)
scaler = StandardScaler()
scaler.fit(X)
Xscaled = scaler.transform(X)
X = Xscaled
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)
plt.figure(figsize=(12*cm, 10*cm))

# First we use the simple GD and find some good eta and n_epochs
#epochs = [int(x) for x in np.linspace(50, 400, 100)]

#etas = [1e-2, 1e-3, 1e-4]
#lmbda = 0
#gamma = 0

# acc_values1 = [] 
# acc_values2 = []
# acc_values3 = []
# for n_epochs in epochs:
#     #eta1
#     logreg = LogisticRegression(lmbda=lmbda, solver="gd", max_iter=n_epochs, gamma=gamma, eta0=1e-2)
#     acc = CrossValidation_classification(logreg, X, y, k=5)
#     acc_values1.append(acc)

#     #eta2
#     logreg = LogisticRegression(lmbda=lmbda, solver="gd", max_iter=n_epochs, gamma=gamma, eta0=1e-3)
#     acc = CrossValidation_classification(logreg, X, y, k=5)
#     acc_values2.append(acc)

#     #eta3
#     logreg = LogisticRegression(lmbda=lmbda, solver="gd", max_iter=n_epochs, gamma=gamma, eta0=1e-4)
#     acc = CrossValidation_classification(logreg, X, y, k=5)
#     acc_values3.append(acc)


# print("MAX LogisticRegression GD Accuracy = ", max(acc_values1))    
# print("For eta = ", 1e-2, " and n_epochs = ", epochs[np.argmax(acc_values1)])

# print("MAX LogisticRegression GD Accuracy = ", max(acc_values2))
# print("For eta = ", 1e-3, " and n_epochs = ", epochs[np.argmax(acc_values2)])

# print("MAX LogisticRegression GD Accuracy = ", max(acc_values3))
# print("For eta = ", 1e-4, " and n_epochs = ", epochs[np.argmax(acc_values3)])

# plt.figure(figsize=(12*cm, 10*cm))

# plt.plot(epochs, acc_values1, label = f"$\eta_1 = {1e-2}$", c='tab:blue')
# plt.plot(epochs, acc_values2, label = f"$\eta_2 = {1e-3}$", c='tab:orange')
# plt.plot(epochs, acc_values3, label = f"$\eta_3 = {1e-4}$", c="tab:green")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# #plt.yscale('log')
# plt.legend()
# plt.tight_layout()
# plt.savefig(f"figs/logistic_acc_epoch_cancer_gamma_{gamma}.pdf")

# ### LETS TRY TO PLOT THIS WITH MOVING AVERAGE
# plt.clf()
# window = 10
# acc_values1_MA = moving_average(acc_values1, window)
# acc_values2_MA = moving_average(acc_values2, window)
# acc_values3_MA = moving_average(acc_values3, window)

# plt.plot(epochs[:-(window-1)], acc_values1_MA, label = f"$\eta_1 = {1e-2}$", c='tab:blue')
# plt.plot(epochs[:-(window-1)], acc_values2_MA, label = f"$\eta_2 = {1e-3}$", c='tab:orange')
# plt.plot(epochs[:-(window-1)], acc_values2_MA, label = f"$\eta_3 = {1e-4}$", c="tab:green")

# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# #plt.yscale('log')
# plt.legend()
# plt.tight_layout()
# plt.savefig(f"figs/logistic_acc_epoch_cancer_gamma_{gamma}_MA{window}.pdf")

# #########################################################
# # Now, with eta = 1e-3, let us try sgd with different optimizers just to see how they perform
# plt.clf()
#acc_values1 = [] 
#acc_values2 = []
#acc_values3 = []
#acc_values4 = []
#acc_values5 = []
#
#plt.figure(figsize=(12*cm, 10*cm))
#
#epochs = [int(x) for x in np.linspace(50, 450, 100)]
#
#for n_epochs in epochs:
#
#    logreg = LogisticRegression(lmbda=lmbda, solver="sgd", optimization=None ,max_iter=n_epochs, batch_size=20, gamma=0, eta0=1e-3)
#    acc = CrossValidation_classification(logreg, X, y, k=5)
#    acc_values1.append(acc)
#
#    logreg = LogisticRegression(lmbda=lmbda, solver="sgd", optimization=None ,max_iter=n_epochs, batch_size=20, gamma=0.9, eta0=1e-3)
#    acc = CrossValidation_classification(logreg, X, y, k=5)
#    acc_values2.append(acc)
#
#    logreg = LogisticRegression(lmbda=lmbda, solver="sgd", optimization='RMSprop' ,max_iter=n_epochs, batch_size=20, gamma=0, eta0=1e-3)
#    acc = CrossValidation_classification(logreg, X, y, k=5)
#    acc_values3.append(acc)
#
#    logreg = LogisticRegression(lmbda=lmbda, solver="sgd", optimization='adam' ,max_iter=n_epochs, batch_size=20, gamma=0, eta0=1e-3)
#    acc = CrossValidation_classification(logreg, X, y, k=5)
#    acc_values4.append(acc)
#
#    logreg = LogisticRegression(lmbda=lmbda, solver="sgd", optimization='adagrad' ,max_iter=n_epochs, batch_size=20, gamma=0, eta0=1e-3)
#    acc = CrossValidation_classification(logreg, X, y, k=5)
#    acc_values5.append(acc)
#
#
#plt.plot(epochs, acc_values1, label = f"No Optimization", c='tab:blue')
#print("best acc no optimization", max(acc_values1), "at epoch", epochs[np.argmax(acc_values1)])
#plt.plot(epochs, acc_values2, label = f"With moment $\gamma=0.9$", c='tab:red')
#print("best acc With moment $\gamma=0.9$", max(acc_values2), "at epoch", epochs[np.argmax(acc_values2)])
#plt.plot(epochs, acc_values3, label = f"RMSprop", c="tab:green")
#print("best acc RMSprop", max(acc_values3) , "at epoch", epochs[np.argmax(acc_values3)])
#plt.plot(epochs, acc_values4, label = f"ADAM", c="tab:orange")
#print("best acc Adam", max(acc_values4), "at epoch", epochs[np.argmax(acc_values4)])
#plt.plot(epochs, acc_values5, label = f"AdaGrad", c="tab:purple")
#print("best acc AdaGrad", max(acc_values5), "at epoch", epochs[np.argmax(acc_values5)])
#
#plt.xlabel("Epochs")
#plt.ylabel("Accuracy")
##plt.yscale('log')
#plt.legend()
#plt.tight_layout()
#plt.savefig("figs/logistic_acc_epoch_cancer_multiple_sgd.pdf")
### LETS TRY TO PLOT THIS WITH MOVING AVERAGE
#plt.clf()
#window = 10
#acc_values1_MA = moving_average(acc_values1, window)
#acc_values2_MA = moving_average(acc_values2, window)
#acc_values3_MA = moving_average(acc_values3, window)
#acc_values4_MA = moving_average(acc_values4, window)
#acc_values5_MA = moving_average(acc_values5, window)
#
#
#plt.plot(epochs[:-(window-1)], acc_values1_MA, label = f"No Optimization", c='tab:blue')
#print("best acc no optimization", max(acc_values1_MA))
#plt.plot(epochs[:-(window-1)], acc_values2_MA, label = f"With moment $\gamma=0.9$", c='tab:red')
#print("best acc With moment $\gamma=0.9$", max(acc_values2_MA))
#plt.plot(epochs[:-(window-1)], acc_values3_MA, label = f"RMSprop", c="tab:green")
#print("best acc RMSprop", max(acc_values3_MA))
#plt.plot(epochs[:-(window-1)], acc_values4_MA, label = f"ADAM", c="tab:orange")
#print("best acc Adam", max(acc_values4_MA))
#plt.plot(epochs[:-(window-1)], acc_values5_MA, label = f"AdaGrad", c="tab:purple")
#print("best acc AdaGrad", max(acc_values5_MA))
#
#plt.xlabel("Epochs")
#plt.ylabel("Accuracy")
##plt.yscale('log')
#plt.legend()
#plt.tight_layout()
#plt.savefig(f"figs/logistic_acc_epoch_cancer_multiple_sgd_MA{window}.pdf")

# best acc no optimization 0.9771774569166279 at epoch 441
# best acc With moment $\gamma=0.9$ 0.9841639496972518 at epoch 171
# best acc RMSprop 0.9859649122807017 at epoch 445
# best acc Adam 0.982425089271852 at epoch 365
# best acc AdaGrad 0.9701599130569788 at epoch 389

#######################################################
# Now we gridsearch batch sizes and epochs 
batch_sizes = np.array([10, 20, 50, 100] )
epochs = np.array([100, 200,400, 600, 800, 1000])


results = GridSearch_LogReg_epochs_batchsize(
    X,
    y, 
    lmbda=0,
    eta=1e-3, 
    solver="sgd",
    optimization=None,
    plot_grid=True,
    gamma=0.9,
    max_iters=epochs,
    batch_sizes=batch_sizes,
    k=5
    )

print(results)

# [[0.95430834 0.94902965 0.95607825 0.96478808 0.9490607  0.94375097]
#  [0.93842571 0.9595715  0.96135693 0.95256948 0.94025772 0.95609377]
#  [0.95430834 0.9560472  0.95433939 0.94733737 0.94733737 0.95601615]
#  [0.95606272 0.96134141 0.9489986  0.95787921 0.94725974 0.95786369]]



###########################################################
# Let us do a gridsearch of lambda and eta with sgd for diferent optimizers
eta_values = np.logspace(-4, -1, 7)
lmbda_values = np.logspace(-8, 0, 7)

# print("######### adam 20 #########")
# result = GridSearch_LogReg(
#     X,
#     y, 
#     lmbda_values, 
#     eta_values, 
#     solver="sgd",
#     optimization="adam",
#     plot_grid=True,
#     gamma=0.9,
#     max_iter=400,
#     batch_size=20,
#     k=5
#     )
# print(result)

# print("######### RMSprop 20 #########")
# result = GridSearch_LogReg(
#     X,
#     y, 
#     lmbda_values, 
#     eta_values, 
#     solver="sgd",
#     optimization="RMSprop",
#     plot_grid=True,
#     gamma=0.9,
#     max_iter=400,
#     batch_size=20,
#     k=5
#     )
# print(result)


# print("######### adagrad 20 #########")

# result = GridSearch_LogReg(
#     X,
#     y, 
#     lmbda_values, 
#     eta_values, 
#     solver="sgd",
#     optimization="adagrad",
#     plot_grid=True,
#     gamma=0.9,
#     max_iter=1000,
#     batch_size=20,
#     k=5
#     )
# print(result)


# print("######### None 20 #########")

# result = GridSearch_LogReg(
#     X,
#     y, 
#     lmbda_values, 
#     eta_values, 
#     solver="sgd",
#     optimization=None,
#     plot_grid=True,
#     gamma=0.9,
#     max_iter=400,
#     batch_size=20,
#     k=5
#     )
# print(result)
# print("######### adam 40 #########")
# result = GridSearch_LogReg(
#     X,
#     y, 
#     lmbda_values, 
#     eta_values, 
#     solver="sgd",
#     optimization="adam",
#     plot_grid=True,
#     gamma=0.9,
#     max_iter=400,
#     batch_size=40,
#     k=5
#     )
# print(result)

# print("######### RMSprop 40 #########")
# result = GridSearch_LogReg(
#     X,
#     y, 
#     lmbda_values, 
#     eta_values, 
#     solver="sgd",
#     optimization="RMSprop",
#     plot_grid=True,
#     gamma=0.9,
#     max_iter=400,
#     batch_size=40,
#     k=5
#     )
# print(result)


# print("######### adagrad 40 #########")

# result = GridSearch_LogReg(
#     X,
#     y, 
#     lmbda_values, 
#     eta_values, 
#     solver="sgd",
#     optimization="adagrad",
#     plot_grid=True,
#     gamma=0.9,
#     max_iter=1000,
#     batch_size=40,
#     k=5
#     )
# print(result)


# print("######### None 40 #########")

# result = GridSearch_LogReg(
#     X,
#     y, 
#     lmbda_values, 
#     eta_values, 
#     solver="sgd",
#     optimization=None,
#     plot_grid=True,
#     gamma=0.9,
#     max_iter=400,
#     batch_size=40,
#     k=5
#     )
# print(result)

######### adam 20 #########
# [[0.87523676 0.85596957 0.84028878 0.82251203 0.91737308 0.85047353
#   0.8138022 ]
#  [0.95075299 0.95960255 0.95247632 0.96131036 0.96134141 0.95786369
#   0.96132588]
#  [0.97189877 0.97186772 0.97183667 0.97362211 0.97012886 0.97189877
#   0.97542307]
#  [0.97188325 0.97716193 0.97714641 0.97011334 0.97890079 0.97716193
#   0.97188325]
#  [0.96840553 0.97537649 0.96483465 0.9648657  0.97713088 0.98242509
#   0.97362211]
#  [0.96132588 0.97719298 0.97363763 0.96660456 0.96481913 0.97893184
#   0.97540755]
#  [0.96481913 0.95078404 0.9648657  0.97015991 0.97716193 0.98593386
#   0.9719143 ]]
# ######### RMSprop 20 #########
# [[0.89809036 0.89807483 0.84330073 0.89976712 0.89809036 0.88762615
#   0.91575842]
#  [0.95784816 0.97012886 0.952585   0.9683279  0.96657351 0.9718522
#   0.97011334]
#  [0.98065518 0.97363763 0.97714641 0.97539202 0.97009781 0.97891632
#   0.97017544]
#  [0.97363763 0.97188325 0.97015991 0.97714641 0.97362211 0.97719298
#   0.97368421]
#  [0.97011334 0.96837448 0.96485018 0.97537649 0.96835895 0.97714641
#   0.97015991]
#  [0.96663562 0.96129483 0.9718522  0.97542307 0.97363763 0.98240956
#   0.97360658]
#  [0.9648657  0.97011334 0.97542307 0.9718522  0.96662009 0.97893184
#   0.97360658]]
# ######### adagrad 20 #########
# [[0.97540755 0.97012886 0.96312684 0.97189877 0.9648657  0.98065518
#   0.97189877]
#  [0.97359106 0.98063965 0.97540755 0.97713088 0.97717746 0.97719298
#   0.97366868]
#  [0.97360658 0.97716193 0.9719143  0.97713088 0.97012886 0.98242509
#   0.97893184]
#  [0.96483465 0.96835895 0.96840553 0.96132588 0.97362211 0.97539202
#   0.97540755]
#  [0.96663562 0.96301816 0.95960255 0.96312684 0.97362211 0.98244061
#   0.97011334]
#  [0.96132588 0.97186772 0.9612793  0.96660456 0.97362211 0.98240956
#   0.97180562]
#  [0.97363763 0.96665114 0.95952492 0.96483465 0.97359106 0.97893184
#   0.97189877]]
# ######### None 20 #########
# [[0.96481913 0.97357553 0.97011334 0.97012886 0.96481913 0.97014439
#   0.97008229]
#  [0.97192982 0.97891632 0.98068623 0.9683279  0.97714641 0.97888527
#   0.97360658]
#  [0.98239404 0.96837448 0.9683279  0.97537649 0.97365316 0.98237851
#   0.97532992]
#  [0.96483465 0.97186772 0.96839    0.96840553 0.96835895 0.98065518
#   0.97362211]
#  [0.96304922 0.96308027 0.96301816 0.96662009 0.97536097 0.98068623
#   0.97536097]
#  [0.95777053 0.96134141 0.95780158 0.96660456 0.97189877 0.97716193
#   0.97537649]
#  [0.94553641 0.95784816 0.95427729 0.96304922 0.96311132 0.9770843
#   0.97362211]]
# ######### adam 40 #########
# [[0.5617606  0.75412203 0.7417016  0.55339233 0.68835585 0.74865704
#   0.71299488]
#  [0.93839466 0.93487036 0.92440615 0.93501009 0.92780624 0.88410185
#   0.92791492]
#  [0.97188325 0.97186772 0.97009781 0.96665114 0.96304922 0.97189877
#   0.97539202]
#  [0.96835895 0.96840553 0.97717746 0.97366868 0.97891632 0.98242509
#   0.98062413]
#  [0.97009781 0.96835895 0.97009781 0.96655799 0.97365316 0.97713088
#   0.97888527]
#  [0.97008229 0.9648657  0.97537649 0.97537649 0.96840553 0.97363763
#   0.97717746]
#  [0.96303369 0.96662009 0.9612793  0.96478808 0.96666667 0.97711535
#   0.97890079]]
# ######### RMSprop 40 #########
# [[0.81925167 0.83504114 0.75916783 0.71125602 0.81566527 0.82406459
#   0.79445738]
#  [0.9490607  0.94727527 0.95084614 0.95601615 0.93851886 0.97365316
#   0.94725974]
#  [0.97534544 0.96663562 0.97540755 0.97888527 0.97183667 0.97711535
#   0.97711535]
#  [0.97180562 0.97363763 0.97714641 0.96660456 0.96662009 0.97363763
#   0.98065518]
#  [0.97360658 0.97189877 0.96489676 0.96840553 0.97012886 0.97714641
#   0.97890079]
#  [0.97009781 0.96662009 0.9719143  0.96834342 0.97186772 0.97539202
#   0.97536097]
#  [0.95784816 0.96837448 0.97362211 0.95783263 0.96837448 0.98065518
#   0.97890079]]
# ######### adagrad 40 #########
# [[0.96481913 0.97540755 0.96660456 0.97714641 0.97017544 0.97540755
#   0.97363763]
#  [0.97894737 0.97540755 0.96834342 0.97186772 0.97893184 0.97188325
#   0.97891632]
#  [0.96837448 0.97189877 0.97188325 0.97366868 0.97011334 0.97537649
#   0.98416395]
#  [0.96663562 0.96835895 0.97012886 0.97365316 0.97186772 0.97011334
#   0.97893184]
#  [0.96135693 0.96655799 0.96660456 0.96483465 0.9648657  0.98240956
#   0.97366868]
#  [0.95778606 0.95432386 0.96311132 0.96837448 0.96312684 0.97363763
#   0.97539202]
#  [0.95961807 0.95253843 0.9719143  0.96837448 0.97359106 0.97713088
#   0.97717746]]
# ######### None 40 #########
# [[0.97015991 0.97014439 0.97539202 0.97890079 0.96666667 0.97893184
#   0.97714641]
#  [0.97890079 0.9806707  0.97360658 0.97539202 0.97542307 0.97363763
#   0.97537649]
#  [0.96660456 0.97365316 0.97365316 0.97893184 0.97188325 0.98065518
#   0.97709983]
#  [0.96135693 0.96132588 0.96657351 0.96483465 0.98068623 0.97009781
#   0.97893184]
#  [0.96135693 0.96311132 0.96309579 0.9648657  0.97359106 0.97012886
#   0.98240956]
#  [0.96663562 0.97188325 0.95783263 0.96483465 0.97714641 0.97539202
#   0.97893184]
#  [0.9595715  0.95255395 0.94542773 0.95432386 0.96485018 0.9806707
#   0.9754386 ]]




# then we get the best results (adam with lambda 10e-1.33 and eta 10e-1) 
# and compare to our FFNN. For this we need to train the FFNN

# n_layers = [1,2,3,4,5]
# n_neurons = [10,30,50,80,100]
# n_epochs = 400
# eta=1e-3

# results = GridSearch_FFNN_classification_architecture(
#     Xtrain,
#     ytrain,
#     n_layers,
#     n_neurons, 
#     eta, 
#     n_epochs,
#     lmbda=0, 
#     plot_grid=True,
#     gamma=0.9,
#     activation_hidden="sigmoid",
#     batch_size=20,
#     k=5
#     )

# print(results)

# [[0.97582418 0.96703297 0.97142857 0.95384615 0.96483516]
#  [0.96923077 0.96703297 0.97362637 0.96483516 0.97142857]
#  [0.96703297 0.97142857 0.8989011  0.81538462 0.76703297]
#  [0.96263736 0.96923077 0.89450549 0.62417582 0.62417582]
#  [0.96703297 0.96703297 0.77802198 0.62417582 0.62417582]]
# # using optimal architecture, we now look at the learning rates and lambdas


# tuning the best FFNN parameters given the architecture
# results = GridSearch_FFNN_classifier(
#      X,
#      y, 
#      lmbda_values, 
#      eta_values, 
#      plot_grid=True,
#      gamma=0.9,
#      activation_hidden="sigmoid",
#      n_epochs=400,
#      batch_size=20,
#      n_hidden_neurons = [10],
#      k=5
#      )
# print(results)

# [[0.9648657  0.96829685 0.97539202 0.97716193 0.97357553 0.96660456
#   0.95429281]
#  [0.97365316 0.97537649 0.97363763 0.98068623 0.97890079 0.97719298
#   0.95787921]
#  [0.97540755 0.97894737 0.97894737 0.97540755 0.97539202 0.97365316
#   0.95427729]
#  [0.97540755 0.98242509 0.97716193 0.97893184 0.97542307 0.97717746
#   0.92279149]
#  [0.97886974 0.97182115 0.98070175 0.97532992 0.97890079 0.97716193
#   0.95606272]
#  [0.97542307 0.97891632 0.9806707  0.97363763 0.98594939 0.98068623
#   0.94378202]
#  [0.98245614 0.97717746 0.98240956 0.98070175 0.97717746 0.95433939
#   0.62738705]]

# Then we get the best results and compare to sklearn

results = GridSearch_LogReg_Sklearn(
    X,
    y, 
    lmbda_values, 
    eta_values, 
    plot_grid=True,
    max_iter=400,
    k=5
    )
print(results)

# ## what if we let sklearn choose the eta and lambda?
logreg = SGDClassifier(max_iter=400, tol=1e-3)
accuracy_score = CrossValidation_classification(logreg, X, y, k=5)
print("Sklearn automatic accuracy score", accuracy_score)

# [[0.96488123 0.9648036  0.95960255 0.96662009 0.97537649 0.97714641
#   0.94373544]
#  [0.95255395 0.96662009 0.96134141 0.96132588 0.96660456 0.97186772
#   0.94898308]
#  [0.94907623 0.97186772 0.96309579 0.97189877 0.96835895 0.97186772
#   0.9489986 ]
#  [0.96663562 0.95609377 0.95783263 0.96663562 0.97714641 0.97717746
#   0.94733737]
#  [0.96831237 0.96837448 0.97008229 0.97015991 0.97716193 0.98068623
#   0.94555193]
#  [0.96839    0.96137246 0.9525229  0.96835895 0.96660456 0.97716193
#   0.94730632]
#  [0.9490607  0.96306474 0.96658904 0.95256948 0.97189877 0.96660456
#   0.94901413]]
# Sklearn automatic accuracy score 0.9665579878900792