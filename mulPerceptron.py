from keras.models import Sequential
from keras.layers import Dense
#aktifkan kalau kcv saja
from sklearn.model_selection import StratifiedKFold

import numpy
#tentukan ukuran seed sehingga nanti bisa dipakai lagi
seed = 7
numpy.random.seed(seed)

#load data
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter = ",")
#split data into input(X) and output(Y)
X = dataset[:,0:8]
Y = dataset[:,8]

################################ validation with automatic verivication

# #create model
# model = Sequential()
# model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
# model.add(Dense(8, init='uniform', activation='relu'))
# model.add(Dense(1, init='uniform', activation='sigmoid'))

# #compile model
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# #fit the model
# model.fit(X, Y, validation_split=0.33, nb_epoch=150, batch_size=10)

# #evaluate the model
# scores = model.evaluate(X, Y)
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

########################   validation with k cross validation

# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(X, Y):
  # create model
  model = Sequential()
  model.add(Dense(12, input_dim=8, init= 'uniform' , activation= 'relu' ))
  model.add(Dense(8, init= 'uniform' , activation= 'relu' ))
  model.add(Dense(1, init= 'uniform' , activation= 'sigmoid' ))
  # Compile model
  model.compile(loss= 'binary_crossentropy' , optimizer= 'adam' , metrics=['accuracy'])
  # Fit the model
  model.fit(X[train], Y[train], nb_epoch=150, batch_size=10, verbose=0)
  # evaluate the model
  scores = model.evaluate(X[test], Y[test], verbose=0)
  print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
  cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))