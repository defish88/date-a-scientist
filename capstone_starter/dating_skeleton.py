import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.linear_model import LinearRegression

import timeit

#Create your df here:
df = pd.read_csv("profiles.csv")

body_type_mapping = {"skinny": 0, "thin": 1, "average": 2, "fit":3, "athletic": 4, "jacked": 5,
"used up": 6, "a little extra": 7, "overweight": 8, "curvy": 9, "full figured": 10, "rather not say": 11}
df["body_type_code"] = df.body_type.map(body_type_mapping)

diet_mapping = {"anything": 0, "vegeterian": 1, "vegan": 2, "kosher": 3, "halal": 4}
df["diet_code"] = df.diet.map(diet_mapping)

sex_mapping = {"m": 0, "f": 1}
df["sex_code"] = df.sex.map(sex_mapping)

drugs_mapping = {"never": 0, "sometimes": 1, "often": 2}
df["drugs_code"] = df.drugs.map(drugs_mapping)

smokes_mapping = {"no": 0, "trying to quit": 1, "when drinking": 2, "sometimes": 3, "yes": 4}
df["smokes_code"] = df.smokes.map(smokes_mapping)

drinks_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
df["drink_code"] = df.drinks.map(drinks_mapping)

bty_data = df[["age", "diet_code", "sex_code", "drugs_code", "smokes_code", "drink_code", "income"]]

x = bty_data.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

bty_data = pd.DataFrame(x_scaled, columns = ["age", "diet_code", "sex_code", "drugs_code", "smokes_code", "drink_code", "income"])
bty_data["body_type_code"] = df["body_type_code"]
bty_data = bty_data.dropna()

x_train, x_test, y_train, y_test = train_test_split(bty_data[["age", "diet_code", "sex_code", "drugs_code", "smokes_code", "drink_code", "income"]], bty_data["body_type_code"], train_size=0.8, test_size=0.2, random_state = 78)

scores = []
max = 0
max_i = 0
for i in range(1, 300):
    classifier = KNeighborsRegressor(n_neighbors = i)
    classifier.fit(x_train, y_train)
    score = classifier.score(x_test, y_test)
    scores.append(score)
    if score > max:
        max = score
        max_i = i
    print(str(i) + " " + str(score) + " " + str(max) + " " + str(max_i))

plt.plot(list(range(1, 300)), scores)
plt.xlabel("N Neighbors")
plt.ylabel("Accuracy")
plt.show()

start = timeit.default_timer()

classifier = KNeighborsClassifier(n_neighbors = 67)
classifier.fit(x_train, y_train)
guesses = classifier.predict(x_test)

stop = timeit.default_timer()

run = stop - start

print(run)


start = timeit.default_timer()

model = LinearRegression()
model.fit(x_train, y_train)
print(model.score(x_test, y_test))

stop = timeit.default_timer()

print(stop - start)

start = timeit.default_timer()

model = KNeighborsRegressor(n_neighbors = 134)
model.fit(x_train, y_train)
print(model.score(x_test, y_test))

stop = timeit.default_timer()

print(stop - start)
