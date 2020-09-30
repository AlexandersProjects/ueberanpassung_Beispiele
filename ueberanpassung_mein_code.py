# Mein utils modul:
import numpy as np


rng = np.random.RandomState(1)


def true_function(x):
    f = - 2 * x * np.cos(x) + 0.5* x**2 - 5
    # f = x * np.sin(5*x**2) - 2 * x * np.cos(x) + 0.5* x**2 - 5
    # f = x * np.sin(0.1 * x**2) - 2 * x + 0.5* x**2 - 5 + 2 * np.cos(0.1*x)
    return f

def get_train_data(n_samples=50):
    # x = np.random.uniform(-5, 5, size=n_samples)
    x = np.random.uniform(0, 10, size=n_samples)
    f = true_function(x)
    noise = np.random.normal(loc=0.0, scale=5.0, size=x.shape[0])
    y = f + noise
    # data = Data(x=x, y=y)
    return (x, y)


def get_test_data(n_samples=1000, mode=1):
    rng = np.random.RandomState(5)
    if mode==1:
        x = rng.uniform(0, 10, size=n_samples)
    elif mode==2:
        x = rng.uniform(-2, 12, size=n_samples)
    f = true_function(x)
    noise = rng.normal(loc=0, scale=5.0, size=x.shape[0])
    y = f + noise
    return x, y

""" Hier fängt das Beispiel mit einem linearen Modell an"""

# Meine Imports
import utils
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression as lr
from sklearn.pipeline import make_pipeline, Pipeline
import numpy as np
import matplotlib.pyplot as plt

### Convenience Functions
# Visualisierung eines Fits
def visualize_predict(model, xmin=0, xmax=10):
    ax = plt.gca()
    xx = np.linspace(0, 10, 100)
    ax.plot(xx, model.predict(xx[:, np.newaxis]), "orange")

# Quadratischer Fehler
def quadratic_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# Absoluter Fehler
def absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# Daten erstellen und umformen
x_train,y_train = utils.get_train_data()
x_train = x_train[:, np.newaxis]

x_test, y_test = utils.get_test_data()
x_test = x_test[:, np.newaxis]

# Lineare Regression
model = lr()
model.fit(x_train,y_train)

### Fehler berechnen
y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)

# Trainingsfehler
quadratic_train_error = quadratic_error(y_train, y_pred_train)
absolute_train_error = absolute_error(y_train, y_pred_train)

# Testfehler
quadratic_test_error = quadratic_error(y_test, y_pred_test)
absolute_test_error = absolute_error(y_test, y_pred_test)

### Visualisierung
# Trainingsdaten
plt.scatter(x_train, y_train)

# Vorhersagefunktion
visualize_predict(model)
plt.title("Lineare Regression");

print(f"Quadratischer Trainingsfehler: {quadratic_train_error:.2f} | Quadratischer Testfehler {quadratic_test_error:.2f}")
print(f"Absoluter Trainingsfehler: {absolute_train_error:.2f} | Absoluter Testfehler: {absolute_test_error:.2f}")


""" Fuer ein quadratisches Modell: """

### Modell mit make_pipeline erstellen und fitten
quadratic_regression = make_pipeline(PolynomialFeatures(include_bias=False, degree=2), LinearRegression())
quadratic_regression.fit(x_train, y_train)

### Vorhersagen
# Trainingsdaten vorhersagen
y_pred_train = quadratic_regression.predict(x_train)
# Testdaten vorhersagen
y_pred_test = quadratic_regression.predict(x_test)

### Fehler berechnen
# Trainingsfehler
quadratic_train_error = quadratic_error(y_train, y_pred_train)
absolute_train_error = absolute_error(y_train, y_pred_train)

# Testfehler
quadratic_test_error = quadratic_error(y_test, y_pred_test)
absolute_test_error = absolute_error(y_test, y_pred_test)


### Visualisierung
# Trainingsdaten
plt.scatter(x_train, y_train)

# Vorhersagefunktion
visualize_predict(quadratic_regression)
plt.title("Quadratische Regression");

print(f"Quadratischer Trainingsfehler: {quadratic_train_error:.2f} | Quadratischer Testfehler {quadratic_test_error:.2f}")
print(f"Absoluter Trainingsfehler: {absolute_train_error:.2f} | Absoluter Testfehler: {absolute_test_error:.2f}")

""" Beispiel für eine Kubische Regression mit Regularisierung und Skalierung: """

# für eine Regression mit Regularisierung benutze `Ridge`
from sklearn.linear_model import Ridge


cubic_ridge_regression = Pipeline([
    ("poly_expansion", PolynomialFeatures(include_bias=False, degree=3)),
    ("scaling", StandardScaler()),
    ("ridge_regression", Ridge())
])

cubic_ridge_regression.fit(x_train, y_train)

y_pred_train = cubic_ridge_regression.predict(x_train)
y_pred_test = cubic_ridge_regression.predict(x_test)

print(quadratic_error(y_pred_train, y_train))
print(quadratic_error(y_pred_test, y_test))

### Visualisierung
# Trainingsdaten
plt.scatter(x_train, y_train)

# Vorhersagefunktion
visualize_predict(cubic_ridge_regression)
plt.title("Kubische Regression");
