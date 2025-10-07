import numpy as np
import matplotlib.pyplot as plt

# ————— Funktion för PCA —————
def pca(X, num_components):
    # Steg 1: centra data
    X_centered = X - np.mean(X, axis=0)
    # Steg 2: kovariansmatris
    cov_matrix = np.cov(X_centered, rowvar=False)
    # Steg 3: egenvärden & egenvektorer
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    # Steg 4: sortera i fallande ordning
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # Steg 5: välj de num_components komponenterna
    components = eigenvectors[:, :num_components]
    # Steg 6: transformera data
    X_pca = np.dot(X_centered, components)
    return X_pca, components, eigenvalues

# ————— Funktion för K-means (egen implementation) —————
def kmeans(X, k, max_iters=100):
    n_samples, n_features = X.shape
    # Initiera centroids med slumpmässiga datapunkter
    centroids = X[np.random.choice(n_samples, k, replace=False)]
    labels = np.zeros(n_samples, dtype=int)

    for it in range(max_iters):
        # Tilldela kluster
        for i, x in enumerate(X):
            # räknar avstånd till varje centroid
            dists = np.linalg.norm(x - centroids, axis=1)
            labels[i] = np.argmin(dists)
        # Uppdatera centroids
        new_centroids = np.zeros((k, n_features))
        for j in range(k):
            cluster_points = X[labels == j]
            if len(cluster_points) > 0:
                new_centroids[j] = np.mean(cluster_points, axis=0)
            else:
                # om ett kluster är tomt, behåll gammal centroid
                new_centroids[j] = centroids[j]
        # Kontrollera konvergens
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids

    # Räkna antal objekt i varje kluster
    counts = [np.sum(labels == j) for j in range(k)]
    return centroids, counts, labels

# ————— Hjälpfunktion: läsa in data —————
def load_wine(filename='WINE.txt'):
    # Filen består av rader som: klass, följt av 13 features, separerade med mellanslag eller mellanslag och/eller blanktecken.
    # Vi använder np.loadtxt med delimiter = None (standard) som hanterar whitespace.
    data = np.loadtxt(filename)  # antar att filen inte har rubrikrad
    # De 14 kolumnerna: klass + 13 feature‑kolumner
    # Vi ska ignorera klasskolumnen, bara använda kolumn 1–13 (index 1 till 13)
    X = data[:, 1:]
    return X, data[:, 0]  # returnera även klasser om du vill analysera dem senare

# ————— Huvudprogram —————
def main():
    # Läs in
    X, classes = load_wine('WINE.txt')
    print("Dataform:", X.shape)  # Förväntat: (178, 13)

    # 1) PCA på alla 13 features, men vi kommer använda 8 komponenter som max för clusteringsteg
    X_pca_all, components_all, eigenvalues = pca(X, num_components=8)
    # components_all är en matris med form (13, 8) — varje kolumn är en principal komponent (koefficienter)
    # eigenvalues ger dig hur mycket varians varje komponent förklarar (i fallande ordning)
    print("\nPrincipal component coefficient matrix (13x8):")
    print(components_all)  # *** Här sparar/visar du alla 8 komponenter, viktigt enligt spec ***

    # --- SAKNAD DEL: Spara/skriv ut nya feature values för alla objekt --- #
    print("\nNya feature values för alla objekt enligt PCA alla rader:")
    print(X_pca_all[:13, :])  # *** Viktigt att spara/skriva ut nya features för alla objekt ***
    # Spara eller skriv ut komponentkoefficienterna:
    print("\nDe första två principal components (koefficienter):")
    print(components_all[:, :2])  # visar de två första kolumnerna

    # Visualisera objekten i top 2 komponenter
    X_pca_2 = X_pca_all[:, :2]
    plt.figure(figsize=(6, 5))
    plt.scatter(X_pca_2[:, 0], X_pca_2[:, 1], c=classes, cmap='viridis', s=50)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Wine data i top 2 principal components")
    plt.colorbar(label="Original klass (1, 2, 3)")
    plt.tight_layout()
    plt.show()

    # 2) K-means clustering med K = 3 på de nya komponenterna
    for num_comp in [2, 5, 8]:
        print(f"\n== K-means clustering med topp {num_comp} komponenter ==")
        X_sub = X_pca_all[:, :num_comp]
        centroids, counts, labels = kmeans(X_sub, 3)
        for j in range(3):
            print(f"Kluster {j+1}: centroid = {centroids[j]}, antal objekt = {counts[j]}")

        # (Valfritt) visualisera kluster för num_comp=2
        if num_comp == 2:
            plt.figure(figsize=(6, 5))
            for j in range(3):
                pts = X_sub[labels == j]
                plt.scatter(pts[:, 0], pts[:, 1], label=f"Kluster {j+1}")
            plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', color='black', s=100, label='Centroids')
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.title("K-means kluster i PCA‑rum (2 komponenter)")
            plt.legend()
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    main()
