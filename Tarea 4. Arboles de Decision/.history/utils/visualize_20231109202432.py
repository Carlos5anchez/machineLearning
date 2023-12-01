import matplotlib.pyplot as plt



def hist(df):


    # Plotting histograms for all columns
    df.hist(bins=50, figsize=(20, 15))
    plt.tight_layout()  # Adjust the subplots to fit into the figure area.
    plt.show()


def display3DChart(clusters=None,data=None,centroids=None,numClusters=None,indx_x=0,indx_y=1,indx_z=2):
    if clusters is None or data is None or centroids is None or numClusters is None:
        raise Exception('No se han pasado los parametros necesarios')
    
    # Crear una figura 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Visualizar los puntos de datos por cluster
    for cluster_id in range(numClusters):
        cluster_points = data[clusters == cluster_id]
        ax.scatter(cluster_points.iloc[:, indx_x], cluster_points.iloc[:, indx_y], cluster_points.iloc[:, indx_z], label=f'Cluster {cluster_id}', alpha=0.5)

    # Visualizar los centroides
    ax.scatter(centroids[:, indx_x], centroids[:, indx_y], centroids[:, indx_z], marker='X', s=100, c='red', label='Centroids')

    # Ajustar etiquetas y título
    ax.set_xlabel(f'Dimension {indx_x + 1}')
    ax.set_ylabel(f'Dimension {indx_y + 1}')
    ax.set_zlabel(f'Dimension {indx_z + 1}')
    ax.set_title('K-Means Clustering in 3D')

    # Mostrar leyenda
    ax.legend()

    # Mostrar e
    plt.pause(10000000000)