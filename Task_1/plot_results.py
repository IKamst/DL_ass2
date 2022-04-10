import pandas as pd
from matplotlib import pyplot as plt


def make_loss_plot():
    # Read from csv
    data_style_loss = pd.read_csv('Results/k_comparison_npt.csv')
    data_content_loss = pd.read_csv('Results/k_comparison_pt.csv')
    data_total_loss = [sum(x) for x in zip(data_style_loss, data_content_loss)]
    k = range(1, len(data_content_loss))

    # HR@K plot
    plt.figure(figsize=(13,8))
    plt.plot(k, data_style_loss['HR'], label="NeuMF npt")
    plt.plot(k, data_content_loss['HR'], label='NeuMF pt')
    plt.legend(loc="upper left", prop={'size': 20})
    plt.xlabel('K', fontsize=24)
    plt.ylabel('HR@K', fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig('HR@Kplot.png')

    # NDCG@K plot
    plt.figure(figsize=(13,8))
    plt.plot(k, data_style_loss['NDCG'], label="NeuMF npt")
    plt.plot(k, data_content_loss['NDCG'], label='NeuMF pt')
    plt.legend(loc="upper left", prop={'size': 20})
    plt.xlabel('K', fontsize=24)
    plt.ylabel('NDCG@K', fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig('NDCG@Kplot.png')