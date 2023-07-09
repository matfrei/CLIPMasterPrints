import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#TODO: comment adjustment of parameters back in
#TODO: commented out for now as it does not play well with pycharm remote debugging
fontsize = 8
params = {'backend': 'ps',
          'axes.labelsize': fontsize,
          'axes.titlesize': fontsize,
          'font.size': fontsize,
          'legend.fontsize': fontsize,
          'xtick.labelsize': fontsize,
          'ytick.labelsize': fontsize}

def plot_similarity_heatmap(captions,images,similarities,path):
    #plt.rcParams.update(params)
    count = len(captions)
    print(len(images))
    plt.figure(figsize=(7, 5.4))
    plt.imshow(similarities, vmin=0.1, vmax=0.3)
    plt.yticks(range(count), captions)
    plt.xticks([])
    for i, image in enumerate(images):
        plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
    for x in range(similarities.shape[1]):
        for y in range(similarities.shape[0]):
            plt.text(x, y, f"{similarities[y, x]:.2f}", ha="center", va="center")

    for side in ["left", "top", "right", "bottom"]:
        plt.gca().spines[side].set_visible(False)

    plt.xlim([-0.5, len(images) - 0.5])
    plt.ylim([count + 0.5, -2])

    plt.tight_layout()
    plt.show()
    plt.savefig(path, bbox_inches='tight')

def scatter_optimized_classes(opt_captions,sim_fooling,sim_imagenet_opt, path):
    df_input = [(key, inner_value, 'ImageNet') for key, value_lst in sim_imagenet_opt.items() for inner_value in
                value_lst]

    df_sim_imagenet = pd.DataFrame(df_input, columns=['class','CLIP Score','type'])
    df_sim_fooling = pd.DataFrame(zip(opt_captions,sim_fooling.flatten(order='F').tolist()),columns=['class', 'CLIP Score'])
    df_sim_fooling['type'] = len(opt_captions) * ['Fooling image']

    df_sim = pd.concat([df_sim_imagenet, df_sim_fooling])
    df_sim['class (short)'] = df_sim['class'].str.split(',').apply(lambda x: x[0])

    sns.set_context("paper")
    #plt.rcParams.update(params)

    plt.figure()
    sns_plot = sns.catplot(data=df_sim, kind="strip", x="class (short)", y="CLIP Score", hue='type',
                           palette=sns.color_palette('colorblind', n_colors=2), legend=False, height=4, aspect=0.875)
    _ = plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()
    sns_plot.savefig(path, dpi=300)


def scatter_optimized_classes_multi(opt_captions,sim_fooling_cma,sim_fooling_grad,sim_imagenet_opt, path):
    df_input = [(key, inner_value, 'ImageNet') for key, value_lst in sim_imagenet_opt.items() for inner_value in
                value_lst]

    df_sim_imagenet = pd.DataFrame(df_input, columns=['class','CLIP Score','type'])
    df_sim_fooling_cma = pd.DataFrame(zip(opt_captions,sim_fooling_cma.flatten(order='F').tolist()),columns=['class', 'CLIP Score'])
    df_sim_fooling_cma['type'] = len(opt_captions) * ['Fooling image CMA']

    df_sim_fooling_grad = pd.DataFrame(zip(opt_captions,sim_fooling_grad.flatten(order='F').tolist()),columns=['class', 'CLIP Score'])
    df_sim_fooling_grad['type'] = len(opt_captions) * ['Fooling image SGD']

    df_sim = pd.concat([df_sim_imagenet, df_sim_fooling_cma,df_sim_fooling_grad])
    df_sim.reset_index(inplace=True)
    df_sim['class (short)'] = df_sim['class'].str.split(',').apply(lambda x: x[0])
    df_sim.to_json('cma_vs_gradient.json')

    sns.set_context("paper")
    #plt.rcParams.update(params)

    plt.figure()
    sns_plot = sns.catplot(data=df_sim, kind="strip", x="class (short)", y="CLIP Score", hue='type',
                           palette=sns.color_palette('colorblind', n_colors=3), legend=False, height=4, aspect=0.875)
    _ = plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()
    sns_plot.savefig(path, dpi=300)