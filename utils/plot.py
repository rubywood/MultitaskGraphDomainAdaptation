import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os

from sklearn.metrics import precision_score, f1_score, confusion_matrix

from .utils import mkdir

def resp_label_names(resp):
    if resp == 'response_cr_nocr':
        return ['NoCR', 'CR']
    elif resp in ['CMS', 'CMS_matching']:
        return ['CMS1', 'CMS2', 'CMS3', 'CMS4']
    elif any(v in resp for v in ['CMS1','CMS2','CMS3','CMS4','epitheli']):
        return [f'Not {resp}', resp]
    else:
        return None


def plot_violin(targets, predictions, resp):
    pos_resp_preds = [item[1] for item in zip(targets, predictions) if item[0] == 1]
    neg_resp_preds = [item[1] for item in zip(targets, predictions) if item[0] == 0]

    mpl.rcParams["figure.dpi"] = 100
    plt.figure(figsize=(6, 4))
    plt.violinplot([neg_resp_preds, pos_resp_preds])
    plt.xticks([1, 2], resp_label_names(resp))
    plt.title(resp)
    plt.ylabel('Prediction')
    plt.show()


def plot_confusion_matrix(targets, predictions, resp, viz_fold=0, viz_epoch=None, save=True,
                          save_img_path=None, thresh=False):
    label_names = resp_label_names(resp)
    cm = confusion_matrix(targets, predictions)
    cm_df = pd.DataFrame(cm,
                         index=label_names,
                         columns=label_names)
    plt.figure(figsize=(7, 5))
    g = sns.heatmap(cm_df, annot=True, fmt='g', cmap='Blues')
    #fig = g.get_figure()
    fig = plt.gcf()
    # or fig = on the plt.figure()
    plt.title(f'Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')

    if save:
        print('Saving figure')
        mkdir(os.path.join(save_img_path, resp))
        fig.savefig(os.path.join(save_img_path, resp,
                                 f'confusion_matrix{"_thresh" if thresh else ""}_{resp}_fold{viz_fold:02d}_epoch{viz_epoch}.png'))
    plt.show()
    return fig
    #    if LOG:
    #        val_summary_writer.add_figure(f'Validation Confusion Matrix{" with Threshold" if thresh else ""} - {resp}',
    #                                      fig)
    #plt.show()


# binary classification only
def density_plot(targets, outputs, resp, viz_fold=0, viz_epoch=None,  save=True, save_img_path=None):
    nocr_idx = [idx for idx, elt in enumerate(targets) if elt == 0]
    nocr_probs = [outputs[idx] for idx in nocr_idx]
    cr_idx = [idx for idx, elt in enumerate(targets) if elt == 1]
    cr_probs = [outputs[idx] for idx in cr_idx]
    label_names = resp_label_names(resp)

    plt.figure(figsize=(6, 4))
    sns.set_style('whitegrid')
    g = sns.kdeplot(nocr_probs, bw_adjust=0.5, label=label_names[0], c=list(colors.TABLEAU_COLORS.values())[0])
    g = sns.kdeplot(cr_probs, bw_adjust=0.5, label=label_names[1], c=list(colors.TABLEAU_COLORS.values())[1])
    g.set(xlim=(0, 1))
    g.set_title(f'Density plot of predicted {resp} probabilties across true labels')
    g.legend()
    #fig = g.get_figure()
    fig = plt.gcf()

    if save:
        print('Saving figure')
        mkdir(os.path.join(save_img_path, resp))
        fig.savefig(os.path.join(save_img_path, resp,
                                 f'prediction_density_{resp}_fold{viz_fold:02d}_epoch{viz_epoch}.png'))
    plt.show()
    return fig
    #    if LOG:
    #        val_summary_writer.add_figure(f'Validation Density Plot - {resp}', fig)
    #plt.show()