import os
#https://stackoverflow.com/questions/37604289/tkinter-tclerror-no-display-name-and-no-display-environment-variable/43592515
#for running the pipeline through SSH
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import pdb

import pandas as pd
import argparse
import numpy as np
import os
from helper_functions import create_folder,write_results_to_json_only
from tqdm import tqdm
import seaborn as sns; sns.set()
sns.set(style="whitegrid")
import json
import math
from scipy import optimize
from collections import defaultdict,Counter
import pwlf

# def split_dataframe(dataset_df):
#     try:
#       dataset_df[['dataset','split','model','loss','sampler']] = dataset_df.Experiment.str.split("_",expand=True)
#     except:
#       dataset_df[['dataset','type']] = dataset_df.Experiment.str.split("_",expand=True) #removed loss for fb

#       # dataset_df[['dataset','split','model','sampler']] = dataset_df.Experiment.str.split("_",expand=True) #removed loss for fb
#     return dataset_df


def split_dataframe(dataset_df):
    # try:
    dataset_df[['dataset','type']] = dataset_df.Experiment.str.split("_",expand=True)
    # except:
      
    #   pdb.set_trace()

      # dataset_df[['dataset','split','model','sampler']] = dataset_df.Experiment.str.split("_",expand=True) #removed loss for fb
    return dataset_df

def range_for_dataset(dataset_df,exp_name):
    # tokens_split = exp_name.split('_')
    # dataset_df = split_dataframe(dataset_df)
    dframe_cluster_plots = dataset_df #.loc[dataset_df['dataset'] == tokens_split[0]]#and dataset_df['model'] in exp_name]
    axis_min = round(dframe_cluster_plots['L_SDash'].min(),0)-1
    if (axis_min<0):
      axis_min=0
    axis_max = round(dframe_cluster_plots['L_SDash'].max(),0)+1
    return axis_max,axis_min

def graph_for_sampling_NBP(dframe_data,path_to_save,graphs_jump):
    diffs_dataset = []
    create_folder(path_to_save)
    datasets = dframe_data["Experiment"].unique()

    for data in datasets:
      exp_name = data
      dframe_cluster_plots = dframe_data.loc[dframe_data['Experiment'] == exp_name]
      topics = dframe_cluster_plots['Topics'].unique()

      plt.figure(figsize=(40,5))
      # plot_range = len(topics)
      plot_range = topics #[i for i in range(int(topics.min()),int(topics.max()),graphs_jump)]
      axis_max,axis_min = range_for_dataset(dframe_data,exp_name)
      for j in range(2):
        # if (j==0):
        if ("NBP" in data and j==0):
          exp_name= data#+"_lda_all_cluster"
          dframe_cluster_plots = dframe_data.loc[dframe_data['Experiment'] == exp_name]

          for i,graph_index in zip(plot_range,range(len(plot_range))):
            #plt.subplot(2, len(topics), i+plot_range+1)#i*(j+1)+1)
            plt.subplot(1, len(plot_range), graph_index+1)
            topic_size = i #topics[i]

            dframe_cluster_plot = dframe_cluster_plots.loc[dframe_cluster_plots['Topics'] == topic_size]

            # N_Median = dframe_cluster_plot['N_Median'].iloc[0]
            L_S = dframe_cluster_plot['L_S'].iloc[0]
            Fraction = dframe_cluster_plot['Fraction'].iloc[0]
            plot_mean = dframe_cluster_plot["L_SDash"].mean()

            plot_stdev = dframe_cluster_plot["L_SDash"].std()
            diff_mean = L_S - plot_mean 

            #difference = round(Fraction-0.5,2)
            if plot_stdev==0.0:
              difference=diff_mean
            else:
              difference = (diff_mean/plot_stdev)
            
            plot_title = "T = "+str(topic_size)#+" Diff="+ str(difference)
            sx = sns.distplot(dframe_cluster_plot["L_SDash"].values).set_title(plot_title)
            #plt.hist(dframe_cluster_plot["L_SDash"].values, normed=True,color="blue")
            plt.title(plot_title)
            plt.axvline(x=L_S, color = "red",ls='-',label="KL")
            diff_value = {"Experiment":str(data),"Topics":topic_size,"Difference":(difference)} #"N_Median":(N_Median)}

            if (i!=0):
              diffs_dataset.append(diff_value)
            axes = plt.gca()
            axes.set_xlim([axis_min,axis_max])
    
            #plt.axvline(x=plot_mean, color = "black",ls='-',label="KL")
        elif ("bootstrap" in data and j!=0):
          exp_name= data#+"_lda_all_bootstrap"
          dframe_cluster_plots = dframe_data.loc[dframe_data['Experiment'] == exp_name]
          for i,graph_index in zip(plot_range,range(len(plot_range))):
            plt.subplot(1, len(plot_range), graph_index+1)
            topic_size = i
            dframe_cluster_plot = dframe_cluster_plots.loc[dframe_cluster_plots['Topics'] == topic_size]
            plot_title = "Topics = "+str(topic_size)
            L_S = dframe_cluster_plot['L_S'].iloc[0]
            Fraction = dframe_cluster_plot['Fraction'].iloc[0]
            plot_mean = dframe_cluster_plot["L_SDash"].mean()

            plot_stdev = dframe_cluster_plot["L_SDash"].std()
            diff_mean = L_S - plot_mean
            #difference = round(Fraction-0.5,2)
            if plot_stdev==0.0:
              difference=diff_mean
            else:
              difference = (diff_mean/plot_stdev)
            plot_title = "T = "+str(topic_size)#+" Diff="+ str(difference)
            sx = sns.distplot(dframe_cluster_plot["L_SDash"].values,color="green").set_title(plot_title)
            #plt.hist(dframe_cluster_plot["L_SDash"].values, normed=True,color="green")
            plt.title(plot_title)
            plt.axvline(x=L_S, color = "red",ls='-',label="KL")
            plot_mean = dframe_cluster_plot["L_SDash"].mean()
            diff_value = {"Experiment":str(data),"Topics":topic_size,"Difference":(difference)}
            if (i!=0):
              diffs_dataset.append(diff_value)
            axes = plt.gca()
            axes.set_xlim([axis_min,axis_max])
            #plt.axvline(x=plot_mean, color = "black",ls='-',label="KL")
      exp_name = data
      plt.savefig(path_to_save+'/'+exp_name+'.pdf')
      print ("Graph Saved to "+path_to_save)
    return pd.DataFrame(diffs_dataset)

def calculate_diff_NBP(dframe_data,path_to_save,graphs_jump):
    diffs_dataset = []
    create_folder(path_to_save)
    datasets = dframe_data["Experiment"].unique()

    for data in datasets:
      exp_name = data
      dframe_cluster_plots = dframe_data.loc[dframe_data['Experiment'] == exp_name]
      topics = dframe_cluster_plots['Topics'].unique()

      plt.figure(figsize=(40,5))
      # plot_range = len(topics)
      plot_range = topics #[i for i in range(int(topics.min()),int(topics.max()),graphs_jump)]
      axis_max,axis_min = range_for_dataset(dframe_data,exp_name)
      for j in range(2):
        # if (j==0):
        if ("NBP" in data and j==0):
          exp_name= data#+"_lda_all_cluster"
          dframe_cluster_plots = dframe_data.loc[dframe_data['Experiment'] == exp_name]

          for i,graph_index in zip(plot_range,range(len(plot_range))):
            #plt.subplot(2, len(topics), i+plot_range+1)#i*(j+1)+1)
            plt.subplot(1, len(plot_range), graph_index+1)
            topic_size = i #topics[i]

            dframe_cluster_plot = dframe_cluster_plots.loc[dframe_cluster_plots['Topics'] == topic_size]

            # N_Median = dframe_cluster_plot['N_Median'].iloc[0]
            L_S = dframe_cluster_plot['L_S'].iloc[0]
            Fraction = dframe_cluster_plot['Fraction'].iloc[0]
            plot_mean = dframe_cluster_plot["L_SDash"].mean()

            plot_stdev = dframe_cluster_plot["L_SDash"].std()
            diff_mean = L_S - plot_mean 

            #difference = round(Fraction-0.5,2)
            if plot_stdev==0.0:
              difference=diff_mean
            else:
              difference = (diff_mean/plot_stdev)
            
            plot_title = "T = "+str(topic_size)#+" Diff="+ str(difference)
            sx = sns.distplot(dframe_cluster_plot["L_SDash"].values).set_title(plot_title)
            #plt.hist(dframe_cluster_plot["L_SDash"].values, normed=True,color="blue")
            plt.title(plot_title)
            plt.axvline(x=L_S, color = "red",ls='-',label="KL")
            diff_value = {"Experiment":str(data),"Topics":topic_size,"Difference":(difference)} #"N_Median":(N_Median)}

            if (i!=0):
              diffs_dataset.append(diff_value)
            axes = plt.gca()
            axes.set_xlim([axis_min,axis_max])
    
            #plt.axvline(x=plot_mean, color = "black",ls='-',label="KL")
        elif ("bootstrap" in data and j!=0):
          exp_name= data#+"_lda_all_bootstrap"
          dframe_cluster_plots = dframe_data.loc[dframe_data['Experiment'] == exp_name]
          for i,graph_index in zip(plot_range,range(len(plot_range))):
            plt.subplot(1, len(plot_range), graph_index+1)
            topic_size = i
            dframe_cluster_plot = dframe_cluster_plots.loc[dframe_cluster_plots['Topics'] == topic_size]
            plot_title = "Topics = "+str(topic_size)
            L_S = dframe_cluster_plot['L_S'].iloc[0]
            Fraction = dframe_cluster_plot['Fraction'].iloc[0]
            plot_mean = dframe_cluster_plot["L_SDash"].mean()

            plot_stdev = dframe_cluster_plot["L_SDash"].std()
            diff_mean = L_S - plot_mean
            #difference = round(Fraction-0.5,2)
            if plot_stdev==0.0:
              difference=diff_mean
            else:
              difference = (diff_mean/plot_stdev)
            plot_title = "T = "+str(topic_size)#+" Diff="+ str(difference)
            sx = sns.distplot(dframe_cluster_plot["L_SDash"].values,color="green").set_title(plot_title)
            #plt.hist(dframe_cluster_plot["L_SDash"].values, normed=True,color="green")
            plt.title(plot_title)
            plt.axvline(x=L_S, color = "red",ls='-',label="KL")
            plot_mean = dframe_cluster_plot["L_SDash"].mean()
            diff_value = {"Experiment":str(data),"Topics":topic_size,"Difference":(difference)}
            if (i!=0):
              diffs_dataset.append(diff_value)
            axes = plt.gca()
            axes.set_xlim([axis_min,axis_max])
            #plt.axvline(x=plot_mean, color = "black",ls='-',label="KL")
      exp_name = data
      plt.savefig(path_to_save+'/'+exp_name+'.pdf')
      print ("Graph Saved to "+path_to_save)
    return pd.DataFrame(diffs_dataset)

def graph_for_sampling_cluster(dframe_data,path_to_save,graphs_jump):
    diffs_dataset = []
    create_folder(path_to_save)
    datasets = dframe_data["Experiment"].unique()
    sampler = dframe_data["Sampler"].unique()

    for data in datasets:
      exp_name = data#+"_lda_all_cluster"
      dframe_cluster_plots = dframe_data.loc[dframe_data['Experiment'] == exp_name]
      topics = dframe_cluster_plots['Topics'].unique()
      plt.figure(figsize=(40,5))
      plot_range = [i for i in range(int(topics.min()),int(topics.max()),graphs_jump)]
      axis_max,axis_min = range_for_dataset(dframe_data,exp_name)
      for j in range(2):
        # if (j==0):
        if ("cluster" in sampler[0] and j==0):
          exp_name= data#+"_lda_all_cluster"
          dframe_cluster_plots = dframe_data.loc[dframe_data['Experiment'] == exp_name]

          for i,graph_index in zip(plot_range,range(len(plot_range))):
            #plt.subplot(2, len(topics), i+plot_range+1)#i*(j+1)+1)
            plt.subplot(1, len(plot_range), graph_index+1)
            topic_size = i #topics[i]
            dframe_cluster_plot = dframe_cluster_plots.loc[dframe_cluster_plots['Topics'] == topic_size]

            try:
              L_S = dframe_cluster_plot['L_S'].iloc[0]
            except:
              pdb.set_trace()
            Fraction = dframe_cluster_plot['Fraction'].iloc[0]
            plot_mean = dframe_cluster_plot["L_SDash"].mean()

            plot_stdev = dframe_cluster_plot["L_SDash"].std()
            diff_mean = L_S - plot_mean 
            #difference = round(Fraction-0.5,2)
            if plot_stdev==0.0:
              difference=diff_mean
            else:
              difference = (diff_mean/plot_stdev)
            
            plot_title = "T = "+str(topic_size)#+" Diff="+ str(difference)
            sx = sns.distplot(dframe_cluster_plot["L_SDash"].values).set_title(plot_title)
            #plt.hist(dframe_cluster_plot["L_SDash"].values, normed=True,color="blue")
            plt.title(plot_title)
            plt.axvline(x=L_S, color = "red",ls='-',label="KL")
            diff_value = {"Experiment":str(data),"Topics":topic_size,"Difference":(difference),"Sampler":sampler[0]}
            diffs_dataset.append(diff_value)
            axes = plt.gca()
            axes.set_xlim([axis_min,axis_max])
 
            #plt.axvline(x=plot_mean, color = "black",ls='-',label="KL")
        elif ("bootstrap" in sampler[0] and j!=0):
          exp_name= data#+"_lda_all_bootstrap"
          dframe_cluster_plots = dframe_data.loc[dframe_data['Experiment'] == exp_name]
          for i,graph_index in zip(plot_range,range(len(plot_range))):
            plt.subplot(1, len(plot_range), graph_index+1)
            topic_size = i #topics[i]
            dframe_cluster_plot = dframe_cluster_plots.loc[dframe_cluster_plots['Topics'] == topic_size]
            plot_title = "Topics = "+str(topic_size)
            L_S = dframe_cluster_plot['L_S'].iloc[0]
            Fraction = dframe_cluster_plot['Fraction'].iloc[0]
            plot_mean = dframe_cluster_plot["L_SDash"].mean()

            plot_stdev = dframe_cluster_plot["L_SDash"].std()
            diff_mean = L_S - plot_mean
            #difference = round(Fraction-0.5,2)
            if plot_stdev==0.0:
              difference=diff_mean
            else:
              difference = (diff_mean/plot_stdev)
            plot_title = "T = "+str(topic_size)#+" Diff="+ str(difference)
            sx = sns.distplot(dframe_cluster_plot["L_SDash"].values,color="green").set_title(plot_title)
            #plt.hist(dframe_cluster_plot["L_SDash"].values, normed=True,color="green")
            plt.title(plot_title)
            plt.axvline(x=L_S, color = "red",ls='-',label="KL")
            plot_mean = dframe_cluster_plot["L_SDash"].mean()
            diff_value = {"Experiment":str(data),"Topics":topic_size,"Difference":(difference),"Sampler":sampler[0]}
            diffs_dataset.append(diff_value)
            axes = plt.gca()
            axes.set_xlim([axis_min,axis_max])
            #plt.axvline(x=plot_mean, color = "black",ls='-',label="KL")
      exp_name = data
      plt.savefig(path_to_save+'/'+exp_name+'.pdf')
      print ("Graph Saved to "+path_to_save)
    return pd.DataFrame(diffs_dataset)

def graph_for_sampling(dframe_data,path_to_save):
    diffs_dataset = []
    datasets = dframe_data["Experiment"].unique()
    create_folder(path_to_save)
    for data in datasets:
      exp_name = data#+"_lda_all_cluster"
      dframe_cluster_plots = dframe_data.loc[dframe_data['Experiment'] == exp_name]
      topics = dframe_cluster_plots['Topics'].unique()
      plt.figure(figsize=(40,5))
      plot_range = len(topics)

      for j in range(2):
        # if (j==0):
        if ("cluster" in data and j==0):
          exp_name= data#+"_lda_all_cluster"
          dframe_cluster_plots = dframe_data.loc[dframe_data['Experiment'] == exp_name]
          for i in range(plot_range):
            #plt.subplot(2, len(topics), i+plot_range+1)#i*(j+1)+1)
            plt.subplot(1, len(topics), i+1)
            topic_size = topics[i]
            dframe_cluster_plot = dframe_cluster_plots.loc[dframe_cluster_plots['Topics'] == topic_size]

            L_S = dframe_cluster_plot['L_S'].iloc[0]
            plot_mean = dframe_cluster_plot["L_SDash"].mean()
            difference = round(plot_mean-L_S,2)
            plot_title = "T = "+str(topic_size)+" Diff="+ str(difference)
            #sx = sns.distplot(dframe_cluster_plot["L_SDash"].values).set_title(plot_title)
            plt.hist(dframe_cluster_plot["L_SDash"].values, normed=True,color="blue")
            plt.title(plot_title)
            plt.axvline(x=L_S, color = "red",ls='-',label="KL")
            diff_value = {"Experiment":str(data),"Topics":topic_size,"Difference":abs(difference)}
            diffs_dataset.append(diff_value)
            #plt.axvline(x=plot_mean, color = "black",ls='-',label="KL")
        elif ("bootstrap" in data and j!=0):
          exp_name= data#+"_lda_all_bootstrap"
          dframe_cluster_plots = dframe_data.loc[dframe_data['Experiment'] == exp_name]
          for i in range(len(topics)):
            plt.subplot(1, len(topics), i+1)
            topic_size = topics[i]
            dframe_cluster_plot = dframe_cluster_plots.loc[dframe_cluster_plots['Topics'] == topic_size]
            plot_title = "Topics = "+str(topic_size)
            L_S = dframe_cluster_plot['L_S'].iloc[0]
            plot_mean = dframe_cluster_plot["L_SDash"].mean()
            difference = round(plot_mean-L_S,2)
            plot_title = "T = "+str(topic_size)+" Diff="+ str(difference)
            # sx = sns.distplot(dframe_cluster_plot["L_SDash"].values,color="green").set_title(plot_title)
            plt.hist(dframe_cluster_plot["L_SDash"].values, normed=True,color="green")
            plt.title(plot_title)
            plt.axvline(x=L_S, color = "red",ls='-',label="KL")
            plot_mean = dframe_cluster_plot["L_SDash"].mean()
            diff_value = {"Experiment":str(data),"Topics":topic_size,"Difference":abs(difference)}
            diffs_dataset.append(diff_value)
            #plt.axvline(x=plot_mean, color = "black",ls='-',label="KL")
      exp_name = data
      plt.savefig(path_to_save+'/'+exp_name+'.pdf')
      print ("Graph Saved to "+path_to_save)
    return pd.DataFrame(diffs_dataset)

def difference_histograms(dframe_data,path_to_save,x_label):
  create_folder(path_to_save)

  # dframe_data = split_dataframe(dframe_data)
  # dframe_data[['dataset','split','model','loss','sampler']] = dframe_data.Experiment.str.split("_",expand=True) 
  # dframe_data[['dataset','model','split','sampler']] = dframe_data.Experiment.str.split("_",expand=True) #removing loss for fb

  datasets = dframe_data["Experiment"].unique()
  samplers = dframe_data["Sampler"].unique()
  # models = dframe_data["model"].unique()
  # colors = ["blue","orange","green","red"]
  for sampler in samplers:
    dframe_cluster_plots_sampler = dframe_data.loc[dframe_data['Sampler'] == sampler]
    for data in datasets:
      dframe_cluster_plots = dframe_cluster_plots_sampler.loc[dframe_cluster_plots_sampler['Experiment'] == data]
      # dframe_cluster_test = dframe_cluster_plots.loc[dframe_cluster_plots['model'] == "FMM"]
      # plt.figure(figsize=(9,7))
      # topics = dframe_cluster_plots['Topics'].unique()
      # title = sampler+" sampler "+data
      # ax = sns.lineplot(x="Topics", y="Difference",hue="model", data=dframe_cluster_plots).set_title(title)
      # exp_name = sampler+"_sampler_"+data
      # plt.xlabel(x_label)
      # for model,plt_color in zip(models,colors):
      dframe_cluster_model = dframe_cluster_plots.loc[dframe_cluster_plots['Experiment'] == data]
      max_values = dframe_cluster_model.min()
      max_row = dframe_cluster_model.loc[dframe_cluster_model['Difference'] == max_values['Difference']]
      max_topics = max_row['Topics'].to_numpy()
      max_topics = max_topics[0]
      # label_line = data + " Min("+str(max_topics)+")"
      #print (label_line)
      # plt.axvline(x=max_topics,ls='--',color = colors[0], label=label_line)
      # plt.legend()
      # plt.savefig(path_to_save+'/'+exp_name+'.pdf')
      # plt.show()

  return max_topics,datasets[0]

def difference_histograms_NBP(dframe_data,path_to_save,x_label,n_pieces):
  create_folder(path_to_save)
  dframe_data = split_dataframe(dframe_data)
  # dframe_data[['dataset','split','model','loss','sampler']] = dframe_data.Experiment.str.split("_",expand=True) 
  datasets = dframe_data["dataset"].unique()
  samplers = "NBP" #dframe_data["sampler"].unique()
  models = dframe_data["model"].unique()
  colors = ["blue","orange","green","red"]
  for sampler in samplers:
    dframe_cluster_plots_sampler = dframe_data.loc[dframe_data['sampler'] == sampler]
    for data in datasets:
      dframe_cluster_plots = dframe_cluster_plots_sampler.loc[dframe_cluster_plots_sampler['dataset'] == data]
      # dframe_cluster_test = dframe_cluster_plots.loc[dframe_cluster_plots['model'] == "FMM"]
      plt.figure(figsize=(9,7))
      topics = dframe_cluster_plots['Topics'].unique()
      title = sampler+" sampler "+data

      ax = sns.lineplot(x="Topics", y="Difference",hue="model", data=dframe_cluster_plots).set_title(title)
      exp_name = sampler+"_sampler_"+data
      plt.xlabel(x_label)
      for model,plt_color in zip(models,colors):
         dframe_cluster_model = dframe_cluster_plots.loc[dframe_cluster_plots['model'] == model]
         max_values = dframe_cluster_model.min()
         max_row = dframe_cluster_model.loc[dframe_cluster_model['Difference'] == max_values['Difference']]
         max_topics = max_row['Topics'].to_numpy()
         max_topics = max_topics[0]
         label_line = model + " Min("+str(max_topics)+")"
         #print (label_line)
         plt.axvline(x=max_topics,ls='--',color = plt_color, label=label_line)
      plt.legend()
      # Piecewise Linear Fit
      x = np.array(dframe_cluster_plots["Topics"])
      y = np.array(dframe_cluster_plots["Difference"])
      degree_list = [1,0]
      pwlf_obj = pwlf.PiecewiseLinFit(x, y,degree=degree_list)
      breaks = pwlf_obj.fit(n_pieces)
      break_point = round(breaks[1],1)
      x_hat = np.linspace(x.min(), x.max(), 100)
      y_hat = pwlf_obj.predict(x_hat)
      plt.plot(x_hat, y_hat, '-')
      # End of Linear Regression
      plt.savefig(path_to_save+'/'+exp_name+'.pdf')
    return break_point,model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_name", help="Database name",default=False)
    parser.add_argument("--col_name", help="Collections name",default=False)
    parser.add_argument("--path_to_save", help="Output Folder",default="Data/Trash")
    parser.add_argument("--db_to_save", help="Results database",default="del")
    parser.add_argument("--col_to_save", help="Collections to Save",default="del")
    parser.add_argument("--graphs", help="Plot Graphs",default=False)
    parser.add_argument("--json_log_file", help="Experimental Log Path",default=False)
    parser.add_argument("--model_selection", help="Model Selection Log Path",default=False)
    args = parser.parse_args()
    db_name = args.db_name
    col_name = args.col_name
    col_to_save = args.col_to_save
    output_db = args.db_to_save
    model_selection_path = args.model_selection
    # data_df = retrive_data_pooling(db_name,col_name)
    
    log_path = args.json_log_file #'data/jobQ1_BOTH/jobQ1_BOTH_log_FMMHY100_predict_only.json'

    data_df = pd.read_json(log_path)

    if (args.graphs):
      data_df_skip = data_df #retrive_data_pooling(db_name,col_name)
      # data_df.loc[(data_df.Experiment == 'fb_NBP_predict_all_NBP_predict'),'Experiment']='fb_nbp_all_NBP'
      # data_df_skip.loc[(data_df_skip.Experiment == 'fb_NBP_predict_all_NBP_predict'),'Experiment']='fb_nbp_all_NBP'

      if "NBP" in db_name or "nbp" in db_name:
        diffs_all_nbp_skips = graph_for_sampling_NBP(data_df_skip,args.path_to_save+"/Results",4)
        diffs_all_nbp = graph_for_sampling_NBP(data_df,args.path_to_save+"/Draft",1)
        diffs_all_nbp = diffs_all_nbp.dropna()
        break_point,model = difference_histograms_NBP(diffs_all_nbp,args.path_to_save+"/Diff","Radius",2)
      else:
        diffs_all_cluster_skips = graph_for_sampling_cluster(data_df_skip,args.path_to_save+"/Results",4)
        diffs_all_cluster = graph_for_sampling_cluster(data_df,args.path_to_save+"/Draft",1)
        break_point,model = difference_histograms(diffs_all_cluster,args.path_to_save+"/Diff","Radius")
    else:
      if "NBP" in db_name or "nbp" in db_name or "nbp" in col_name or "NBP" in col_name:
        diffs_all_nbp = graph_for_sampling_NBP(data_df,args.path_to_save+"/Draft",1)
        diffs_all_nbp = diffs_all_nbp.dropna()
        break_point,model = difference_histograms_NBP(diffs_all_nbp,args.path_to_save+"/Diff","Radius",2)
      else:
        diffs_all_cluster = graph_for_sampling_cluster(data_df,args.path_to_save+"/Draft",1)
        break_point,model = difference_histograms(diffs_all_cluster,args.path_to_save+"/Diff","Radius")
    results = {}
    results['exp_db'] = db_name
    results['model_selected'] = break_point.item()
    write_results_to_json_only(results,model_selection_path)

    #difference_histograms(df_diffs,args.path_to_save)

if __name__ == '__main__':
    main()
